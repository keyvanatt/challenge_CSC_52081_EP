import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from models.linear import CEMLinearPolicy, CEMSimpleLinearPolicy
from concurrent.futures import ThreadPoolExecutor
from student_client import create_student_gym_env, StudentGymEnv
from student_client.student_gym_env_vectorized import StudentGymEnvVectorized, create_student_gym_env_vectorized
from datetime import datetime
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import reshape_last_state
import logging
from typing import List


def evaluate_workers(thetas: List[np.ndarray], env: StudentGymEnvVectorized, agents:List[CEMLinearPolicy]) -> np.ndarray:
    """
    Simule un vol complet pour un vecteur de paramètres theta.
    Possède son propre environnement local pour éviter les collisions HTTP.
    """
    import logging
    logging.getLogger("student_gym_env").setLevel(logging.CRITICAL)
    
    nbr_workers = len(thetas)
    total_reward = np.zeros(nbr_workers)
    for agent, theta in zip(agents, thetas):
        agent.set_weights(theta)
    state, _ = env.reset()
    #state has shape (4, 9), we unsqueeze it to (4, 1, 9) and then tile it to (4, 10, 9) to match the expected input shape of the agents.
    state = np.expand_dims(state, axis=1)  # (4, 1, 9)
    state = np.tile(state, (1, 10, 1))  # (4, 10, 9)
    timestamp = np.tile(np.arange(state.shape[1]), (state.shape[0], 1))
    state = np.concatenate([state, np.expand_dims(timestamp, axis=2)], axis=2)  # (4, 10, 10)
    print(state.shape)
            
    done_i = [False] * nbr_workers
    
    while not all(done_i):
        actions = []
        for i, agent in enumerate(agents):
            if done_i[i]:
                actions.append(0)  # Action neutre pour les épisodes terminés
                continue
            action = agent(state[i])
            actions.append(action)
        actions = np.array(actions)
        print(f"Actions : {actions}", "Done : ", done_i)
        
        next_state, rewards, terminated, truncated, infos = env.step(actions)
        total_reward += rewards * (~np.array(done_i))  # Ne pas accumuler les récompenses pour les épisodes déjà terminés
        print(f"Récompenses : {rewards} | Total : {total_reward}")
        
        for i in range(nbr_workers):
            if terminated[i] or truncated[i]:
                done_i[i] = True
                state[i] = np.zeros((10, 10))  # Remplace l'état par un état neutre pour les épisodes terminés
            else:
                state[i,:,:-1] = next_state[i]
                step = infos[i].get("step", np.nan)  # Récupère le numéro de step
                state[i,:,-1] = np.arange(step, step+10)


        done_i = [terminated[i] or truncated[i] or done_i[i] for i in range(nbr_workers)]

    return total_reward


def train_cem_vectorized(
    agents : List[CEMLinearPolicy], 
    env: StudentGymEnvVectorized,
    nbr_env: int,
    max_iterations: int, 
    pop_size: int , 
    elite_frac: float , 
    initial_std: float,
    noise_decay: float,
    min_noise: float
) -> np.ndarray:
    theta_dim = len(agents[0].get_weights())
    
    mu = np.zeros(theta_dim)
    sigma = np.ones(theta_dim) * initial_std
    
    num_elites = int(pop_size * elite_frac)
    
    best_global_reward = -np.inf
    best_global_theta = None
    
    wandb.init(project="CSC-52081-EP", name=f"CEM-Training-{datetime.now().strftime('%Y%m%d_%H%M%S')}", config={
        "theta_dim": theta_dim,
        "pop_size": pop_size,
        "num_elites": num_elites,
        "max_iterations": max_iterations,
        "initial_std": initial_std,
        "noise_decay": noise_decay
    })
    
    print(f"Début CEM | Paramètres : {theta_dim} | Population : {pop_size} | Élites conservées : {num_elites}")
    print(f"Max Évaluations : {max_iterations * pop_size}")
    
    for it in range(max_iterations):
        population = np.random.randn(pop_size, theta_dim) * sigma + mu
        fitness_values = np.zeros(pop_size)
        
        for i in tqdm(range(0, pop_size, nbr_env)):
            batch_thetas = population[i:i+nbr_env]
            rewards = evaluate_workers(batch_thetas, env, agents)
            fitness_values[i:i+nbr_env] = rewards
                    
        elite_indices = np.argsort(fitness_values)[-num_elites:]
        elites = population[elite_indices]
        
        max_local_reward = np.max(fitness_values)
        if max_local_reward > best_global_reward:
            best_global_reward = max_local_reward
            best_global_theta = population[np.argmax(fitness_values)]
            
        mu = np.mean(elites, axis=0)
        current_noise = max(min_noise, initial_std * (noise_decay ** it))
        sigma = np.std(elites, axis=0) + current_noise
        
        mean_elite_reward = np.mean(fitness_values[elite_indices])
        
        wandb.log({
            "iteration": it + 1,
            "max_local_reward": max_local_reward,
            "mean_elite_reward": mean_elite_reward,
            "best_global_reward": best_global_reward,
            "noise": current_noise,
            "fitness_values": wandb.Histogram(fitness_values)
        })
        

    wandb.finish()
    print("\nConvergence terminée.")
    print(f"Meilleur score trouvé : {best_global_reward:.1f}")
    return best_global_theta # type: ignore

def main(max_iterations: int = 10, 
    pop_size: int = 20, 
    elite_frac: float = 0.2, 
    initial_std: float = 1.0,
    noise_decay: float = 0.9,
    min_noise: float = 0.01):

    env = create_student_gym_env_vectorized(num_envs=4)
    agents = [CEMSimpleLinearPolicy(num_sensors=10) for _ in range(4)]
    logging.getLogger("student_gym_env").setLevel(logging.CRITICAL)

    best_weights = train_cem_vectorized(
        agents=agents,
        env=env,
        nbr_env=4,
        max_iterations=max_iterations,
        pop_size=pop_size,
        elite_frac=elite_frac,
        initial_std=initial_std,
        noise_decay=noise_decay,
        min_noise=min_noise
    )
    np.save("saves/cem_best_weights.npy", best_weights)
    print("Poids optimaux sauvegardés dans 'saves/cem_best_weights.npy'.")
    env.close()
if __name__ == "__main__":
    main()