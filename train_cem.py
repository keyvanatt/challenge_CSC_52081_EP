import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from models.conditionnal_model import ConditionalPolicy
from student_client.student_gym_env_vectorized import StudentGymEnvVectorized, create_student_gym_env_vectorized
from datetime import datetime
from typing import Callable, Optional
import logging
from typing import List


def evaluate_workers(thetas: List[np.ndarray], env: StudentGymEnvVectorized, agents: List[ConditionalPolicy]) -> np.ndarray:
    """
    Simule un vol complet pour un vecteur de paramètres theta.
    Possède son propre environnement local pour éviter les collisions HTTP.
    """
    
    nbr_workers = len(thetas)
    total_reward = np.zeros(nbr_workers)
    
    for agent, theta in zip(agents, thetas):
        agent.set_weights(theta)
        agent.reset() 
        
    state, _ = env.reset()
    
    # Sécurisation de la dimension initiale si l'environnement renvoie du (4, 9)
    if state.ndim == 2 and state.shape[1] == 9:
        state = np.expand_dims(state, axis=1)  # (4, 1, 9)
        state = np.tile(state, (1, 10, 1))     # (4, 10, 9)
            
    # 2. CORRECTION : Utilisation d'un tableau Numpy ultra-rapide
    done_i = np.zeros(nbr_workers, dtype=bool)
    step = 0    
    while not np.all(done_i):
        actions = np.zeros(nbr_workers, dtype=int)
        
        for i, agent in enumerate(agents):
            if not done_i[i]:
                actions[i] = agent(state[i])
        print(f"Step {step} - Actions prises par les agents : {actions}")
        step += 1

        # Avancement de l'environnement vectoriel
        next_state, rewards, terminated, truncated, infos = env.step(actions)
        
        # Accumulation stricte des récompenses (masque inversé)
        total_reward += rewards * (~done_i)

        print("Rewards :", rewards, "Total Rewards :", total_reward)
        print("Done Flags :", done_i)
        
        for i in range(nbr_workers):
            if not done_i[i]:
                if terminated[i] or truncated[i]:
                    done_i[i] = True
                    state[i] = np.zeros((10, 9))
                    agents[i].reset()
                else:
                    state[i] = next_state[i]

    return total_reward


def train_cem_vectorized(
    agents : List[ConditionalPolicy], 
    env: StudentGymEnvVectorized,
    nbr_env: int,
    max_iterations: int, 
    pop_size: int , 
    elite_frac: float , 
    initial_std: float,
    noise_decay: float,
    min_noise: float,
    mu_init: Optional[np.ndarray] = None
) -> np.ndarray:
    theta_dim = len(agents[0].get_weights())
    
    if mu_init is not None:
        assert mu_init.shape == (theta_dim,)
        mu = mu_init
        base_noise = np.abs(mu) * 0.20 + 0.05
    else:
        mu = np.zeros(theta_dim)
        base_noise = np.ones(theta_dim) * initial_std
        
    sigma = base_noise.copy()
    
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
        min_noise_vector = np.maximum(min_noise, np.abs(mu) * 0.01)
        current_noise_vector = np.maximum(min_noise_vector, base_noise * (noise_decay ** it))
        sigma = np.std(elites, axis=0) + current_noise_vector
        
        mean_elite_reward = np.mean(fitness_values[elite_indices])
        

        # -- log --
        current_normal_values = mu[0:9]
        current_sensor_weights = mu[9:18]

        log_metrics = {
            "iteration": it + 1,
            "max_local_reward": max_local_reward,
            "mean_elite_reward": mean_elite_reward,
            "best_global_reward": best_global_reward,
            "noise_mean": np.mean(sigma),
            "fitness_values": wandb.Histogram(fitness_values),
            
            "policy/no_repair_confidence": mu[18],
            "policy/no_sell_confidence": mu[19],
            "policy/repair_every": mu[20],
            "policy/sell_after": mu[21]
        }

        for s_idx in range(9):
            log_metrics[f"policy/normal_values_{s_idx}"] = current_normal_values[s_idx]
            log_metrics[f"policy/sensor_weights_{s_idx}"] = current_sensor_weights[s_idx]

        wandb.log(log_metrics)
        

    wandb.finish()
    print("\nConvergence terminée.")
    print(f"Meilleur score trouvé : {best_global_reward:.1f}")
    return best_global_theta # type: ignore

def main(max_iterations: int = 30, 
    pop_size: int = 20, 
    elite_frac: float = 0.2, 
    initial_std: float = 1.0,
    noise_decay: float = 0.9,
    min_noise: float = 0.01):

    env = create_student_gym_env_vectorized(num_envs=4)
    agents = [ConditionalPolicy(num_sensors=9, num_actions=3) for _ in range(4)]
    agents[0].initialize_weights(jsonl_datapath="local/ExplorationAgent.jsonl", first_n_steps=5)
    mu_0 = agents[0].get_weights()
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
        min_noise=min_noise,
        mu_init=mu_0
    )
    np.save("saves/cem_best_weights.npy", best_weights)
    print("Poids optimaux sauvegardés dans 'saves/cem_best_weights.npy'.")
    env.close()
if __name__ == "__main__":
    main()