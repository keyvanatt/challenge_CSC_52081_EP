import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cma
import wandb
from models.linear import CEMLinearPolicy
from concurrent.futures import ThreadPoolExecutor
from student_client import create_student_gym_env
from datetime import datetime


# 2. La fonction d'évaluation (Objectif : MINIMISER l'opposé du score)
def evaluate_for_cma(theta: np.ndarray, env: gym.Env, agent: torch.nn.Module) -> float:
    """
    Joue un épisode avec les poids `theta`.
    Retourne -Reward pour que CMA-ES maximise la vraie récompense.
    """
    agent.set_weights(theta)
    state, _ = env.reset()
    
    # Rigueur sur le format [10, 9]
    if isinstance(state, np.ndarray):
        if state.shape == (9,):
            state = np.tile(state, (10, 1))
        elif state.shape == (1, 9):
            state = np.repeat(state, 10, axis=0)
            
    total_reward = 0.0
    done = False
    
    while not done:
        action = agent(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        state = next_state
        
    # CRUCIAL : inversion du signe
    return -total_reward 

def evaluate_worker(theta):
    """
    Fonction isolée pour un thread. 
    Chaque évaluation possède sa propre instance de connexion au serveur.
    """
    local_env = create_student_gym_env()
    local_agent = CEMLinearPolicy(sequence_length=10, num_sensors=9, num_actions=3)
    
    score = evaluate_for_cma(theta, local_env, local_agent)
    local_env.close()
    return score

# 3. La boucle d'optimisation CMA-ES
def train_cma_es(env: gym.Env, agent: torch.nn.Module, max_iterations: int = 150, sigma0: float = 0.5):
    """
    Optimise les poids de l'agent via CMA-ES.
    """
    theta_dim = len(agent.get_weights())
    initial_theta = np.zeros(theta_dim)
    
    # Configuration stricte de CMA-ES
    es = cma.CMAEvolutionStrategy(initial_theta, sigma0, {
        'maxiter': max_iterations,
        'verbose': -9
    })
    
    print(f"Début CMA-ES | Paramètres : {theta_dim} | Taille de population auto-calculée : {es.popsize}")
    
    iteration = 0
    while not es.stop():
        solutions = es.ask()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            fitness_values = list(executor.map(evaluate_worker, solutions))
        
        es.tell(solutions, fitness_values)
        
        current_best_score = -np.min(fitness_values)
        global_best_score = -es.result.fbest
        print(f"Iter {iteration+1:03d} | Meilleur Local : {current_best_score:.1f} | Meilleur Global : {global_best_score:.1f}")
        
        wandb.log({
            "iteration": iteration + 1,
            "best_local_score": current_best_score,
            "best_global_score": global_best_score,
            "mean_fitness": -np.mean(fitness_values)
        })
        
        iteration += 1

    best_theta = es.result.xbest
    agent.set_weights(best_theta)
    
    print("\nConvergence atteinte ou limite d'itérations dépassée.")
    print(f"Score optimal théorique (sur 1 épisode) : {-es.result.fbest:.1f}")
    
    return best_theta

# 4. Exécution
if __name__ == "__main__":
    wandb.init(project="CSC-52081-EP", name=f"CMA-ES-Training-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    env = create_student_gym_env()
    agent = CEMLinearPolicy(sequence_length=10, num_sensors=9, num_actions=3)
    
    best_weights = train_cma_es(env, agent, max_iterations=200, sigma0=0.5)
    
    np.save("saves/cma_es_best_weights.npy", best_weights)
    wandb.finish()
