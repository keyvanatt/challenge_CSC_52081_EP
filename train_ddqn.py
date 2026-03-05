import itertools
from typing import Callable, List 
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb
from utils import EpsilonGreedy, ReplayBuffer, MinimumExponentialLR, evaluate_qpolicy
from models.cnn import CNNEnginePolicy, init_weights_biased
from student_client import create_student_gym_env
from student_client.student_gym_env_vectorized import StudentGymEnvVectorized, create_student_gym_env_vectorized
from student_client.plotting import plot_observations
from datetime import datetime
import numpy as np


def train_agent_vectorized(
    env: StudentGymEnvVectorized,
    q_network: torch.nn.Module,
    target_q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epsilon_greedy: EpsilonGreedy,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    max_it: int,
    gamma: float,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    target_q_network_sync_period: int,
    save_every: int,
    env_batch_size: int,
    interation_init: int
)-> bool:
    """
    Train the Q-network on the given environment.
    Source : Lab5
    """
    iteration = interation_init
    episode_reward_list = []
    wandb.watch(q_network, log="all", log_freq=5)
    episode_rewards = np.zeros(env_batch_size)
    states = np.zeros((env_batch_size, 10, 9))  # Initial state placeholder
    episode = 0
    for _ in tqdm(range(max_it)):
        # A) Check if any environments terminated
        terminated_envs = env.get_terminated_env_indices()
        if terminated_envs:
            print(f"   ⚠️  Environments {terminated_envs} terminated")
            reset_states, reset_infos = env.reset_specific_envs(terminated_envs)
            print(f"   Reset observations shape: {reset_states.shape}")
            for i, env_id in enumerate(terminated_envs):
                states[env_id] = np.tile(reset_states[i], (10, 1))  # reset previous state
                print(f"   State for env {env_id} after reset: {states[env_id].shape}")
                episode += 1
                wandb.log({f"episode_reward_env_{env_id}": episode_rewards[env_id], "episode_total": episode})
                episode_rewards[env_id] = 0.0  # reset episode reward for terminated env
       
        actions = epsilon_greedy(states)
        actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else np.array(actions)
        print(f"Actions chosen: {actions}")
        next_states, rewards, terminateds, truncateds, infos = env.step(actions) 

        for i in range(env_batch_size):
            done_i = terminateds[i] or truncateds[i]
            if done_i:
                next_states[i] = np.tile(next_states[i], (10, 1))  # reset next state for terminated env
            try:
                replay_buffer.add(states[i], actions[i], float(rewards[i]), next_states[i], done_i)
            except AssertionError as e:
                print(f"Error adding to replay buffer for env {i}: {e}")
                print(f"State: {states[i]}, Action: {actions[i]}, Reward: {rewards[i]}, Next state: {next_states[i]}, Done: {done_i}")
                continue
            episode_rewards[i] += float(rewards[i])

        if len(replay_buffer) > batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_state_q_values, _ = target_q_network(batch_next_states_tensor).max(dim=1)
                targets = batch_rewards_tensor + gamma * next_state_q_values * (1 - batch_dones_tensor)

            current_q_values = q_network(batch_states_tensor)
            loss = loss_fn(current_q_values[range(batch_size), batch_actions_tensor], targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "iteration": iteration})

        if iteration % target_q_network_sync_period == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        states = np.array(next_states)

        epsilon_greedy.decay_epsilon()
        if len(replay_buffer) > batch_size: # On ne fait le scheduler que si on a commencé à apprendre, sinon on perd du temps à faire du scheduler pour rien
            lr_scheduler.step()
        
        wandb.log({"iteration": iteration, 
                   "epsilon": epsilon_greedy.epsilon, 
                   "buffer_size": len(replay_buffer),
                   "one_learning_rate": optimizer.param_groups[0]['lr']
                   })
        if iteration % save_every == 0:
            torch.save(q_network.state_dict(), f"saves/ddqn_q_network_iteration_{iteration}.pth")
            torch.save(target_q_network.state_dict(), f"saves/ddqn_target_q_network_iteration_{iteration}.pth")
        
        iteration += 1

    return True

def main(
    learning_rate: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    replay_buffer_capacity: int = 100_000,
    max_it: int = 1500,
    gamma: float = 0.99,
    batch_size: int = 128,
    target_q_network_sync_period: int = 100,
    lr_decay: float = 0.995,
    min_lr: float = 5e-5,
    save_every: int = 250,
    env_batch_size: int = 8,
    load_models = None
):
    wandb.init(project="CSC-52081-EP", name=f"DDQN-Training-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    wandb.config.update({
        "learning_rate": learning_rate,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "replay_buffer_capacity": replay_buffer_capacity,
        "max_it": max_it,
        "gamma": gamma,
        "batch_size": batch_size,
        "target_q_network_sync_period": target_q_network_sync_period,
        "lr_decay": lr_decay,
        "min_lr": min_lr,
        "save_every": save_every,
        "env_batch_size": env_batch_size
    })
    # --- Initialisation de l'environnement, des réseaux, de l'optimiseur, etc. --- 
    env = create_student_gym_env_vectorized(num_envs=env_batch_size)  # Environnement vectorisé pour gérer plusieurs épisodes en parallèle
    env_batch_size = env.num_envs
    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"   Episode IDs: {env.episode_ids}")

    # Reset all environments
    print(f"\n🔄 Resetting all environments...")
    observations, infos = env.reset()

    print(f"   Observations shape: {observations.shape}")
    print(f"   First observation: {observations[0]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if load_models is not None:
        q_network = CNNEnginePolicy().to(device)
        q_network.load_state_dict(torch.load(load_models['q_network'], map_location=device))
        target_q_network = CNNEnginePolicy().to(device)
        target_q_network.load_state_dict(torch.load(load_models['target_q_network'], map_location=device))
        print(f"Models loaded from {load_models['q_network']} and {load_models['target_q_network']}")
    else:
        q_network = CNNEnginePolicy().to(device)
        q_network.apply(init_weights_biased)  # Initialisation avec biais pour favoriser "Do Nothing"
        target_q_network = CNNEnginePolicy().to(device)
        target_q_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()  # Huber Loss, plus stable que MSE pour les Q-values
    
    epsilon_greedy = EpsilonGreedy(
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        env=env,
        q_network=q_network,
        device=device,
        nbr_envs=env_batch_size
    )
    
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=lr_decay, min_lr=min_lr)
        
    train_agent_vectorized(
        env=env,
        q_network=q_network,
        target_q_network=target_q_network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epsilon_greedy=epsilon_greedy,
        device=device,
        lr_scheduler=lr_scheduler,
        max_it=max_it,
        gamma=gamma,
        batch_size=batch_size,
        replay_buffer=replay_buffer,
        target_q_network_sync_period=target_q_network_sync_period,
        save_every=save_every,
        env_batch_size=env_batch_size,
        interation_init =  load_models["iteration"] if load_models is not None else 0
    )
    # save final models
    torch.save(q_network.state_dict(), f"saves/ddqn_q_network_final.pth")
    torch.save(target_q_network.state_dict(), f"saves/ddqn_target_q_network_final.pth")

    env.close()

if __name__ == "__main__":
    main()