import itertools
from typing import Callable, List 
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb
from utils import EpsilonGreedy, ReplayBuffer, MinimumExponentialLR, evaluate_qpolicy
from models.cnn import CNNEnginePolicy, init_weights_biased
from student_client import create_student_gym_env
from student_client.plotting import plot_observations
from datetime import datetime
import numpy as np


def train_agent(
    env: gym.Env,
    q_network: torch.nn.Module,
    target_q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epsilon_greedy: EpsilonGreedy,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_episodes: int,
    gamma: float,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    target_q_network_sync_period: int,
    save_every: int
) -> List[float]:
    """
    Train the Q-network on the given environment.
    Source : Lab5
    """
    iteration = 0
    episode_reward_list = []
    wandb.watch(q_network, log="all", log_freq=5)
    for episode_index in tqdm(range(1, num_episodes)):
        state, _ = env.reset()
        episode_reward = 0.0
        state = np.tile(state, (10, 1) )

        for _ in itertools.count():
            action = epsilon_greedy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done :
                next_state = np.tile(next_state, (10, 1) )
            replay_buffer.add(state, action, float(reward), next_state, done)
            episode_reward += float(reward)

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
                lr_scheduler.step()

                wandb.log({"loss": loss.item(), "iteration": iteration})

            if iteration % target_q_network_sync_period == 0:
                target_q_network.load_state_dict(q_network.state_dict())

            iteration += 1

            if done:
                break

            state = next_state

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()
        
        wandb.log({"episode_reward": episode_reward, 
                   "epsilon": epsilon_greedy.epsilon, 
                   "episode": episode_index,
                   "buffer_size": len(replay_buffer),
                   "one_learning_rate": optimizer.param_groups[0]['lr']
                   })
        if episode_index % save_every == 0:
            torch.save(q_network.state_dict(), f"saves/ddqn_q_network_episode_{episode_index}.pth")
            torch.save(target_q_network.state_dict(), f"saves/ddqn_target_q_network_episode_{episode_index}.pth")
    return episode_reward_list

def main(
    learning_rate: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.998,
    replay_buffer_capacity: int = 10000,
    num_episodes: int = 1000,
    gamma: float = 0.9,
    batch_size: int = 64,
    target_q_network_sync_period: int = 100,
    lr_decay: float = 0.998,
    min_lr: float = 5e-5,
    save_every: int = 500,
):
    wandb.init(project="CSC-52081-EP", name=f"DDQN-Training-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    wandb.config.update({
        "learning_rate": learning_rate,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "replay_buffer_capacity": replay_buffer_capacity,
        "num_episodes": num_episodes,
        "gamma": gamma,
        "batch_size": batch_size,
        "target_q_network_sync_period": target_q_network_sync_period,
        "lr_decay": lr_decay,
        "min_lr": min_lr,
        "save_every": save_every
    })
    # --- Initialisation de l'environnement, des réseaux, de l'optimiseur, etc. --- 
    env = create_student_gym_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
        device=device
    )
    
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=lr_decay, min_lr=min_lr)
    
    episode_rewards = train_agent(
        env=env,
        q_network=q_network,
        target_q_network=target_q_network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epsilon_greedy=epsilon_greedy,
        device=device,
        lr_scheduler=lr_scheduler,
        num_episodes=num_episodes,
        gamma=gamma,
        batch_size=batch_size,
        replay_buffer=replay_buffer,
        target_q_network_sync_period=target_q_network_sync_period,
        save_every=save_every
    )
    # save final models
    torch.save(q_network.state_dict(), f"saves/ddqn_q_network_final.pth")
    torch.save(target_q_network.state_dict(), f"saves/ddqn_target_q_network_final.pth")
    np.save("saves/episode_rewards.npy", np.array(episode_rewards))

if __name__ == "__main__":
    main()