import random
import numpy as np
import torch
import gymnasium as gym
import collections
from typing import List, Tuple, Union

"""
This module contains utility classes and functions for the Student Gym Environment package, including:
- MinimumExponentialLR: A learning rate scheduler that applies exponential decay with a minimum learning rate.
- EpsilonGreedy: An epsilon-greedy policy for action selection in reinforcement learning.
- ReplayBuffer: A replay buffer for storing and sampling transitions in reinforcement learning.
Source : Lab5
"""

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_decay: float,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The decay factor (gamma) for exponential learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]

class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        env: gym.Env,
        q_network: torch.nn.Module,
        device: torch.device,
        nbr_envs: int = 1,
    ):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        nbr_envs : int, optional
            The number of environments being trained on. Default is 1.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network
        self.device = device
        self.nbr_envs = nbr_envs    

    def __call__(self, state: np.ndarray) -> torch.Tensor:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        torch.Tensor
            The chosen action(s).
        """

        if random.random() < self.epsilon:
            if self.nbr_envs > 1:
                actions = torch.tensor([self.env.action_space.sample() for _ in range(self.nbr_envs)], dtype=torch.int64)
            else:
                actions = torch.tensor(self.env.action_space.sample(), dtype=torch.int64)

        else:
            with torch.no_grad():
                if self.nbr_envs > 1:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (nbr_envs, 10, 9)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) #shape (1, 10, 9)

                q_values = self.q_network(state_tensor) # Compute the Q-values for the current state using the Q-network

                actions = torch.argmax(q_values, dim=1).detach() # Select the action with the highest Q-value

        return actions # Return the chosen action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.int64,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        assert state.shape == (10, 9), f"Expected state shape (10, 9), got {state.shape}"
        assert next_state.shape == (10, 9), f"Expected next_state shape (10, 9), got {next_state.shape}"
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Tuple[int], Tuple[float], np.ndarray, Tuple[bool]]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)


def evaluate_qpolicy(
    env: gym.Env,
    q_network: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, List[float], List[np.ndarray], List[int]]:
    """
    Evaluate the given Q-network on the environment.

    Parameters
    ----------
    env : gym.Env
        The environment to evaluate the policy on.
    q_network : torch.nn.Module
        The Q-network to evaluate.
    num_episodes : int
        The number of episodes to run for evaluation.
    device : torch.device
        The device to run the evaluation on.

    Returns
    -------
    List[float]
        A list of episode rewards obtained during evaluation.
    """
    rewards = []
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    observations = [] 
    actions = []

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward # type: ignore
        state = next_state
        observations.append(state)
        actions.append(action)
        rewards.append(reward)

    return total_reward, rewards, observations, actions
