import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size:int, action_size:int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.l = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, action_size)
        )

    def forward(self, state:torch.Tensor)->torch.Tensor:
        """Build a network that maps state -> action values."""
        return self.l(state)

class QNetworkD(nn.Module):
    """Dueling version of QNetwork"""

    def __init__(self, state_size:int, action_size:int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(QNetworkD, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True)
        )
        self.stream_state = nn.Linear(32, 1)
        self.stream_advantage = nn.Linear(32, action_size)

    def forward(self, state:torch.Tensor)->torch.Tensor:
        """Build a network that maps state -> action values."""
        features = self.l(state)
        features_state, features_advantage = torch.split(features, 32, 1)
        state_value = self.stream_state(features_state)
        advantage_value = self.stream_advantage(features_advantage)
        advantage_value = advantage_value - advantage_value.mean(1, keepdim=True)
        q_value = advantage_value + state_value
        return q_value


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, lr: float, batch_size: int,
                 update_every: int, gamma: float, tau: float, buffer_size: int,
                 dueling: bool = True, decoupled: bool = True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            lr (float): learning rate
            batch_size (float): batch size
            update_every (float): how often update qnetwork
            gamma (float): discount factor
            tau (float): interpolation parameter 
            buffer_size: size of buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.decoupled = decoupled
        # self.seed = random.seed(seed)

        # Q-Network
        if dueling:
            self.qnetwork_local = QNetworkD(state_size, action_size).to(device)
            self.qnetwork_target = QNetworkD(state_size, action_size).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, batch_size)
        self.update_every = update_every
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:float):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma, self.decoupled)

    def act(self, state:np.ndarray, eps=0.)->int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return action_values.argmax(-1).item()
        else:
            return random.randint(0, self.action_size - 1)

    def learn(self, experiences:Tuple, gamma:float, decoupled:bool=True):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        self.qnetwork_target.eval()

        with torch.no_grad():
            if decoupled:
                q_actions = self.qnetwork_local(next_states).argmax(1).view(-1, 1)
                q_targets = self.qnetwork_target(next_states).gather(1, q_actions)
            else:
                q_targets = self.qnetwork_target(next_states).max(1)[0].view(-1, 1)
            q_targets *= gamma * (1.0 - dones)
            q_targets += rewards

        q_values = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_values, q_targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model:nn.Module, target_model:nn.Module):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size:int, buffer_size:int, batch_size:int):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory:Deque[Experience] = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
