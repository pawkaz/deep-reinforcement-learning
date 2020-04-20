import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Deque
from PrioritizedSampler import PrioritizedSampler

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
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
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

        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        # self.memory = ReplayBuffer(action_size, self.buffer_size, batch_size)
        self.memory = PriorReplayBuffer(action_size, state_size, self.buffer_size, batch_size)
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
                experiences, indicies, weigths = self.memory.sample()
                errors = self.learn(experiences, self.gamma, self.decoupled, weigths)
                if indicies is not None:
                    self.memory.update_probabilities(indicies, errors)

    def act(self, state:np.ndarray, eps=0.)->int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return action_values.argmax(-1).item()
        else:
            return random.randint(0, self.action_size - 1)

    def learn(self, experiences:Tuple, gamma:float, decoupled:bool=True, weights:torch.Tensor=None):
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
        errors = loss.detach()

        if weights is not None:
            loss = loss * weights.view(-1, 1)
            loss.sum().backward()
        else:
            loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return errors

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

class PriorReplayBuffer:
    """Prioritized Fixed-size buffer to store experience tuples."""
    pointer = 0
    memory_occupied = 0
    def __init__(self, action_size:int, state_size:int, buffer_size:int, batch_size:int, alpha:float=0.7, beta:float=0.5, beta_incremental:float=1e-3, epsilon:float=1e-2):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.state_size = state_size
        self.memory = torch.zeros(buffer_size, state_size + state_size + 1 + 1 + 1, dtype=torch.float32)
        self.probs = torch.zeros(buffer_size, dtype=torch.float32, device=device) + epsilon
        self.batch_size = batch_size
        self.capacity = buffer_size
        self.dist = PrioritizedSampler.create(self.probs, device=device)
        self.alpha = alpha
        self.beta = beta
        self.beta_incremental = beta_incremental
        self.epsilon = epsilon
        self.max_p = epsilon
        self.min_p = epsilon
        

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.pointer %= self.capacity
        exp = np.concatenate([state.flatten(), [action, reward], next_state.flatten(), [done]])
        self.memory[self.pointer] = torch.from_numpy(exp)
        self.probs[self.pointer] = self.max_p
        self.pointer += 1
        self.memory_occupied = max(self.memory_occupied, self.pointer)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory_occupied

    def update_probabilities(self, idxs, errors, clip=1.0):
        probs = errors.abs().add_(self.epsilon).clamp_(max=clip).pow_(self.alpha)
        self.probs[idxs] = probs.view(-1)
        self.max_p = self.probs.max()
        self.min_p = self.probs.min()
        self.dist.update(self.probs)        

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
       
        indicies = self.dist.sample(self.batch_size) % self.__len__()
        # indicies = torch.zeros(self.batch_size, dtype=torch.long)
        # weights = torch.ones(self.batch_size, device=device).div_(self.batch_size)
        experiences = self.memory[indicies]

        max_weight = (self.min_p * self.__len__() / self.dist.tree[0]) ** (-self.beta)        
        weights = (self.__len__()  / self.dist.tree[0]) * self.probs[indicies].to(device, non_blocking=True)
        weights.pow_(-self.beta).div_(max_weight)
        
        
        states = experiences[:, :self.state_size].to(device, non_blocking=True)
        actions = experiences[:, self.state_size].view(-1, 1).to(device, non_blocking=True, dtype=torch.long)
        rewards = experiences[:, self.state_size + 1].view(-1, 1).to(device, non_blocking=True)
        next_states = experiences[:, self.state_size +2 : -1].to(device, non_blocking=True)
        dones = experiences[:, -1].view(-1, 1).to(device, non_blocking=True)

        return (states, actions, rewards, next_states, dones), indicies, weights



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

        return (states, actions, rewards, next_states, dones), None, None

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
