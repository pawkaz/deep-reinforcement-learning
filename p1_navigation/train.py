import argparse
import random
from collections import deque, namedtuple
from typing import Deque, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from Agent import Agent

matplotlib.use('Agg')


def main(args):
    print(args)
    
    env = UnityEnvironment(file_name=args.path)

    env_wr = EnvWrapper(env)

    agent = Agent(state_size=37, action_size=4, lr=args.lr,
                  batch_size=args.batch_size,
                  update_every=args.update_qnetwork,
                  gamma=args.gamma,
                  tau=args.tau,
                  buffer_size=args.buffer_size,
                  dueling=args.dueling,
                  decoupled=args.decoupled)

    scores = train(env_wr, agent,
                   n_episodes=args.episodes,
                   eps_start=args.eps_start,
                   eps_decay=args.eps_decay,
                   eps_end=args.eps_min)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(scores)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")
    fig.savefig("scores.png")
    env_wr.close()


def train(env, agent, n_episodes:int=1000, max_t:int=1000, eps_start:float=1.0, eps_end:float=0.01, eps_decay:float=0.995, score_threshold:float=13)->list:
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    scores_window:Deque[float] = deque(maxlen=100)
    eps = eps_start
    best_score = float("-inf")
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        window_score = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}', end="")

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}')

        if window_score >= score_threshold and best_score < score_threshold:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {window_score:.2f}')

        if window_score > best_score and window_score >= score_threshold:
            best_score = window_score
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pt')

    print(f"Best average score: {best_score}")
    return scores


class EnvWrapper():
    """Wrapper for the unity environment.

    Params
    ======
        env (UnityEnvironment): unity environment
    """
    def __init__(self, env:UnityEnvironment):
        self.env = env
        self.brain_name = env.brain_names[0]

    def step(self, action:int)->Tuple[np.ndarray, float, float, None]:
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None

    def reset(self, train_mode:bool=True)->np.ndarray:
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def close(self):
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN Network')
    parser.add_argument('--path', dest='path', help='path to environment', default="Banana_Linux/Banana.x86_64", type=str)
    parser.add_argument('--episodes', dest='episodes', help='number of episodes', default=1000, type=int)
    parser.add_argument('--no_deuling', dest='dueling', action="store_false", help='Use dueling network', default=True)
    parser.add_argument('--no_decoupled', dest='decoupled', action="store_false", help='Do not use decoupled target', default=True)
    parser.add_argument('--epsilon', dest='eps_start', help='Epsilon greedy policy', default=1.0, type=float)
    parser.add_argument('--gamma', dest='gamma', help='Discount factor', default=0.99, type=float)
    parser.add_argument('--tau', dest='tau', help='For soft update of target parameters', default=1e-3, type=float)
    parser.add_argument('--epsilon_decay', dest='eps_decay', help='Epsilon decay rate for every episode', default=0.995, type=float)
    parser.add_argument('--epsilon_min', dest='eps_min', help='Min epsilon value', default=0.01, type=float)
    parser.add_argument('--batch_size', "-bs", dest='batch_size', help='Batch size for the learn step', default=64, type=int)
    parser.add_argument('--buffer_size', dest='buffer_size', help='Replay buffer size', default=int(1e5), type=int)
    parser.add_argument('--learning_rate', "-lr", dest='lr', help='Learning rate', default=5e-4, type=float)
    parser.add_argument('--update_every', dest='update_qnetwork', help='How often update qnetwork', default=4, type=int)
    args = parser.parse_args()
    main(args)
