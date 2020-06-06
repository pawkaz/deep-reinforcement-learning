import argparse
import random
from collections import deque, namedtuple
from typing import Deque, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment
from torch.utils import tensorboard
from time import time
from ddpg_agent import Agent

matplotlib.use('Agg')


def main(args):
    print(args)
    
    env = UnityEnvironment(file_name=args.path)

    env_wr = EnvWrapper(env)

    agent = Agent(state_size=33, action_size=4, random_seed=10)

    scores = train(env_wr, agent, n_episodes=args.episodes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(scores)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")
    fig.savefig("scores.png")
    env_wr.close()

def evaluate(env, agent)->int:
    """
    """


def train(env, agent, n_episodes:int=1000, score_threshold:float=32)->list:
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    scores = []
    scores_window:Deque[float] = deque(maxlen=100)
    best_score = float("-inf")
    writer = tensorboard.SummaryWriter(f"runs/{int(time())}")
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        
        start = time()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.mean(reward)
            if np.any(done):
                break

        time_for_episode = time() - start
        writer.add_scalar("train/time", time_for_episode, i_episode)
        scores_window.append(score)
        scores.append(score)
        window_score = np.mean(scores_window)

        writer.add_scalar("train/reward", score, i_episode)        
        writer.add_scalar("train/window", window_score, i_episode)
        writer.add_scalar("train/memory_size", len(agent.memory), i_episode)

        
        print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}\tTime: {time_for_episode:.2f}', end="")
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}')

        if window_score >= score_threshold and best_score < score_threshold:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {window_score:.2f}')

        if window_score > best_score and window_score >= score_threshold:
            best_score = window_score
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pt')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pt')

    print(f"Best average score: {best_score}")
    writer.close()
    return scores


class EnvWrapper():
    """Wrapper for the unity environment.

    Params
    ======
        env (UnityEnvironment): unity environment
    """
    def __init__(self, env:UnityEnvironment):
        self.env = env
        self.brain_name = self.env.brain_names[0]

    def step(self, action:np.ndarray)->Tuple[np.ndarray, float, float, None]:
        env_info = self.env.step(action.reshape(-1))[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        return next_state, reward, done, None

    def reset(self, train_mode:bool=True)->np.ndarray:
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations
        return state

    def close(self):
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPG Network')
    parser.add_argument('--path', dest='path', help='path to environment', default="Reacher_Linux/Reacher.x86_64", type=str)
    parser.add_argument('--episodes', dest='episodes', help='number of episodes', default=200, type=int)
    parser.add_argument('--eval', dest='eval', help='run evaluation', default=200, type=int)
    args = parser.parse_args()
    main(args)
