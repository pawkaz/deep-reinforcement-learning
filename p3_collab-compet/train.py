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
from maddpg_agent import Agent

matplotlib.use('Agg')


def main(args):
    """
    Setup agents and run training.
    Params
    ======
        args (dict): maximum number of training episodes
    """
    print(args)
    
    env = UnityEnvironment(file_name=args.path)

    env_wr = EnvWrapper(env)

    agentA = Agent(state_size=24, action_size=2, order=0)
    agentB = Agent(state_size=24, action_size=2, order=1)

    scores = train(env_wr, (agentA, agentB), n_episodes=args.episodes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(scores)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")
    fig.savefig("scores.png")
    env_wr.close()



def train(env, agents, n_episodes:int=1000, score_threshold:float=.5)->list:
    """
    Train agents.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        score_threshold (float): score threshold 
    """

    scores = []
    scores_window:Deque[float] = deque(maxlen=100)
    best_score = float("-inf")
    writer = tensorboard.SummaryWriter(f"runs/{int(time())}")
    agentA, agentB = agents
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        
        start = time()
        while True:
            actionA = agentA.act(state[0])
            actionB = agentB.act(state[1])
            action = np.stack((actionA, actionB))
            next_state, reward, done, _ = env.step(action)
            next_actionA = agentA.act(next_state[0], False)
            next_actionB = agentB.act(next_state[1], False)
            agentA.step(state, action, reward, next_state, done, next_actionB)
            agentB.step(state, action, reward, next_state, done, next_actionA)
            state = next_state
            score += np.max(reward)
            if np.any(done):
                break

        time_for_episode = time() - start
        writer.add_scalar("train/time", time_for_episode, i_episode)
        scores_window.append(score)
        scores.append(score)
        window_score = np.mean(scores_window)

        writer.add_scalar("train/reward", score, i_episode)        
        writer.add_scalar("train/window", window_score, i_episode)

        
        print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}\tTime: {time_for_episode:.2f}', end="")
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {window_score:.2f}')

        if window_score >= score_threshold and best_score < score_threshold:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {window_score:.2f}')

        if window_score > best_score and window_score >= score_threshold:
            best_score = window_score
            torch.save(agentA.actor_local.state_dict(), 'checkpoint_actor_A.pt')
            torch.save(agentA.critic_local.state_dict(), 'checkpoint_critic_A.pt')
            torch.save(agentB.actor_local.state_dict(), 'checkpoint_actor_B.pt')
            torch.save(agentB.critic_local.state_dict(), 'checkpoint_critic_B.pt')

    print(f"Best average score: {best_score}")
    writer.close()
    return scores


class EnvWrapper():
    """
    Wrapper for the unity environment.
    """
    def __init__(self, env:UnityEnvironment):
        """
        Params
        ======
        env (UnityEnvironment): unity environment
        """
        self.env = env
        self.brain_name = self.env.brain_names[0]

    def step(self, action:np.ndarray)->Tuple[np.ndarray, float, float, None]:
        """
        Params
        ======
            action (np.ndarray): Agent action
        """
        env_info = self.env.step(action.reshape(-1))[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        return next_state, reward, done, None

    def reset(self, train_mode:bool=True)->np.ndarray:
        """
        Reset the environment.
        Params
        ======
            train_mode (bool): maximum number of training episodes
        """
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations
        return state

    def close(self):
        """
        Close the environment.
        """
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MADDPG Network')
    parser.add_argument('--path', dest='path', help='path to environment', default="Tennis_Linux/Tennis.x86_64", type=str)
    parser.add_argument('--episodes', dest='episodes', help='number of episodes', default=2000, type=int)
    args = parser.parse_args()
    main(args)
