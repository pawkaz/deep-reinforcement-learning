from unityagents import UnityEnvironment
import numpy as np
import torch
from Agent import QNetwork


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

    brain_name = env.brain_names[0]

    network = QNetwork(37, 4).to(device)
    network.load_state_dict(torch.load("checkpoint.pt"))

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    i = 0
    while True:
        i += 1
        print(i, end="\r")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)        # select an action
        action = network(state).argmax(-1).item()
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

    env.close()


if __name__ == "__main__":
    main()