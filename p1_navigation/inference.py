from unityagents import UnityEnvironment
import numpy as np
import torch
from Agent import QNetwork, QNetworkD
import argparse

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=args.env_path)

    brain_name = env.brain_names[0]

    if args.dueling:
        network = QNetworkD(37, 4).to(device).eval()
    else:
        network = QNetwork(37, 4).to(device).eval()

    network.load_state_dict(torch.load(args.model_path, map_location=str(device)))

    env_info = env.reset(train_mode=False)[brain_name] 
    state = env_info.vector_observations[0]            
    score = 0
    i = 0
    while True:
        i += 1
        print(i, end="\r")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = network(state).argmax(-1).item()
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
        
    print("Score: {}".format(score))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference DQN Network')
    parser.add_argument('--env_path', "-e", dest='env_path', help='path to environment', default="Banana_Linux/Banana.x86_64", type=str)
    parser.add_argument('--path_weights', "-m", dest='model_path', help='path to model weights', default="checkpoint.pt", type=str)
    parser.add_argument('--no_deuling', dest='dueling', action="store_false", help='Use dueling network', default=True)
    args = parser.parse_args()
    main(args)