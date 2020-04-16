[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This repository contains a solution for the first project from the Udacity Deep Reinforcement Learning Program. Inference score is equal to ```14```, best achievied mean score over 100 consecutive episodes during training is equal to ```16.73```. The agent has solved the task in ```461 episodes```.

### Environment

![Trained Agent][image1]

The environment is based on the unity ml-agents library.  A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Setting up the environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Follow instructions from the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install required python dependencies.

### Train model
To train the agent, run in the terminal:

```bash
python train.py --path  <environment path>
```


You should see a result similar to one of the following.
Notice that, you will be notified when the enviornment is solved. The best model is saved according to the mean score over 100 consecutive episodes in the file "checkpoint.pt". The train.py file has many other additional parameters, to display all of them, type in the terminal ```python train.py -h```.

```Terminal output```

```terminal
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
Episode 100     Average Score: 0.735
..
Episode 487     Average Score: 13.01
Environment solved in 487 episodes!     Average Score: 13.01
Episode 500     Average Score: 13.26
...
Episode 1900    Average Score: 16.47
Episode 2000    Average Score: 15.73
Best average score: 17.06 
```


### Inference

To inference the trained model, run in the terminal:

```bash
python inference.py -m <checkpoint path> -e <environment path> 
```

The command will return an average score during the episode. For the saved checkpoint (checkpoint.pt file) it is equal to ```14```.

