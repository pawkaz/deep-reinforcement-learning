# Report

Two agents are able to solve the environment in ```1649 episodes```. The implementation is adapted from the ddpg repository. 

## Learning Algorithm 
The [MADDPG algorithm](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) was used. The baseline DDPG implementation consists of training loop, agent and replay buffer. The implementation was amended by adding compatibility with two agents. MADDPG differs from DDPG in the way how the information is given to critics, that is they receive all observations from each agent. Each agent has a pair of networks, actor and critic. At execution time only actor is being used and local observation. At training time both networks are being used and critic receives all information from both agents. It is shown in the picture below, which comes from the original paper. This approach is called multiagent decentralized actor, centralized critic.
![Trained Agent](screen.png)
### Model

Two types of the deep neural network were used: actor and critic models.

The actor architecture is based on 4 fully connected layers.

* FullyConnected shape: <state_size, 128>
* FullyConnected shape: <128, 256>
* FullyConnected shape: <256, 128>
* FullyConnected shape: <128, action_size>

The critic architecture is similar:

* FullyConnected shape: <state_size * 2, 256>
* FullyConnected shape: <256 + action_size * 2, 256>
* FullyConnected shape: <256, 128>
* FullyConnected shape: <128, 1>

For this project state_size is equal to ```24```, and action size is equal to ```2```. The ReLU and LeakyReLu functions were used as activation for the actor and critic models respectively.

### Optimizer
To optimize the weights of the models the agent uses Adam algorithm. Ornstein-Uhlenbeck noise is scaled by the factor 0.5 compared to the standard ddpg implementation. During training the update is being applied every timestamp. Each reward is scaled by a factor of 100, this helps speed up training.

### Hyperparameters
All hyperparameters are stored at the top of the maddpg_agent.py file. All of them are listed below.

PARAMETER  | Description | Default value
------------ | -------------| -------------|
GAMMA  | Discount factor | 0.99
TAU  | Controls soft update of a target network| 1e-3
BATCH_SIZE  | Batch size | 128
BUFFER_SIZE   | Replay buffer size | 2e5
LR_ACTOR   | Learning rate of the actor model | 1e-4
LR_CRITIC   | Learning rate of the critic model | 4e-4
UPDATE_EVERY   | How often turn on learning | 1
UPDATES   | How many updates during learning | 1

## Results
The weights are saved in files ```checkpoint_actor_A.pt```, ```checkpoint_critic_A.pt```, ```checkpoint_actor_B.pt```, ```checkpoint_critic_B.pt```. The task was solved in ```1642``` episodes. Best average score over 100 consecutive episodes is ```2.71```. The scores are plotted below. 

![Trained Agent](scores.png)

```Logs from the terminal:```
```

Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :

Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
Episode 100     Average Score: 0.00     Time: 0.22
Episode 200     Average Score: 0.01     Time: 0.23
Episode 300     Average Score: 0.00     Time: 0.50
Episode 400     Average Score: 0.02     Time: 0.23
Episode 500     Average Score: 0.00     Time: 0.23
Episode 600     Average Score: 0.01     Time: 0.23
Episode 700     Average Score: 0.02     Time: 0.23
Episode 800     Average Score: 0.06     Time: 0.23
Episode 900     Average Score: 0.02     Time: 0.36
Episode 1000    Average Score: 0.07     Time: 0.23
Episode 1100    Average Score: 0.07     Time: 0.49
Episode 1200    Average Score: 0.18     Time: 0.496
Episode 1300    Average Score: 0.18     Time: 0.94
Episode 1400    Average Score: 0.19     Time: 0.54
Episode 1500    Average Score: 0.35     Time: 4.463
Episode 1600    Average Score: 0.41     Time: 1.078
Episode 1649    Average Score: 0.52     Time: 16.90
Environment solved in 1649 episodes!    Average Score: 0.52
Episode 1700    Average Score: 1.21     Time: 16.84
Episode 1800    Average Score: 2.30     Time: 17.09
Best average score: 2.71
```
## Ideas for Future Work

There are many other potential algorithms that can solve the task. My plans are listed below:

- Try different noise parameters.
- Try PPO algorithm.
- Limit the number of timestamps in the episode. 
