# Report

The agent is able to solve the environment in 500 episodes on average. The code is adapted from the Deep Q-Networks exercise.

## Learning Algorithm 
As in the exercise, the [DQN algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) was used. The basic model is able to solve the environment without any changes in less than 1000 episodes. The baseline implementation consists of training loop, agent and replay buffer. Additional improvements were implemented such as [Dueling network](https://arxiv.org/abs/1511.06581) and [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461). These changes helped to stabilize and speed up training. 

### Model

Deep neural network was used as a model. The architecture is based on three fully connected layers.

* FullyConnected shape: <state_size, 64>
* FullyConnected shape: <64, 64>
* FullyConnected shape: <64, action_size>

For this project state_size is equal to ```37```, and action size is equal to ```4```. The intermediate layer has 64 neurons, the ReLU function was used as activation. In case of using duealing, last layer is replaced by two layers, each represents a separate stream. 

### Loss function and optimizer
As a loss function the Mean Squared Error was used. It is standard function for a regression task. To optimize the weights of the model the agent uses Adam algorithm. Both implementations are from pytorch library. According the the DQN paper, the target is calculated using the separate target QNetwork, which is copy of QNetwork from previous episodes.

### Hyperparameters
All hyperparameters are controlled by train.py optional parameters. All of them are listed below. Default values are taken from the exercise.

Optional arguments:  | Description | Default value
------------ | -------------| -------------|
--path PATH  | Path to environment | Banana_Linux/Banana.x86_64
--no_deuling  | Disables dueling | False
--no_decoupled  | Use not decoupled target | False
--epsilon EPS_START   | Epsilon start value for greedy policy | 1.0
--epsilon_decay EPS_DECAY | Epsilon decay value every episode | 0.995
--epsilon_min EPS_MIN   | Epsilon min value | 0.01
--gamma GAMMA  | Discount factor | 0.99
--tau TAU  | Controls soft update of a target qnetwork| 0.001
--batch_size BATCH_SIZE  | Batch size | 64
--buffer_size BUFFER_SIZE   | Replay buffer size | 10000
--learning_rate LR, -lr   | LR Learning rate | 5e-4
--update_every UPDATE_EVERY   | How often update qnetwork | 4

## Results
The saved weights are in the file ```checkpoint.pt```, the task was solved in ```461``` episodes using default hyperparameters. Best average score over 100 consecutive episodes is ```16.73```. The scores are plotted below.

![Trained Agent](scores.png)

```Logs from the terminal:```
```
'Academy' started successfully!
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
Episode 100     Average Score: 0.91
Episode 200     Average Score: 4.56
Episode 300     Average Score: 7.31
Episode 400     Average Score: 11.45
Episode 461     Average Score: 13.01
Environment solved in 461 episodes!     Average Score: 13.01
Episode 500     Average Score: 13.39
Episode 600     Average Score: 14.69
Episode 700     Average Score: 14.57
Episode 800     Average Score: 15.16
Episode 900     Average Score: 16.05
Episode 1000    Average Score: 16.26
Best average score: 16.73
```
## Ideas for Future Work

There are many other potentail improvements to the vanilla DQN framework. My plans are listed below:

- [x] Implement dueling network.
- [x] Implement double DQN.
- [ ] Implement Prioritized Replay.
- [ ] Try other models.
- [ ] Implement model which takes raw pixel values as the input.
- [ ] Try other optimizers.
- [ ] Try other environments.

The most crucial for performance is Prioritized Replay which should help model to converge faster.
