import random
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.core import Env
from torch import nn
import optuna

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int], activation):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        #Modified the init so we can pass any activation we want to use
        self.activation = activation
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()

def training(hyperparameters):

    NUM_RUNS = hyperparameters["NUM_RUNS"]
    EPISODES = hyperparameters["EPISODES"]
    EPSILON = hyperparameters["EPSILON"]
    EPSILON_DECAY = hyperparameters["EPSILON_DECAY"]
    LEARNING_RATE = hyperparameters["LEARNING_RATE"]
    ACTIVATION_FUNCTION = hyperparameters["ACTIVATION_FUNCTION"]
    NN_NETWORK = hyperparameters["NN_NETWORK"]
    OPTIMIZER = hyperparameters["OPTIMIZER"]
    REPLAY_BUFFER_SIZE = hyperparameters["REPLAY_BUFFER_SIZE"]
    BATCH_SIZE = hyperparameters["BATCH_SIZE"]
    UPDATE_FREQUENCY = hyperparameters["UPDATE_FREQUENCY"]

    # Store results in list
    runs_results = []

    # Load CartPole env
    env = gym.make('CartPole-v1')

    for run in range(NUM_RUNS):
        print(f"Starting run {run+1} of {NUM_RUNS}")
        policy_net = DQN(NN_NETWORK, ACTIVATION_FUNCTION)
        target_net = DQN(NN_NETWORK, ACTIVATION_FUNCTION)
        update_target(target_net, policy_net)
        target_net.eval()
        epsilon = EPSILON

        optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
        }

        optimizer = optimizers[OPTIMIZER]

        optimizer = optimizer(policy_net.parameters(), lr=LEARNING_RATE)
        memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

        steps_done = 0

        episode_durations = []

        for i_episode in range(EPISODES):
            # Print every 50 episodes
            if (i_episode+1) % 50 == 0:
                print("episode ", i_episode+1, "/", EPISODES)

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, policy_net, state)

                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1

                epsilon = max(0.01, epsilon - EPSILON_DECAY/NUM_RUNS)

            if i_episode % UPDATE_FREQUENCY == 0: 
                update_target(target_net, policy_net)

        runs_results.append(episode_durations)
        
    print('Complete All Runs')

    return runs_results

def train_policy_network(hyperparameters):

    NUM_RUNS = hyperparameters["NUM_RUNS"]
    EPISODES = hyperparameters["EPISODES"]
    EPSILON = hyperparameters["EPSILON"]
    EPSILON_DECAY = hyperparameters["EPSILON_DECAY"]
    LEARNING_RATE = hyperparameters["LEARNING_RATE"]
    ACTIVATION_FUNCTION = hyperparameters["ACTIVATION_FUNCTION"]
    NN_NETWORK = hyperparameters["NN_NETWORK"]
    OPTIMIZER = hyperparameters["OPTIMIZER"]
    REPLAY_BUFFER_SIZE = hyperparameters["REPLAY_BUFFER_SIZE"]
    BATCH_SIZE = hyperparameters["BATCH_SIZE"]
    UPDATE_FREQUENCY = hyperparameters["UPDATE_FREQUENCY"]

    # Store results in list
    runs_results = []

    # Load CartPole env
    env = gym.make('CartPole-v1')

    for run in range(NUM_RUNS):
        print(f"Starting run {run+1} of {NUM_RUNS}")
        policy_net = DQN(NN_NETWORK, ACTIVATION_FUNCTION)
        target_net = DQN(NN_NETWORK, ACTIVATION_FUNCTION)
        update_target(target_net, policy_net)
        target_net.eval()
        epsilon = EPSILON

        optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
        }

        optimizer = optimizers[OPTIMIZER]

        optimizer = optimizer(policy_net.parameters(), lr=LEARNING_RATE)
        memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

        steps_done = 0

        episode_durations = []

        for i_episode in range(EPISODES):
            # Print every 50 episodes
            if (i_episode+1) % 50 == 0:
                print("episode ", i_episode+1, "/", EPISODES)

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, policy_net, state)

                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1

                epsilon = max(0.01, epsilon - EPSILON_DECAY/NUM_RUNS)

            if i_episode % UPDATE_FREQUENCY == 0: 
                update_target(target_net, policy_net)

        runs_results.append(episode_durations)
        
    print('Complete All Runs')

    return policy_net

# Create a Function to define the initial unoptimised Network given
def unoptimised_training():
    runs_results_unoptimised = []

    env = gym.make('CartPole-v1')
    for run in range(10):
        print(f"Starting run {run+1} of {10}")
        policy_net = DQN([4,2], F.relu)
        target_net = DQN([4,2], F.relu)
        update_target(target_net, policy_net)
        target_net.eval()

        optimizer = optim.SGD(policy_net.parameters(), lr=1.)
        memory = ReplayBuffer(1)

        steps_done = 0

        episode_durations = []

        for i_episode in range(250):
            if (i_episode+1) % 50 == 0:
                print("episode ", i_episode+1, "/", 250)

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(1, policy_net, state)

                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < 1:
                    transitions = memory.sample(1)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1
            # Update the target network, copying all weights and biases in DQN
            if i_episode % 1 == 0: 
                update_target(target_net, policy_net)
        runs_results_unoptimised.append(episode_durations)
    print('Complete')

    return runs_results_unoptimised

### Create the Optuna Study Functions

# Defining the objective function
def objective_function(runs_results):
    """
    Args:
    - runs_results (list): List of run results for multiple trials.

    Output:
    - int: Number of values in the mean exceeding a specified threshold.
    """
    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    treshold = 100  # Try to get the most of values higher than the threshold, set here to 100
    return (len(means.detach().numpy()) - np.count_nonzero(means.detach().numpy() > treshold))

# Define the objective with possible hyperparameters
def objective(trial):
    """
    Args:
    - trial (optuna.Trial): An Optuna trial object.

    Output:
    - int: The result of the objective function for the given hyperparameters.
    """
    # Defining the possible hyperparameters for Optuna
    EPSILON = trial.suggest_uniform("EPSILON", 0, 1)  # epsilon can take values from 0 to 1
    EPSILON_DECAY = trial.suggest_categorical('EPSILON_DECAY', [0.9, 0.95, 0.99]) 
    LEARNING_RATE = trial.suggest_loguniform("LEARNING_RATE", 1e-4, 1)
    NN_NETWORK = trial.suggest_categorical('NN_NETWORK', [
        [4, 18, 2],
        [4, 32, 2],
        [4, 32, 18, 2],
        [4, 64, 32, 2],
        [4, 32, 128, 2],
        [4, 64, 128, 2],
        [4, 128, 64, 2]
    ])
    ACTIVATION_FUNCTION = trial.suggest_categorical('ACTIVATION_FUNCTION', [F.sigmoid, F.relu, F.leaky_relu])
    OPTIMIZER = trial.suggest_categorical('OPTIMIZER', ['Adam', 'SGD'])
    REPLAY_BUFFER_SIZE = trial.suggest_categorical('REPLAY_BUFFER_SIZE', [100, 500, 1000, 10000, 15000, 20000])
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [1, 2, 5, 10, 15, 20])
    UPDATE_FREQUENCY = trial.suggest_categorical('UPDATE FREQUENCY', [1, 2, 5, 10, 20, 40])

    hyperparameters = {
        'NUM_RUNS': 10,
        'EPISODES': 250,
        'EPSILON': EPSILON,
        'EPSILON_DECAY': EPSILON_DECAY,
        'LEARNING_RATE': LEARNING_RATE,
        'NN_NETWORK': NN_NETWORK,
        'ACTIVATION_FUNCTION': ACTIVATION_FUNCTION,
        'OPTIMIZER': OPTIMIZER,
        'REPLAY_BUFFER_SIZE': REPLAY_BUFFER_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'UPDATE_FREQUENCY': UPDATE_FREQUENCY,
    }

    # Run with hyperparameters
    runs_results = training(hyperparameters)   

    # Try to be minimised
    run_value = objective_function(runs_results)

    return run_value

def create_optuna_study(n_trials):
    """
    Args:
    - n_trials (int): The number of trials for the Optuna study.

    Output:
    - None
    """

    study = optuna.create_study(study_name="CartPole-v1")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("Threshold_value: %s", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print("%s: %s", key, value)
