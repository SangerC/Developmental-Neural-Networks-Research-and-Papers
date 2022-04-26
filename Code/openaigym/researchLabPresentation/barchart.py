import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.distributions import Categorical
from torch.nn import parameter

class Policy(nn.Module):
    def __init__(self, input, output, hidden, layers=0):
        super(Policy, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input, hidden))
        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.ReLU())

        for i in range(layers):
            self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden, output))
        self.layers.append(nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()
    
    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


def predict(state, currEnv):
    # Select an action (0 or 1) by running policy model
    # and choosing based on the probabilities in state
    new_state = [0]*input_size
    x = 0
    for i in range(len(env) - currEnv - 2):
      x += env[i].observation_space.shape[0]

    for i in range(env[currEnv].observation_space.shape[0]):
      new_state[i + x] = state[i]
    state = torch.from_numpy(np.array(new_state)).type(torch.FloatTensor)
    action_probs = policy(state)
    distribution = Categorical(action_probs)
    action = distribution.sample()

    # Add log probability of our chosen action to our history
    policy.episode_actions = torch.cat([
        policy.episode_actions,
        distribution.log_prob(action).reshape(1)
    ])

    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1))

    # Update network weights
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.episode_rewards))
    policy.reset()

def train(episodes):
    scores = []
    currentEnv = 0
    for episode in range(episodes):
        state = env[currentEnv].reset()

        for time in range(1000):
            action = predict(state, currentEnv)

            # Uncomment to render the visual state in a window
            # env.render()

            # Step through environment using chosen action
            state, reward, done, _ = env[currentEnv].step(action.item())

            # Save reward
            policy.episode_rewards.append(reward)
            if done:
                break

        update_policy()

        # Calculate score to determine when the environment has been solved
        scores.append(time)
        mean_score = np.mean(scores[-100:])

        if episode % 50 == 0:
            print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(episode, mean_score))
            currentEnv += 1
            if currentEnv >= len(env):
              currentEnv = 0

        if mean_score > env[currentEnv].spec.reward_threshold:
            print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
                  .format(episode, mean_score, time))
            for i in range(episode, episodes):
               policy.reward_history.append(mean_score)
            break



env = []
env.append(gym.make('CartPole-v1'))

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
hidden_size = 512

num_seeds = 10
num_episodes = 500

input_size = sum(i.observation_space.shape[0] for i in env)
output_size = env[0].action_space.n

runs = []

runs.append([4, 0])
runs.append([4, 1])

runresults = []

for run in runs:

    total = 0

    for i in range(num_seeds):
    
        policy = Policy(input_size, output_size, run[0], run[1])
        print(policy.layers)
        pytorchSeed = random.randint(0, 1000)
        cartSeed = random.randint(0, 1000)
        for i in env:
            i.seed(cartSeed)
        torch.manual_seed(pytorchSeed)
    
        train(episodes=num_episodes)
        
        total += policy.reward_history[-1]
    runresults.append(total/num_seeds)
	
for i in range(len(runs)):
    print("Run with " + str(runs[i][1]) + " layers of " + str(runs[i][0]) + " size: " + str(runresults[i]))

def get_avg(arr_of_arrs, i):
    total = 0
    for k in range(len(arr_of_arrs)):
        total += arr_of_arrs[k][i]
    return total/len(arr_of_arrs)
