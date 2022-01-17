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
    def __init__(self, input, output, hidden):
        super(Policy, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input, hidden))
        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden, output))
        self.layers.append(nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def addNode(self):
        old = self.layers[-5]
        connect = self.layers[-2]

        new = nn.Linear(old.in_features, old.out_features + 1)
        newOut = nn.Linear(connect.in_features + 1, connect.out_features)
        
        with torch.no_grad():
            for i in range(len(old.weight)):
                for j in range(len(old.weight[i])):
                    new.weight[i][j] = old.weight[i][j]
            for i in range(len(new.weight[-1])):
                new.weight[-1][i] = 0
            
            for i in range(len(connect.weight)):
                for j in range(len(connect.weight[i])):
                    newOut.weight[i][j] = connect.weight[i][j]
                newOut.weight[i][-1] = 0
        self.layers[-5] = new
        self.layers[-2] = newOut
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def addLayer(self, size):
        new_layer = nn.Linear(size, size)
        torch.nn.init.constant_(new_layer.weight, 0)
        new_layer.bias.data.fill_(0)
        with torch.no_grad():
            for i in range(len(new_layer.weight)):
                new_layer.weight[i, i] = 1
        self.layers.insert(len(self.layers) - 2, new_layer)
        self.layers.insert(len(self.layers) - 2, nn.Dropout(p=.5))
        self.layers.insert(len(self.layers) - 2, nn.ReLU())
        for p in self.layers:
            p.requires_grad = False
        self.layers[-5].requires_grad = True
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x, printThis = False):
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
        if episode % 50 == 0:
          policy.addNode()
          for layer in policy.layers:
              print(layer)
        
        if episode == 400:
            for p in policy.layers:
                p.requires_grad = True

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
#env.append(gym.make('Pendulum-v0')) 

folderName = 'startAt64GrowFirstLayer'

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
hidden_size = 64

num_seeds = 10
num_episodes = 500


input_size = sum(i.observation_space.shape[0] for i in env)
#output_size = sum(i.action_space.n for i in env)
output_size = env[0].action_space.n


rewards_history_by_run = []

for i in range(num_seeds):

	policy = Policy(input_size, output_size, hidden_size)
	pytorchSeed = random.randint(0, 1000)
	cartSeed = random.randint(0, 1000)
	for i in env:
	  i.seed(cartSeed)
	torch.manual_seed(pytorchSeed)

	train(episodes=num_episodes)

	rewards_history_by_run.append(policy.reward_history)

	# number of episodes for rolling average
	window = 50

	fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
	rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
	std = pd.Series(policy.reward_history).rolling(window).std()
	ax1.plot(rolling_mean)
	ax1.fill_between(range(len(policy.reward_history)), rolling_mean -
	                 std, rolling_mean+std, color='orange', alpha=0.2)
	ax1.set_title(
	    'Episode Length Moving Average ({}-episode window)'.format(window))
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Episode Length')

	ax2.plot(policy.reward_history)
	ax2.set_title('Episode Length')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Episode Length')

	fig.tight_layout(pad=2)
	plt.savefig(folderName + '/PytorchSeed' + str(pytorchSeed) + 'cartSeed' + str(cartSeed) )
	

average_history = []

def get_avg(arr_of_arrs, i):
    total = 0
    for k in range(len(arr_of_arrs)):
        total += arr_of_arrs[k][i]
    return total/len(arr_of_arrs)

for i in range(num_episodes):
    average_history.append(get_avg(rewards_history_by_run, i))

## number of episodes for rolling average
window = 50

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
rolling_mean = pd.Series(average_history).rolling(window).mean()
std = pd.Series(average_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(average_history)), rolling_mean -
                 std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title(
    'Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Length')

ax2.plot(average_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.savefig(folderName + '/average')
