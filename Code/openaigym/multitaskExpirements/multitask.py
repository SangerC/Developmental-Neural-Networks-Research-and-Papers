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
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden, output))
        self.layers.append(nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def addNode(self, layer, n = 1, optimizeAll = True):
        actualLayer = layer * 2
        nextLayer = actualLayer + 2

        old = self.layers[actualLayer]
        oldNextLayer = None
        if nextLayer < len(self.layers):
            oldNextLayer = self.layers[nextLayer]

        new = nn.Linear(old.in_features, old.out_features + n)
        if oldNextLayer:
            newNext = nn.Linear(oldNextLayer.in_features + n, oldNextLayer.out_features)
        
        with torch.no_grad():
            for i in range(len(old.weight)):
                for j in range(len(old.weight[i])):
                    new.weight[i][j] = old.weight[i][j]
            for i in range(len(old.weight), (len(new.weight))):
                for j in range(len(new.weight[i])):
                    new.weight[i][j] = 0
            
            if oldNextLayer:
                for i in range(len(oldNextLayer.weight)):
                    for j in range(len(oldNextLayer.weight[i])):
                        newNext.weight[i][j] = oldNextLayer.weight[i][j]
                    for j in range(len(oldNextLayer.weight[i]), len(newNext.weight[i])):
                        newNext.weight[i][j] = 0
        self.layers[actualLayer] = new
        if oldNextLayer:
            self.layers[nextLayer] = newNext
        if optimizeAll:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            if oldNextLayer:
                self.optimizer = optim.Adam(iter(self.layers[actualLayer:nextLayer + 1]), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(iter(self.layers[actualLayer:actualLayer + 1]), lr=learning_rate)


    
    def addLayer(self, size, optimizeAll = True):
        oldLayerSize = len(self.layers[-4].weight)
        new_layer = nn.Linear(size, size)
        torch.nn.init.constant_(new_layer.weight, 0)
        new_layer.bias.data.fill_(0)
        with torch.no_grad():
            for i in range(min(oldLayerSize, len(new_layer.weight))):
                new_layer.weight[i, i] = 1
        self.layers.insert(len(self.layers) - 2, new_layer)
        self.layers.insert(len(self.layers) - 2, nn.ReLU())
        if optimizeAll:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(iter(self.layers[-4:-3]), lr=learning_rate)
   
    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Training_State():

    def __init__(self, envs = []):
        self.envs = envs
        self.scores = [[]]*len(envs)
        self.curr_env = 0

    def addEnvironment(self, env):
        self.envs.append(env)
        self.scores.append([])

    def advanceEnvironment(self):
        self.curr_env += 1
        if self.curr_env >= len(self.envs):
            self.curr_env = 0

    def getCurrentEnv(self):
        return self.envs[self.curr_env]

    def addScore(self, score):
        self.scores[self.curr_env].append(score)

    def getMeanScore(self):
        return np.mean(self.scores[self.curr_env][-100:])



def predict(state, ts):
    # Select an action (0 or 1) by running policy model
    # and choosing based on the probabilities in state
    new_state = [0]*input_size
    x = 0
    for i in range(len(ts.envs) - ts.curr_env - 2):
      x += ts.envs[i].observation_space.shape[0]

    for i in range(ts.envs[ts.curr_env].observation_space.shape[0]):
      new_state[i + x] = state[i]
    state = torch.from_numpy(np.array(new_state)).type(torch.FloatTensor)
    action_probs = policy(state)
    x = 0
    for i in range(len(ts.envs) - ts.curr_env - 2):
      x += ts.envs[i].action_space.n
    new_action_probs = action_probs.narrow(0, x, x+ts.envs[ts.curr_env].action_space.n)
    distribution = Categorical(new_action_probs)
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

def addEnvironment(env, policy, ts):
    ts.addEnvironment(env)
    policy.addNode(0, env.observation_space.shape[0]) # add nodes to the input layer
    policy.addNode(int(len(policy.layers)/2) - 1, env.action_space.n) # add nodes to the input layer

def train(episodes, envs):
    ts = Training_State(envs)
    addEnvironment(gym.make('MountainCar-v0'), policy, ts)
    for episode in range(episodes):
#        if episode % 50 == 0:
#          policy.addNode(random.randint(0,2))
#          for layer in policy.layers:
#              print(layer)
        
        state = ts.getCurrentEnv().reset()

        for time in range(1000):
            action = predict(state, ts)

            # Uncomment to render the visual state in a window
            #env[currentEnv].render()

            # Step through environment using chosen action
            state, reward, done, _ = ts.getCurrentEnv().step(action.item())

            # Save reward
            policy.episode_rewards.append(reward)
            if done:
                break

        update_policy()

        # Calculate score to determine when the environment has been solved
        ts.addScore(time)
        mean_score = ts.getMeanScore()

        if episode % 50 == 0:
            print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(episode, mean_score))
            #ts.advanceEnvironment()

        #if mean_score > env[currentEnv].spec.reward_threshold:
         #   print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
          #        .format(episode, mean_score, time))
           # for i in range(episode, episodes):
           #    policy.reward_history.append(mean_score)
            #break



envs = []
envs.append(gym.make('CartPole-v1'))
#envs.append(gym.make('MountainCar-v0'))

folderName = 'test'

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
hidden_size = 32

num_seeds = 10
num_episodes = 500

input_size = sum(i.observation_space.shape[0] for i in envs)
output_size = sum(i.action_space.n for i in envs)

rewards_history_by_run = []

for i in range(num_seeds):

	policy = Policy(input_size, output_size, hidden_size)
	pytorchSeed = random.randint(0, 1000)
	cartSeed = random.randint(0, 1000)
	for i in envs:
	  i.seed(cartSeed)
	torch.manual_seed(pytorchSeed)

	train(episodes=num_episodes, envs=envs)

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
