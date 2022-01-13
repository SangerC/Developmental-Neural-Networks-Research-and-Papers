import gym
cart = gym.make('CartPole-v0')
mountain = gym.make('MoutainCar-v1')

print(cart.action_space)
print(cart.observation_space)
print(mountain.action_space)
print(mountain.observation_space)


"""
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
"""
