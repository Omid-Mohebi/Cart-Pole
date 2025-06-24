import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import time

GAMMA = 0.99
ALPHA = 0.1


def epsilon_greedy(model, state, epsilon=0.1):
    if np.random.random() < (1 - epsilon):
        q_values = model.predict_all_actions(state)
        return np.argmax(q_values)
    else:
        return model.env.action_space.sample()

def gather_samples(env, episodes=10000):
    samples = []
    for _ in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            state_action = np.concatenate((state, [action]))
            samples.append(state_action)
            state, reward, terminated, truncated, info = env.step(action)
    return samples

class Model:
    def __init__(self, env):
        self.env = env
        print("Gathering samples to fit the featurizer...")
        sample_env = gym.make('CartPole-v1')
        samples = gather_samples(sample_env)
        sample_env.close()
        
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dimensions = self.featurizer.n_components
        print("Featurizer fitted.")

        self.w = np.zeros(dimensions)

    def _transform(self, state, action):
        state_action = np.concatenate((state, [action]))
        return self.featurizer.transform([state_action])[0]

    def predict(self, state, action):
        x = self._transform(state, action)
        return x @ self.w

    def predict_all_actions(self, state):
        return [self.predict(state, action) for action in range(self.env.action_space.n)]

    def grad(self, state, action):
        return self._transform(state, action)