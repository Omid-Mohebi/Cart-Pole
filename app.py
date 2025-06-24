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