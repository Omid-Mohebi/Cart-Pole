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
    
def test_agent(model, env, episodes=20):
    total_reward = 0
    for _ in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = epsilon_greedy(model, state, epsilon=0)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / episodes

def watch_agent(model, env):
    state, info = env.reset()
    episode_reward = 0
    terminated = False
    truncated = False
    
    print("\nWatching trained agent. Press Ctrl+C in the terminal to stop.")
    try:
        while not (terminated or truncated):
            action = epsilon_greedy(model, state, epsilon=0)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopping agent watch.")
    print(f"Final reward for watched episode: {episode_reward}")


if __name__ == '__main__':
    train_env = gym.make('CartPole-v1')
    
    model = Model(train_env)
    
    reward_per_episode = []
    episodes = 2000
    print(f"\nStarting training for {episodes} episodes...")

    for i in range(episodes):
        state, info = train_env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = epsilon_greedy(model, state)
            new_state, reward, terminated, truncated, info = train_env.step(action)
            
            done = terminated or truncated
            if done:
                target = reward
            else:
                q_values_next = model.predict_all_actions(new_state)
                target = reward + GAMMA * np.max(q_values_next)

            gradient = model.grad(state, action)
            error = target - model.predict(state, action)
            model.w += ALPHA * error * gradient

            episode_reward += reward
            state = new_state

        reward_per_episode.append(episode_reward)

        if (i + 1) % 100 == 0:
            print(f"Episode {i + 1}/{episodes} | Total Reward: {episode_reward}")

        if i >= 100 and np.mean(reward_per_episode[-100:]) >= 475.0:
            print(f"\nEnvironment solved in {i + 1} episodes!")
            break
            
    train_env.close()
    print("Training finished.")

    test_env = gym.make('CartPole-v1')
    avg_test_reward = test_agent(model, test_env)
    print(f'\nAverage test reward over 20 episodes: {avg_test_reward:.2f}')
    test_env.close()

    plt.plot(reward_per_episode)
    moving_avg = [np.mean(reward_per_episode[max(0, i-100):i+1]) for i in range(len(reward_per_episode))]
    plt.plot(moving_avg, color='red', linewidth=2, label='100-episode moving average')
    plt.title('Reward per Episode during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    for _ in range(10):
        watch_env = gym.make('CartPole-v1', render_mode='human')
        watch_agent(model, watch_env)
        watch_env.close()