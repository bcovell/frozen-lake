import gym
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as window
import matplotlib.pyplot as plt


def main():
    env: gym.Env = gym.make("FrozenLake-v1", is_slippery=True)
    env.render()

    q_learning(env, 10000, .99, .05)


def q_learning(env: gym.Env, num_episodes: int, discount: float, alpha: float):
    epsilon = 1
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def eps_greedy(action_values: np.ndarray):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(action_values)

    rewards = []
    for ep_idx in range(num_episodes):
        state = env.reset()

        while True:
            action = eps_greedy(q_table[state, :])
            new_state, reward, done, info = env.step(action)
            q_table[state, action] += alpha * (reward + discount * max(q_table[new_state, :]) - q_table[state, action])
            state = new_state
            if done:
                break

        rewards.append(reward)
        epsilon -= 1 / num_episodes

    filtered = np.mean(window(rewards, int(num_episodes / 100)), axis=1)
    filtered = np.pad(filtered, [num_episodes - len(filtered), 0])
    plt.plot(range(num_episodes), filtered)
    plt.title('q_learning for 4x4 frozen lake')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
    print(q_table)


if __name__ == '__main__':
    main()
