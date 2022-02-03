import numpy as np
import gym
from numpy.lib.stride_tricks import sliding_window_view as window
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env: gym.Env = gym.make("FrozenLake-v1", is_slippery=True)
    epsilon = 1  # exploration
    gamma = .99  # discount
    alpha = .05  # learning rate
    n = 3  # number of steps for look back
    q_table = np.zeros([env.observation_space.n, env.action_space.n])


    def eps_greedy(action_values: np.ndarray):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(action_values)

    final_rewards = []
    num_episodes = 10000
    for ep_idx in range(num_episodes):
        states = [env.reset()]
        actions = [eps_greedy(q_table[states[0], :])]
        rewards = []
        t_final = np.inf
        t = -1
        while True:
            t += 1
            if t < t_final:
                new_state, reward, done, info = env.step(actions[t])
                rewards.append(reward)
                states.append(new_state)
                if done:
                    t_final = t + 1
                else:
                    actions.append(eps_greedy(q_table[states[t + 1], :]))

            tao = t - n + 1
            if tao >= 0:
                discounted_reward = sum(
                    [gamma ** (i - tao) * rewards[i] for i in
                     range(tao, min(tao + n, t_final))]
                )
                if tao + n < t_final:
                    discounted_reward += gamma ** n * q_table[states[tao + n], actions[tao + n]]
                q_table[states[tao], actions[tao]] += alpha * (discounted_reward - q_table[states[tao], actions[tao]])

            if tao == t_final - 1:
                break

        final_rewards.append(rewards[-1])
        epsilon -= 1 / num_episodes
        epsilon = max(epsilon, 0)

    filtered = np.mean(window(final_rewards, int(num_episodes / 100)), axis=1)
    filtered = np.pad(filtered, [num_episodes - len(filtered), 0])
    plt.plot(range(num_episodes), filtered)
    plt.title(f'{n}-step sarsa for 4x4 frozen lake')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
    print(q_table)
