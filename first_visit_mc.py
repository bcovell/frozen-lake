import gym
import numpy as np
import matplotlib.pyplot as plt


def main():
    env: gym.Env = gym.make("FrozenLake-v1", is_slippery=False)
    env.render()

    first_visit_mc(env, 100, .99)


def first_visit_mc(env: gym.Env, num_episodes: int, discount: float):
    exploration = np.linspace(1, 0, num_episodes)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    return_sums = np.zeros([env.observation_space.n, env.action_space.n])
    return_counts = np.zeros([env.observation_space.n, env.action_space.n])
    last_10_rewards = [0] * 10
    avg_rewards = []
    for ep_idx, epsilon in zip(range(num_episodes), exploration):
        states = []
        actions = []
        rewards = []
        states.append(env.reset())
        while True:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[states[-1], :])
            new_state, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            states.append(new_state)
            if done:
                break
        last_10_rewards.pop(0)
        last_10_rewards.append(reward)
        avg_rewards.append(sum(last_10_rewards) / len(last_10_rewards))
        discounted_reward = 0
        visited_state_actions = []
        for s, a, r in zip(reversed(states[:-1]), reversed(actions),
                           reversed(rewards)):
            discounted_reward = discount * discounted_reward + r
            if s not in visited_state_actions:
                return_sums[s, a] += discounted_reward
                return_counts[s, a] += 1
                q_table[s, a] = return_sums[s, a] / return_counts[s, a]
                visited_state_actions.append((s, a))
    plt.plot(np.arange(0, num_episodes), avg_rewards)
    plt.xlabel('episodes')
    plt.ylabel('average reward')
    plt.show()


if __name__ == '__main__':
    main()
