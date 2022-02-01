import gym
import numpy as np


def main():
    env: gym.Env = gym.make("FrozenLake-v1", is_slippery=False)
    env.render()

    first_visit_mc(env, 1000, .99)


def first_visit_mc(env: gym.Env, num_episodes: int, discount: float):
    value = np.zeros(env.observation_space.n)
    returns = [[] for s in range(env.observation_space.n)]
    for idx in range(num_episodes):
        states = []
        actions = []
        rewards = []
        states.append(env.reset())
        while True:
            rand_action = env.action_space.sample()
            new_state, reward, done, info = env.step(rand_action)
            actions.append(rand_action)
            rewards.append(reward)
            states.append(new_state)
            # if done and rewards[-1] != 1:
            #     rewards[-1] = -1
            if done:
                break
        discounted_reward = 0
        visited_states = []
        for s, r in zip(states, rewards):
            discounted_reward = discount * discounted_reward + r
            if s not in visited_states:
                returns[s].append(discounted_reward)
                value[s] = np.mean(returns[s])
                visited_states.append(s)
    print(value.reshape([4, 4], order='C'))


if __name__ == '__main__':
    main()
