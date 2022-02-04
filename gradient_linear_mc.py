import numpy as np
import gym


def main():
    env: gym.Env = gym.make("FrozenLake-v1", is_slippery=True)

    weights = np.zeros(env.observation_space.n)
    alpha = .1  # step size
    epsilon = 1
    num_episodes = 1000
    gamma = .99  # discount

    def feature_vector(state: int) -> np.ndarray:
        fv = np.zeros(env.observation_space.n)
        fv[state] = 1
        return fv

    def value(state: int) -> float:
        return float(np.dot(weights, feature_vector(state)))

    for _ in range(num_episodes):
        state = env.reset()
        while True:
            action = env.action_space.sample()  # random policy
            new_state, reward, done, info = env.step(action)
            gradient = feature_vector(state)
            weights += alpha * (reward + gamma * value(new_state) - value(state)) * gradient
            state = new_state
            if done:
                break

    print(weights.reshape([4, 4]))


if __name__ == '__main__':
    main()
