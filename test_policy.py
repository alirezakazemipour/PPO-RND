import gym
from utils import *


def evaluate_policy(env_name, agent, state_shape):
    env = make_atari(env_name)
    # env = gym.make(env_name)
    stacked_states = np.zeros(state_shape, dtype=np.uint8)
    s = env.reset()
    stacked_states = stack_states(stacked_states, s, True)
    episode_reward = 0
    done = False
    while not done:
        action, _, _ = agent.get_actions_and_values(stacked_states)
        next_s, r, done, _ = env.step(action)
        stacked_states = stack_states(stacked_states, next_s, False)
        episode_reward += np.sign(r)
    env.close()
    return episode_reward
