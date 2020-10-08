import numpy as np
import cv2
import gym
from copy import deepcopy


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(img):
    img = rgb2gray(img)  # / 255.0 -> Do it later in order to open up more RAM !!!!
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=2)
    else:
        stacked_frames = stacked_frames[..., 1:]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=2)], axis=2)
    return stacked_frames


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def make_atari(env_id, max_episode_steps):
    main_env = gym.make(env_id)
    main_env._max_episode_steps = max_episode_steps
    assert 'NoFrameskip' in main_env.spec.id
    env = StickyActionEnv(main_env)
    env = RepeatActionEnv(env)
    env = MontezumaVisitedRoomEnv(env, 3)
    env = AddRandomStateToInfoEnv(env)

    return env


class StickyActionEnv:
    def __init__(self, env, p=0.25):
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.ale = self.env.ale
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        else:
            self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class RepeatActionEnv:
    def __init__(self, env):
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.ale = self.env.ale
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class MontezumaVisitedRoomEnv:
    def __init__(self, env, room_address):
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.ale = self.env.ale
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.room_address = room_address
        self.visited_rooms = set()  # Only stores unique numbers.

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        self.visited_rooms.add(ram[self.room_address])
        if done:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(visited_room=deepcopy(self.visited_rooms))
            self.visited_rooms.clear()
        return state, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class AddRandomStateToInfoEnv:
    def __init__(self, env):
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.ale = self.env.ale
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['rng_at_episode_start'] = self.rng_at_episode_start
        return state, reward, done, info

    def reset(self):
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
