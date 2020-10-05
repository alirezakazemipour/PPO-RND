import numpy as np
import cv2
import gym


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


def make_atari(env_id):
    main_env = gym.make(env_id)
    assert 'NoFrameskip' in main_env.spec.id
    env = NoopResetEnv(main_env)
    env = RepeatActionEnv(env)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in main_env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env


class NoopResetEnv:
    def __init__(self, env):
        self.noop_max = 30
        self.noop_action = 0
        self.env = env
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.ale = self.env.ale
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self.observation_space = self.env.observation_space

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        state = None
        for _ in range(noops):
            state, _, done, _ = self.env.step(self.noop_action)
            if done:
                state = self.env.reset()

        return state

    def step(self, action):
        return self.env.step(action)

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


class EpisodicLifeEnv:
    def __init__(self, env):
        self.env = env
        self.ale = self.env.ale
        self.unwrapped = self.env.unwrapped
        self.observation_space = env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        self.natural_done = True
        self.lives = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.natural_done = done

        if self.lives > info["ale.lives"] > 0:
            done = True
        self.lives = info["ale.lives"]

        return state, reward, done, info

    def reset(self):
        if self.natural_done:
            state = self.env.reset()
        else:
            state, _, _, _ = self.env.step(0)
        self.lives = self.env.ale.lives()
        return state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class FireResetEnv:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.ale = self.env.ale
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
