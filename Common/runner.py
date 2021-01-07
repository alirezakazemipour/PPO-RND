from .utils import *
import os


class Worker:
    def __init__(self, id, **config):
        self._id = id
        self._config = config
        self._state_shape = self._config["state_shape"]
        self._max_episode_steps = self._config["max_frames_per_episode"]
        self._env = make_mario(self._config["env_name"], self._max_episode_steps)
        self._stacked_states = np.zeros(self._state_shape, dtype=np.uint8)
        self._score = 0
        self._pos = 0
        self._episode_reward = 0
        self._lives = 2
        self._life_done = False

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Trajectories"):
            os.mkdir("Trajectories")
        self.VideoWriter = cv2.VideoWriter("Trajectories/" + "worker_" + f"{self.id}" + ".avi", self.fourcc, 20.0,
                                           self._env.observation_space.shape[1::-1])
        self._frames = []
        self.reset()
        print(f"Worker {self.id}: initiated.")

    @property
    def id(self):
        return self._id

    def render(self):
        self._env.render()

    def reset(self):
        state = self._env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)
        self._score = 0
        self._pos = 0
        self._frames = []
        self._episode_reward = 0
        self._lives = 2
        self._life_done = False

    def step(self, conn):
        t = 1
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, _, d, info = self._env.step(action)
            t += 1
            if t % self._max_episode_steps == 0:
                d = True

            if info["flag_get"]:
                r = 1
            else:
                r = 0

            if info["life"] < self._lives:
                self._life_done = True
                self._lives = info["life"]

            self._pos = info["x_pos"]
            self._episode_reward += r

            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            self._frames.append(next_state)

            conn.send(dict(next_state=self._stacked_states,
                           reward=r,
                           real_done=d or self._life_done,
                           game_done=d,
                           info=info
                           )
                      )
            self._life_done = False

            if info["flag_get"]:
                print("\n---------------------------------------")
                print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩\n"
                      f"Worker {self._id}: got the flag!!!!!!!\n"
                      f"Episode Reward: {self._episode_reward:.1f}\n"
                      f"Position: {self._pos}\n"
                      f"World: {info['world']}\n"
                      f"Stage: {info['stage']}")
                print("---------------------------------------")

                for frame in self._frames:
                    self.VideoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if d:
                self.reset()
                t = 1
