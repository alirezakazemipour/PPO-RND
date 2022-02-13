from Common.utils import *
from torch.multiprocessing import Process


class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.conn = conn
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_frames_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(self.env_name, self.max_episode_steps)
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)

    def run(self):
        t = 1
        while True:
            self.conn.send(self._stacked_states)
            action = self.conn.recv()
            next_state, r, d, info = self.env.step(action)
            t += 1
            if t % self.max_episode_steps == 0:
                d = True
            if self.config["render"]:
                self.render()
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            self.conn.send((self._stacked_states, np.sign(r), d, info))
            if d:
                self.reset()
                t = 1
