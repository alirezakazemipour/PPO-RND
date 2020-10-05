import torch
from torch import device
import time
import os
from utils import *


class Play:
    def __init__(self, env, agent, max_episode=10):
        self.env = make_atari(env)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_params()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "ppo_breakout" + ".avi", self.fourcc, 50.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)
        mean_ep_reward = []
        for ep in range(self.max_episode):
            self.env.seed(ep)
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            clipped_ep_reward = 0
            # x = input("Push any button to proceed...")
            for _ in range(self.env._max_episode_steps):
                action, _, _ = self.agent.get_actions_and_values(stacked_states)
                s_, r, done, info = self.env.step(action)
                episode_reward += r
                clipped_ep_reward += np.sign(r)
                if done and info["ale.lives"] == 0:
                    break
                stacked_states = stack_states(stacked_states, s_, False)
                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward:.3f}| "
                  f"clipped episode reward:{clipped_ep_reward:.3f}")
            mean_ep_reward.append(episode_reward)
            self.env.close()
            self.VideoWriter.release()
            cv2.destroyAllWindows()
        print(f"Mean episode reward:{sum(mean_ep_reward) / len(mean_ep_reward):0.3f}")
