import time
import numpy as np
import torch
import os
import datetime
import glob
from collections import deque


class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.experiment = self.config["experiment"]
        self.brain = brain
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_ext_reward = 0
        self.running_ext_reward = 0
        self.running_int_reward = 0
        self.running_act_prob = 0
        self.running_training_logs = dict()
        self.x_pos = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_ext_r = 0  # It is not correct but does not matter.

        if self.config["do_train"] and self.config["train_from_scratch"]:
            self.create_wights_folder()
            self.experiment.log_parameters(self.config)

        self.exp_avg = lambda x, y: 0.99 * x + 0.01 * y if y != 0 else y

    def create_wights_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + self.log_dir)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, training_logs, int_reward, action_prob = args

        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)
        if iteration == 1:
            for k, v in training_logs.items():
                self.running_training_logs.update({k: v})
        else:
            for k, v in training_logs.items():
                self.running_training_logs[k] = self.exp_avg(self.running_training_logs[k], v)

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        self.experiment.log_metric("Episode Ext Reward", self.episode_ext_reward, self.episode)
        self.experiment.log_metric("Running Episode Ext Reward", self.running_ext_reward, self.episode)
        self.experiment.log_metric("Position", self.x_pos, self.episode)
        self.experiment.log_metric("Running last 10 Ext Reward", self.running_last_10_ext_r, self.episode)
        self.experiment.log_metric("Max Episode Ext Reward", self.max_episode_reward, self.episode)
        self.experiment.log_metric("Running Action Probability", self.running_act_prob, iteration)
        self.experiment.log_metric("Running Intrinsic Reward", self.running_int_reward, iteration)
        self.experiment.log_metric("Running PG Loss", self.running_training_logs["pg_loss"], iteration)
        self.experiment.log_metric("Running Ext Value Loss", self.running_training_logs["ext_value_loss"], iteration)
        self.experiment.log_metric("Running Int Value Loss", self.running_training_logs["int_value_loss"], iteration)
        self.experiment.log_metric("Running RND Loss", self.running_training_logs["rnd_loss"], iteration)
        self.experiment.log_metric("Running Entropy", self.running_training_logs["entropy"], iteration)
        self.experiment.log_metric("Running Intrinsic Explained variance",
                                   self.running_training_logs["int_ep"], iteration)
        self.experiment.log_metric("Running Extrinsic Explained variance",
                                   self.running_training_logs["ext_ep"], iteration)
        self.experiment.log_metric("Running grad norm", self.running_training_logs["grad_norm"], iteration)

        self.off()
        if iteration % self.config["interval"] == 0:
            print("Iter:{}| "
                  "EP:{}| "
                  "EP_Reward:{}| "
                  "EP_Running_Reward:{:.3f}| "
                  "Position:{:.1f}| "
                  "Iter_Duration:{:.3f}| "
                  "Time:{} "
                  .format(iteration,
                          self.episode,
                          self.episode_ext_reward,
                          self.running_ext_reward,
                          self.x_pos,
                          self.duration,
                          datetime.datetime.now().strftime("%H:%M:%S")
                          )
                  )
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_ext_reward, x_pos = args

        self.max_episode_reward = max(self.max_episode_reward, self.episode_ext_reward)

        self.running_ext_reward = self.exp_avg(self.running_ext_reward, self.episode_ext_reward)
        self.x_pos = self.exp_avg(self.x_pos, x_pos)

        self.last_10_ep_rewards.append(self.episode_ext_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_ext_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    def save_params(self, episode, iteration):
        torch.save({"policy_state_dict": self.brain.policy.state_dict(),
                    "predictor_model_state_dict": self.brain.predictor_model.state_dict(),
                    "target_model_state_dict": self.brain.target_model.state_dict(),
                    "optimizer_state_dict": self.brain.optimizer.state_dict(),
                    "state_rms_mean": self.brain.state_rms.mean,
                    "state_rms_var": self.brain.state_rms.var,
                    "state_rms_count": self.brain.state_rms.count,
                    "int_reward_rms_mean": self.brain.int_reward_rms.mean,
                    "int_reward_rms_var": self.brain.int_reward_rms.var,
                    "int_reward_rms_count": self.brain.int_reward_rms.count,
                    "iteration": iteration,
                    "episode": episode,
                    "running_ext_reward": self.running_ext_reward,
                    "running_int_reward": self.running_int_reward,
                    "running_act_prob": self.running_act_prob,
                    "running_training_logs": self.running_training_logs,
                    "x_pos": self.x_pos
                    },
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")

        self.brain.set_from_checkpoint(checkpoint)
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        self.running_ext_reward = checkpoint["running_ext_reward"]
        self.x_pos = checkpoint["x_pos"]
        self.episode = checkpoint["episode"]
        self.running_training_logs = checkpoint["running_training_logs"]
        self.running_act_prob = checkpoint["running_act_prob"]
        self.running_int_reward = checkpoint["running_int_reward"]

        return checkpoint["iteration"], self.episode
