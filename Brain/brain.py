from Brain.model import PolicyModel, PredictorModel, TargetModel
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam
from Common.utils import mean_of_list, RunningMeanStd, clip_grad_norm_


class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.obs_shape = self.config["obs_shape"]

        self.current_policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(self.obs_shape).to(self.device)
        self.target_model = TargetModel(self.obs_shape).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.total_trainable_params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters())
        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])

        self.state_rms = RunningMeanStd(shape=self.obs_shape[::-1])
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).permute([0, 3, 1, 2]).contiguous().to(self.device)
        with torch.no_grad():
            dist, int_value, ext_value, action_prob = self.current_policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), int_value.detach().cpu().numpy().squeeze(), \
               ext_value.detach().cpu().numpy().squeeze(), log_prob.cpu().numpy(), action_prob.cpu().numpy()

    def choose_mini_batch(self, states, actions, int_returns, ext_returns, advs, log_probs, next_states):
        idxes = np.random.randint(0, len(states), self.config["batch_size"])

        yield states[idxes], actions[idxes], int_returns[idxes], ext_returns[idxes], advs[idxes], \
              log_probs[idxes], next_states[idxes]

    @mean_of_list
    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, total_next_obs):

        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values,
                                dones, self.config["ext_gamma"])

        ext_values = np.concatenate(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = np.concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]

        self.state_rms.update(total_next_obs)
        total_next_obs = ((total_next_obs - self.state_rms.mean) / np.sqrt(self.state_rms.var)).clip(-5, 5)

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies = [], [], [], [], []
        for epoch in range(self.config["n_epochs"]):
            for state, action, int_return, ext_return, adv, old_log_prob, next_state in \
                    self.choose_mini_batch(states=states,
                                           actions=actions,
                                           int_returns=int_rets,
                                           ext_returns=ext_rets,
                                           advs=advs,
                                           log_probs=log_probs,
                                           next_states=total_next_obs):
                state = torch.ByteTensor(state).permute([0, 3, 1, 2]).contiguous().to(self.device)
                next_state = torch.Tensor(next_state).to(self.device)
                action = torch.ByteTensor(action).to(self.device)
                adv = torch.Tensor(adv).to(self.device)
                int_return = torch.Tensor(int_return).to(self.device)
                ext_return = torch.Tensor(ext_return).to(self.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.device)

                dist, int_value, ext_value, _ = self.current_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action)
                ratio = (new_log_prob - old_log_prob).exp()
                pg_loss = self.compute_ac_loss(ratio, adv)

                int_value_loss = self.mse_loss(int_value.squeeze(-1), int_return)
                ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_return)

                critic_loss = 0.5 * (int_value_loss + ext_value_loss)

                rnd_loss = self.calculate_rnd_loss(next_state)

                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy + rnd_loss
                self.optimize(total_loss)

                pg_losses.append(pg_loss.item())
                ext_v_losses.append(ext_value_loss.item())
                int_v_losses.append(int_value_loss.item())
                rnd_losses.append(rnd_loss.item())
                entropies.append(entropy.item())
                # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187
                # approxkl = 0.5 * ((new_log_prob - old_log_prob).pow(2)).mean()
                # maxkl = 0.5 * ((new_log_prob - old_log_prob).pow(2)).max()

        return pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies, int_values, int_rets, ext_values, ext_rets

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.total_trainable_params)
        self.optimizer.step()

    def get_gae(self, rewards, values, next_values, dones, gamma):

        returns = [[] for _ in range(self.config["n_workers"])]
        extended_values = np.zeros((self.config["n_workers"], self.config["rollout_length"] + 1))
        for worker in range(self.config["n_workers"]):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (extended_values[worker][step + 1]) * (1 - dones[worker][step]) \
                        - extended_values[worker][step]
                gae = delta + gamma * self.config["lambda"] * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return np.concatenate(returns)

    def calculate_int_rewards(self, next_states):
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5)
        next_states = from_numpy(next_states).float().to(self.device)
        predictor_encoded_features = self.predictor_model(next_states)
        target_encoded_features = self.target_model(next_states)

        int_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1)
        return int_reward.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def normalize_int_rewards(self, intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        intrinsic_returns = [[] for _ in range(self.config["n_workers"])]
        for worker in range(self.config["n_workers"]):
            rewems = 0
            for step in reversed(range(len(intrinsic_rewards[worker]))):
                rewems = rewems * self.config["int_gamma"] + intrinsic_rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        return intrinsic_rewards / np.sqrt(self.int_reward_rms.var)

    def compute_ac_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"]) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss

    def calculate_rnd_loss(self, next_state):
        encoded_target_features = self.target_model(next_state)
        encoded_predictor_features = self.predictor_model(next_state)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)
        mask = torch.rand(loss.size()).to(self.device)
        mask = (mask < self.config["predictor_proportion"]).float()
        loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return loss

    def load_params(self):
        checkpoint = torch.load("params.pth", map_location=self.device)
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.predictor_model.load_state_dict(checkpoint["predictor_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_rms.mean = checkpoint["state_rms_mean"]
        self.state_rms.var = checkpoint["state_rms_var"]
        self.state_rms.count = checkpoint["state_rms_count"]
        self.int_reward_rms.mean = checkpoint["int_reward_rms_mean"]
        self.int_reward_rms.var = checkpoint["int_reward_rms_var"]
        self.int_reward_rms.count = checkpoint["int_reward_rms_count"]
        iteration = checkpoint["iteration"]
        running_reward = checkpoint["running_reward"]
        episode = checkpoint["episode"]
        visited_rooms = checkpoint["visited_rooms"]

        return running_reward, iteration, episode, visited_rooms

    def set_to_eval_mode(self):
        self.current_policy.eval()
