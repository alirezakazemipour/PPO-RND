from model import PolicyModel, PredictorModel, TargetModel
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils import explained_variance, RunningMeanStd, RewardForwardFilter


class Brain:
    def __init__(self, stacked_state_shape, state_shape, n_actions, device, n_workers, epochs, n_iters, epsilon, lr,
                 ext_gamma, int_gamma, int_adv_coeff, ext_adv_coeff, batch_size):
        self.stacked_state_shape = stacked_state_shape
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.n_workers = n_workers
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.initial_epsilon = epsilon
        self.epsilon = self.initial_epsilon
        self.lr = lr
        self.ext_gamma = ext_gamma
        self.int_gamma = int_gamma
        self.ext_adv_coeff = ext_adv_coeff
        self.int_adv_coeff = int_adv_coeff

        self.current_policy = PolicyModel(self.stacked_state_shape, self.n_actions).to(self.device)
        self.predictor_model = PredictorModel(self.state_shape).to(self.device)
        self.target_model = TargetModel(self.state_shape).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.optimizer = Adam(self.current_policy.parameters(), lr=self.lr, eps=1e-5)
        self._schedule_fn = lambda step: max(1.0 - float(step / self.n_iters), 0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._schedule_fn)

        self.state_rms = RunningMeanStd(shape=self.state_shape[:2])
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).byte().permute([0, 3, 1, 2]).to(self.device)
        with torch.no_grad():
            dist, int_value, ext_value = self.current_policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), int_value.detach().cpu().numpy().squeeze(), \
               ext_value.detach().cpu().numpy().squeeze(), log_prob.cpu().numpy()

    def choose_mini_batch(self, states, actions, int_returns, ext_returns, advs, int_values, ext_values, log_probs):
        idxes = np.random.randint(0, len(states), self.batch_size)

        yield states[idxes], actions[idxes], int_returns[idxes], ext_returns[idxes], advs[idxes], \
              int_values[idxes], ext_values[idxes], log_probs[idxes]

    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, total_next_obs):

        int_returns = self.get_gae(int_rewards, int_values.copy(), next_int_values,
                                   np.zeros_like(dones), self.int_gamma)
        ext_returns = self.get_gae(ext_rewards, ext_values.copy(), next_ext_values,
                                   dones, self.ext_gamma)

        ext_values = np.vstack(ext_values).reshape((len(ext_values[0]) * self.n_workers,))
        ext_advs = ext_returns - ext_values

        int_values = np.vstack(int_values).reshape((len(int_values[0]) * self.n_workers,))
        int_advs = int_returns - int_values

        advs = ext_advs * self.ext_adv_coeff + int_advs * self.int_adv_coeff

        self.state_rms.update(np.vstack(total_next_obs))
        for epoch in range(self.epochs):
            for state, action, int_return, ext_return, \
                adv, old_int_value, old_ext_value, old_log_prob in self.choose_mini_batch(states, actions,
                                                                                          int_returns, ext_returns,
                                                                                          advs, int_values,
                                                                                          ext_values, log_probs):
                state = torch.ByteTensor(state).permute([0, 3, 1, 2]).to(self.device)
                action = torch.Tensor(action).to(self.device)
                adv = torch.Tensor(adv).to(self.device)
                int_return = torch.Tensor(int_return).to(self.device)
                ext_return = torch.Tensor(ext_return).to(self.device)
                old_int_value = torch.Tensor(old_int_value).to(self.device)
                old_ext_value = torch.Tensor(old_ext_value).to(self.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.device)

                dist, int_value, ext_value = self.current_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.current_policy, state, action)
                ratio = (new_log_prob - old_log_prob).exp()
                actor_loss = self.compute_ac_loss(ratio, adv)

                clipped_value = old_value + torch.clamp(value.squeeze() - old_value, -self.epsilon, self.epsilon)
                clipped_v_loss = (clipped_value - q_value).pow(2)
                unclipped_v_loss = (value.squeeze() - q_value).pow(2)
                critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()

                total_loss = critic_loss + actor_loss - 0.01 * entropy
                self.optimize(total_loss)

        return total_loss.item(), entropy.item(), \
               explained_variance(values.reshape((len(returns[0]) * self.n_workers,)),
                                  returns.reshape((len(returns[0]) * self.n_workers,)))

    def schedule_lr(self):
        self.scheduler.step()

    def schedule_clip_range(self, iter):
        self.epsilon = max(1.0 - float(iter / self.n_iters), 0) * self.initial_epsilon

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        self.optimizer.step()

    def get_gae(self, rewards, values, next_values, dones, gamma, lam=0.95):

        returns = [[] for _ in range(self.n_workers)]
        extended_values = np.zeros((self.n_workers, len(rewards[0]) + 1))
        extended_dones = np.zeros((self.n_workers, len(rewards[0]) + 1))
        for worker in range(self.n_workers):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            extended_dones[worker] = np.append(dones[worker], 0)
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + \
                        gamma * (extended_values[worker][step + 1]) * (1 - extended_dones[worker][step + 1]) \
                        - extended_values[worker][step]
                gae = delta + gamma * lam * (1 - extended_dones[worker][step + 1]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return np.vstack(returns).reshape((len(returns[0]) * self.n_workers,))

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution, _ = model(states)
        return policy_distribution.log_prob(actions)

    def calculate_int_rewards(self, next_states):
        next_states = np.clip((next_states - self.state_rms.mean) / self.state_rms.var ** 0.5, -5, 5)
        next_states = np.expand_dims(next_states, 1)
        next_states = from_numpy(next_states).float().to(self.device)
        predictor_encoded_features = self.predictor_model(next_states)
        target_encoded_features = self.target_model(next_states)

        int_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1)
        return int_reward.detach().cpu().numpy()

    def normalize_int_rewards(self, intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong.
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        intrinsic_returns = [[] for _ in range(self.n_workers)]
        for worker in range(self.n_workers):
            rewems = 0
            for step in reversed(range(len(intrinsic_rewards[worker]))):
                rewems = rewems * self.int_gamma + intrinsic_rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        return intrinsic_rewards / np.sqrt(self.int_reward_rms.var)

    def compute_ac_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss

    def save_params(self, iteration, running_reward):
        torch.save({"current_policy_state_dict": self.current_policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "iteration": iteration,
                    "running_reward": running_reward,
                    "clip_range": self.epsilon},
                   "params.pth")

    def load_params(self):
        checkpoint = torch.load("params.pth", map_location=self.device)
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        running_reward = checkpoint["running_reward"]
        self.epsilon = checkpoint["clip_range"]

        return running_reward, iteration

    def set_to_eval_mode(self):
        self.current_policy.eval()
