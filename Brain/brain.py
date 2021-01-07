from .model import PolicyModel, PredictorModel, TargetModel
import torch
from torch import from_numpy
import numpy as np
from numpy import concatenate  # Make coder faster.
from torch.optim.adam import Adam
from Common import RunningMeanStd, explained_variance
from torch.optim.lr_scheduler import LambdaLR

torch.backends.cudnn.benchmark = True


class Brain:
    def __init__(self, **config):
        self.config = config
        self.mini_batch_size = self.config["batch_size"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.obs_shape = self.config["obs_shape"]
        self.epsilon = self.config["clip_range"]

        self.policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(self.obs_shape).to(self.device)
        self.target_model = TargetModel(self.obs_shape).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.total_trainable_params = list(self.policy.parameters()) + list(self.predictor_model.parameters())
        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])
        self.schedule_fn = lambda step: max(1.0 - float(step / self.config["total_rollouts_per_env"]), 0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.schedule_fn)

        self.state_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, int_value, ext_value, action_prob = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), int_value.cpu().numpy().squeeze(), \
               ext_value.cpu().numpy().squeeze(), log_prob.cpu().numpy(), action_prob.cpu().numpy()

    def choose_mini_batch(self, states,
                          actions,
                          int_returns,
                          ext_returns,
                          advs,
                          log_probs,
                          next_states,
                          int_values,
                          ext_values):
        states = torch.ByteTensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.ByteTensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        int_returns = torch.Tensor(int_returns).to(self.device)
        ext_returns = torch.Tensor(ext_returns).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)
        int_values = torch.Tensor(int_values).to(self.device)
        ext_values = torch.Tensor(ext_values).to(self.device)

        indices = np.random.randint(0, len(states), (self.config["n_mini_batch"], self.mini_batch_size))

        for idx in indices:
            yield states[idx], actions[idx], int_returns[idx], ext_returns[idx], advs[idx], \
                  log_probs[idx], next_states[idx], int_values[idx], ext_values[idx]

    def train(self, states, actions, int_rewards,
              ext_rewards, dones, int_values, ext_values,
              log_probs, next_int_values, next_ext_values, total_next_obs):

        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values,
                                dones, self.config["ext_gamma"])

        ext_values = concatenate(ext_values)
        ext_advs = ext_rets - ext_values

        int_values = concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]

        self.state_rms.update(total_next_obs)
        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies, grad_norms = [], [], [], [], [], []
        for epoch in range(self.config["n_epochs"]):
            for state, action, int_return, ext_return, adv, old_log_prob, next_state, old_int_value, old_ext_value in \
                    self.choose_mini_batch(states=states,
                                           actions=actions,
                                           int_returns=int_rets,
                                           ext_returns=ext_rets,
                                           advs=advs,
                                           log_probs=log_probs,
                                           next_states=total_next_obs,
                                           int_values=int_values,
                                           ext_values=ext_values):
                dist, int_value, ext_value, *_ = self.policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = dist.log_prob(action)
                ratio = (new_log_prob - old_log_prob).exp()
                pg_loss = self.compute_pg_loss(ratio, adv)

                clipped_ext_value = old_ext_value + torch.clamp(ext_value.squeeze() - old_ext_value,
                                                                -self.epsilon, self.epsilon)
                clipped_ext_v_loss = (clipped_ext_value - ext_return).pow(2)
                unclipped_ext_v_loss = (ext_value.squeeze() - ext_return).pow(2)
                ext_value_loss = 0.5 * torch.max(clipped_ext_v_loss, unclipped_ext_v_loss).mean()

                clipped_int_value = old_int_value + torch.clamp(int_value.squeeze() - old_int_value,
                                                                -self.epsilon, self.epsilon)
                clipped_int_v_loss = (clipped_int_value - int_return).pow(2)
                unclipped_int_v_loss = (int_value.squeeze() - int_return).pow(2)
                int_value_loss = 0.5 * torch.max(clipped_int_v_loss, unclipped_int_v_loss).mean()

                # int_value_loss = self.mse_loss(int_value.squeeze(-1), int_return)
                # ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_return)

                critic_loss = 0.5 * (int_value_loss + ext_value_loss)

                rnd_loss = self.calculate_rnd_loss(next_state)

                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy + rnd_loss
                grad_norm = self.optimize(total_loss)

                pg_losses.append(pg_loss.item())
                ext_v_losses.append(ext_value_loss.item())
                int_v_losses.append(int_value_loss.item())
                rnd_losses.append(rnd_loss.item())
                entropies.append(entropy.item())
                grad_norms.append(grad_norm.item())
                # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L187

        iteration_log = dict(pg_loss=sum(pg_losses) / len(pg_losses),
                             ext_value_loss=sum(ext_v_losses) / len(ext_v_losses),
                             int_value_loss=sum(int_v_losses) / len(int_v_losses),
                             rnd_loss=sum(rnd_losses) / len(rnd_losses),
                             entropy=sum(entropies) / len(entropies),
                             grad_norm=sum(grad_norms) / len(grad_norms),
                             int_ep=explained_variance(int_values, int_rets),
                             ext_ep=explained_variance(ext_values, ext_rets)
                             )
        return iteration_log

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.total_trainable_params, self.config["max_grad_norm"])
        self.optimizer.step()
        return grad_norm

    def schedule_lr(self):
        self.scheduler.step()

    def schedule_clip_range(self, iter):
        self.epsilon = max(1.0 - float(iter / self.config["total_rollouts_per_env"]), 0) * self.config["clip_range"]

    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"]  # Make code faster.
        returns = [[] for _ in range(self.config["n_workers"])]
        extended_values = np.zeros((self.config["n_workers"], self.config["rollout_length"] + 1))
        for worker in range(self.config["n_workers"]):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (extended_values[worker][step + 1]) * (1 - dones[worker][step]) \
                        - extended_values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return concatenate(returns)

    def calculate_int_rewards(self, next_states, batch=True):
        if not batch:
            next_states = np.expand_dims(next_states, 0)
        next_states = np.clip((next_states - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")  # dtype to avoid '.float()' call for pytorch.
        next_states = from_numpy(next_states).to(self.device)
        predictor_encoded_features = self.predictor_model(next_states)
        target_encoded_features = self.target_model(next_states)

        int_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1)
        if not batch:
            return int_reward.detach().cpu().numpy()
        else:
            return int_reward.detach().cpu().numpy().reshape((self.config["n_workers"], self.config["rollout_length"]))

    def normalize_int_rewards(self, intrinsic_rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        gamma = self.config["int_gamma"]  # Make code faster.
        intrinsic_returns = [[] for _ in range(self.config["n_workers"])]
        for worker in range(self.config["n_workers"]):
            rewems = 0
            for step in reversed(range(self.config["rollout_length"])):
                rewems = rewems * gamma + intrinsic_rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.int_reward_rms.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        return intrinsic_rewards / (self.int_reward_rms.var ** 0.5)

    def compute_pg_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss

    def calculate_rnd_loss(self, next_state):
        encoded_target_features = self.target_model(next_state)
        encoded_predictor_features = self.predictor_model(next_state)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)
        mask = torch.rand(loss.size(), device=self.device)
        mask = (mask < self.config["predictor_proportion"]).float()
        loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return loss

    def set_from_checkpoint(self, checkpoint):
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
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

    def set_to_eval_mode(self):
        self.policy.eval()
