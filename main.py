from runner import Worker
from torch.multiprocessing import Process, Pipe
import numpy as np
from brain import Brain
import gym
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from test_policy import evaluate_policy
from play import Play

env_name = "MontezumaRevengeNoFrameskip-v4"
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
n_workers = 2  # 128
stacked_state_shape = (84, 84, 4)
state_shape = (84, 84, 1)
device = "cuda"
iterations = int(30e3)
max_episode_steps = int(18e3)
log_period = 50
T = 128
epochs = 4
n_mini_batch = 4
batch_size = T * n_workers // n_mini_batch
lr = 1e-4
ext_gamma = 0.999
int_gamma = 0.99
ext_adv_coeff = 2
int_adv_coeff = 1
ent_coeff = 0.001
clip_range = 0.1
predictor_proportion = 0.25
pre_normalization_steps = 50
LOAD_FROM_CKP = False
Train = True


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    brain = Brain(stacked_state_shape, state_shape, n_actions, device, n_workers, epochs, iterations, clip_range, lr,
                  ext_gamma, int_gamma, int_adv_coeff, ext_adv_coeff, ent_coeff, batch_size, predictor_proportion)
    if Train:
        if LOAD_FROM_CKP:
            running_reward, init_iteration = brain.load_params()
        else:
            init_iteration = 0
            running_reward = 0

        workers = [Worker(i, stacked_state_shape, env_name, max_episode_steps) for i in range(n_workers)]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            parents.append(parent_conn)
            p.start()

        print("---Pre_normalization started.---")
        states = []
        for t in range(T * pre_normalization_steps):
            for worker_id, parent in enumerate(parents):
                parent.recv()  # Only collects next_states for normalization,
                # otherwise there would have been repetitive states in normalization input.
            actions = np.random.randint(0, n_actions, n_workers)
            for parent, a in zip(parents, actions):
                parent.send(a)

            for parent in parents:
                s_, *_ = parent.recv()
                states.append(s_[:, :, -1])

            if len(states) % (n_workers * T) == 0:
                brain.state_rms.update(np.stack(states))
                states = []
        print("---Pre_normalization is done.---")

        for iteration in tqdm(range(init_iteration + 1, iterations + 1)):
            start_time = time.time()
            total_states = np.zeros((n_workers, T,) + stacked_state_shape)
            total_actions = np.zeros((n_workers, T))
            total_int_rewards = np.zeros((n_workers, T))
            total_ext_rewards = np.zeros((n_workers, T))
            total_dones = np.zeros((n_workers, T))
            total_int_values = np.zeros((n_workers, T))
            total_ext_values = np.zeros((n_workers, T))
            total_log_probs = np.zeros((n_workers, T))
            next_states = np.zeros((n_workers,) + stacked_state_shape)
            total_next_obs = np.zeros((n_workers, T,) + state_shape[:2])
            next_values = np.zeros(n_workers)

            for t in range(T):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_int_values[:, t], total_ext_values[:, t], total_log_probs[:, t] = \
                    brain.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(int(a))

                for worker_id, parent in enumerate(parents):
                    s_, r, d = parent.recv()
                    total_ext_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_
                    total_next_obs[worker_id, t] = s_[..., -1]
                total_int_rewards[:, t] = brain.calculate_int_rewards(next_states[..., -1])
            _, next_int_values, next_ext_values, _ = brain.get_actions_and_values(next_states, batch=True)
            next_ext_values *= (1 - total_dones[:, -1])

            total_int_rewards = brain.normalize_int_rewards(total_int_rewards)

            total_states = total_states.reshape((n_workers * T,) + stacked_state_shape)
            total_next_obs = total_next_obs.reshape((n_workers * T,) + state_shape[:2])
            total_actions = total_actions.reshape(n_workers * T)
            total_log_probs = total_actions.reshape(n_workers * T)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            total_loss, entropy, ev = brain.train(total_states, total_actions, total_int_rewards,
                                                  total_ext_rewards, total_dones, total_int_values, total_ext_values,
                                                  total_log_probs, next_int_values, next_ext_values, total_next_obs)
            # brain.schedule_lr()
            # brain.schedule_clip_range(iteration)
            episode_reward = evaluate_policy(env_name, brain, stacked_state_shape)

            if iteration == 1:
                running_reward = episode_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * episode_reward

            if iteration % log_period == 0:
                print(f"Iter: {iteration}| "
                      f"Ep_reward: {episode_reward:.3f}| "
                      f"Running_reward: {running_reward:.3f}| "
                      f"Total_loss: {total_loss:.3f}| "
                      f"Explained variance:{ev:.3f}| "
                      f"Entropy: {entropy:.3f}| "
                      f"Iter_duration: {time.time() - start_time:.3f}| "
                      f"Lr: {brain.scheduler.get_last_lr()}| "
                      f"Clip_range:{brain.epsilon:.3f}")
                brain.save_params(iteration, running_reward)

            with SummaryWriter(env_name + "/logs") as writer:
                writer.add_scalar("running reward", running_reward, iteration)
                writer.add_scalar("episode reward", episode_reward, iteration)
                writer.add_scalar("explained variance", ev, iteration)
                writer.add_scalar("loss", total_loss, iteration)
                writer.add_scalar("entropy", entropy, iteration)
    else:
        play = Play(env_name, brain)
        play.evaluate()
