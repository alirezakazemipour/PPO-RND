from runner import Worker
from torch.multiprocessing import Process, Pipe
import numpy as np
from brain import Brain
import gym
from tqdm import tqdm
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from play import Play
import torch

env_name = "MontezumaRevengeNoFrameskip-v4"
time_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
n_workers = 2  # 128 # 32
stacked_state_shape = (84, 84, 4)
state_shape = (84, 84, 1)
device = torch.device("cuda")
iterations = int(30e3)
max_episode_steps = int(18e3)
log_period = 25
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
predictor_proportion = 32 / n_workers
pre_normalization_steps = 50
LOAD_FROM_CKP = False
Train = True


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':

    brain = Brain(stacked_state_shape=stacked_state_shape,
                  state_shape=state_shape,
                  n_actions=n_actions,
                  device=device,
                  n_workers=n_workers,
                  epochs=epochs,
                  n_iters=iterations,
                  epsilon=clip_range,
                  lr=lr,
                  ext_gamma=ext_gamma,
                  int_gamma=int_gamma,
                  int_adv_coeff=int_adv_coeff,
                  ext_adv_coeff=ext_adv_coeff,
                  ent_coeff=ent_coeff,
                  batch_size=batch_size,
                  predictor_proportion=predictor_proportion)
    if Train:
        if LOAD_FROM_CKP:
            running_ext_reward, init_iteration, episode = brain.load_params()
        else:
            init_iteration = 0
            running_ext_reward = 0
            episode = 0

        workers = [Worker(i, stacked_state_shape, env_name, max_episode_steps) for i in range(n_workers)]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            p.daemon = True
            parents.append(parent_conn)
            p.start()

        if not LOAD_FROM_CKP:
            print("---Pre_normalization started.---")
            states = []
            actions = np.random.randint(0, n_actions, (T * pre_normalization_steps, n_workers))
            for t in range(T * pre_normalization_steps):
                for worker_id, parent in enumerate(parents):
                    parent.recv()  # Only collects next_states for normalization.
                for parent, a in zip(parents, actions[t]):
                    parent.send(a)

                for parent in parents:
                    s_, *_ = parent.recv()
                    states.append(s_[..., -1].reshape(1, 84, 84))

                if len(states) % (n_workers * T) == 0:
                    brain.state_rms.update(np.stack(states))
                    states = []
            print("---Pre_normalization is done.---")

        episode_ext_reward = 0
        visited_rooms = set([1])
        for iteration in tqdm(range(init_iteration + 1, iterations + 1)):
            start_time = time.time()
            total_states = np.zeros((n_workers, T,) + stacked_state_shape, dtype=np.uint8)
            total_actions = np.zeros((n_workers, T), dtype=np.uint8)
            total_int_rewards = np.zeros((n_workers, T))
            total_ext_rewards = np.zeros((n_workers, T), dtype=np.int)
            total_dones = np.zeros((n_workers, T), dtype=np.bool)
            total_int_values = np.zeros((n_workers, T))
            total_ext_values = np.zeros((n_workers, T))
            total_log_probs = np.zeros((n_workers, T))
            next_states = np.zeros((n_workers,) + stacked_state_shape, dtype=np.uint8)
            total_next_obs = np.zeros((n_workers, T,) + state_shape[::-1], dtype=np.uint8)
            next_values = np.zeros(n_workers)

            for t in range(T):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_int_values[:, t], total_ext_values[:, t], total_log_probs[:, t] = \
                    brain.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(a)

                infos = []
                for worker_id, parent in enumerate(parents):
                    s_, r, d, info = parent.recv()
                    infos.append(info)
                    total_ext_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_
                    total_next_obs[worker_id, t] = s_[..., -1]

                episode_ext_reward += total_ext_rewards[0, t]
                if total_dones[0, t]:
                    episode += 1
                    visited_rooms = infos[0]["episode"]["visited_room"]
                    if episode == 1:
                        running_ext_reward = episode_ext_reward
                    else:
                        running_ext_reward = 0.99 * running_ext_reward + 0.01 * episode_ext_reward
                    episode_ext_reward = 0

            total_next_obs = total_next_obs.reshape((n_workers * T,) + state_shape[::-1])
            total_int_rewards = brain.calculate_int_rewards(total_next_obs, T)
            _, next_int_values, next_ext_values, _ = brain.get_actions_and_values(next_states, batch=True)
            # next_ext_values *= (1 - total_dones[:, -1])

            total_int_rewards = brain.normalize_int_rewards(total_int_rewards)

            total_states = total_states.reshape((n_workers * T,) + stacked_state_shape)
            total_actions = np.concatenate(total_actions)
            total_log_probs = np.concatenate(total_log_probs)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            training_logs = brain.train(states=total_states, actions=total_actions, int_rewards=total_int_rewards,
                                        ext_rewards=total_ext_rewards, dones=total_dones,
                                        int_values=total_int_values, ext_values=total_ext_values,
                                        log_probs=total_log_probs, next_int_values=next_int_values,
                                        next_ext_values=next_ext_values, total_next_obs=total_next_obs)

            if iteration % log_period == 0:
                print(f"Iter: {iteration}| "
                      f"Visited_rooms: {visited_rooms}| "
                      f"Ep_ext_reward: {episode_ext_reward:.3f}| "
                      f"Running_ext_reward: {running_ext_reward:.3f}| "
                      f"Int_reward: {total_int_rewards[0].mean():.3f}| "
                      f"Iter_duration: {time.time() - start_time:.3f}| ")
                brain.save_params(episode, iteration, running_ext_reward)

            with SummaryWriter(env_name + "/logs/" + time_dir) as writer:
                writer.add_scalar("Running_ext_reward", running_ext_reward, episode)
                writer.add_scalar("Visited rooms", len(list(visited_rooms)), episode)
                writer.add_scalar("Int_reward", total_int_rewards[0].mean(), iteration)
                writer.add_scalar("Actor_loss", training_logs[0], iteration)
                writer.add_scalar("Ext value loss", training_logs[1], iteration)
                writer.add_scalar("Int value loss", training_logs[2], iteration)
                writer.add_scalar("RND loss", training_logs[3], iteration)
                writer.add_scalar("Entropy", training_logs[4], iteration)
                writer.add_scalar("Explained int variance", training_logs[5], iteration)
                writer.add_scalar("Explained ext variance", training_logs[6], iteration)

    else:
        play = Play(env_name, brain)
        play.evaluate()
