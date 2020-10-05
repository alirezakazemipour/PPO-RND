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

env_name = "BreakoutNoFrameskip-v4"
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
n_workers = 8
state_shape = (84, 84, 4)
device = "cuda"
iterations = int(2e4)
log_period = 50
T = 128
epochs = 3
lr = 2.5e-4
clip_range = 0.1
LOAD_FROM_CKP = True
Train = False


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    brain = Brain(state_shape, n_actions, device, n_workers, epochs, iterations, clip_range, lr)
    if Train:
        if LOAD_FROM_CKP:
            running_reward, init_iteration = brain.load_params()
        else:
            init_iteration = 0
            running_reward = 0

        workers = [Worker(i, state_shape, env_name) for i in range(n_workers)]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            parents.append(parent_conn)
            p.start()

        for iteration in tqdm(range(init_iteration + 1, iterations + 1)):
            start_time = time.time()
            total_states = np.zeros((n_workers, T,) + state_shape)
            total_actions = np.zeros((n_workers, T))
            total_rewards = np.zeros((n_workers, T))
            total_dones = np.zeros((n_workers, T))
            total_values = np.zeros((n_workers, T))
            total_log_probs = np.zeros((n_workers, T))
            next_states = np.zeros((n_workers,) + state_shape)
            next_values = np.zeros(n_workers)

            for t in range(T):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_values[:, t], total_log_probs[:, t] =\
                    brain.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(int(a))

                for worker_id, parent in enumerate(parents):
                    s_, r, d = parent.recv()
                    total_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_
            _, next_values, _ = brain.get_actions_and_values(next_states, batch=True)
            next_values *= (1 - total_dones[:, -1])

            # total_states = total_states.reshape((n_workers * T,) + state_shape)
            # total_actions = total_actions.reshape(n_workers * T)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            total_loss, entropy, ev = brain.train(total_states, total_actions, total_rewards,
                                              total_dones, total_values, total_log_probs, next_values)
            brain.schedule_lr()
            brain.schedule_clip_range(iteration)
            episode_reward = evaluate_policy(env_name, brain, state_shape)

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
