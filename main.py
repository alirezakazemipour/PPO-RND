from Common.runner import Worker
from Common.play import Play
from Common.config import get_params
from Common.logger import Logger
from torch.multiprocessing import Process, Pipe
import numpy as np
from Brain.brain import Brain
import gym
from tqdm import tqdm


# TODO: Load weights !!!

def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    config = get_params()

    test_env = gym.make(config["env_name"])
    config.update({"n_actions": test_env.action_space.n})
    test_env.close()

    config.update({"batch_size": config["rollout_length"] * config["n_workers"] // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})

    brain = Brain(**config)
    logger = Logger(brain, **config)

    if config["do_train"]:
        if not config["train_from_scratch"]:
            running_ext_reward, init_iteration, episode, visited_rooms = brain.load_params()
        else:
            init_iteration = 0
            running_ext_reward = 0
            episode = 0
            visited_rooms = set([1])

        workers = [Worker(i, **config) for i in range(config["n_workers"])]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            p.daemon = True
            parents.append(parent_conn)
            p.start()

        if not config["train_from_scratch"]:
            print("---Pre_normalization started.---")
            states = []
            total_pre_normalization_steps = config["rollout_length"] * config["pre_normalization_steps"]
            actions = np.random.randint(0, config["n_actions"], (total_pre_normalization_steps, config["n_workers"]))
            for t in range(total_pre_normalization_steps):
                for worker_id, parent in enumerate(parents):
                    parent.recv()  # Only collects next_states for normalization.
                for parent, a in zip(parents, actions[t]):
                    parent.send(a)

                for parent in parents:
                    s_, *_ = parent.recv()
                    states.append(s_[..., -1].reshape(1, 84, 84))

                if len(states) % (config["n_workers"] * config["rollout_length"]) == 0:
                    brain.state_rms.update(np.stack(states))
                    states = []
            print("---Pre_normalization is done.---")

        logger.on()
        episode_ext_reward = 0
        rollout_base_shape = config["n_workers"], config["rollout_length"]
        for iteration in tqdm(range(init_iteration + 1, config["total_rollouts_per_env"] + 1)):
            total_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)
            total_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
            total_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
            total_int_rewards = np.zeros(rollout_base_shape)
            total_ext_rewards = np.zeros(rollout_base_shape, dtype=np.int8)
            total_dones = np.zeros(rollout_base_shape, dtype=np.bool)
            total_int_values = np.zeros(rollout_base_shape)
            total_ext_values = np.zeros(rollout_base_shape)
            total_log_probs = np.zeros(rollout_base_shape)
            next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)
            total_next_obs = np.zeros(rollout_base_shape + config["obs_shape"][::-1], dtype=np.uint8)
            next_values = np.zeros(rollout_base_shape[0])

            for t in range(config["rollout_length"]):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_int_values[:, t], total_ext_values[:, t], total_log_probs[:, t], \
                total_action_probs[:, t] = brain.get_actions_and_values(total_states[:, t], batch=True)
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
                    if "visited_room" in infos[0]["episode"]:
                        visited_rooms = infos[0]["episode"]["visited_room"]
                        logger.log_episode(episode, episode_ext_reward, visited_rooms)
                    episode_ext_reward = 0

            total_next_obs = np.concatenate(total_next_obs)
            total_int_rewards = brain.calculate_int_rewards(total_next_obs)
            _, next_int_values, next_ext_values, *_ = brain.get_actions_and_values(next_states, batch=True)
            # next_ext_values *= (1 - total_dones[:, -1])

            total_int_rewards = brain.normalize_int_rewards(total_int_rewards)

            total_states = np.concatenate(total_states)
            total_actions = np.concatenate(total_actions)
            total_log_probs = np.concatenate(total_log_probs)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            training_logs = brain.train(states=total_states,
                                        actions=total_actions,
                                        int_rewards=total_int_rewards,
                                        ext_rewards=total_ext_rewards,
                                        dones=total_dones,
                                        int_values=total_int_values,
                                        ext_values=total_ext_values,
                                        log_probs=total_log_probs,
                                        next_int_values=next_int_values,
                                        next_ext_values=next_ext_values,
                                        total_next_obs=total_next_obs)

            logger.log_iteration(iteration,
                                 training_logs,
                                 total_int_rewards[0].mean(),
                                 total_action_probs[0].max(-1).mean())

    else:
        play = Play(config["env_name"], brain)
        play.evaluate()
