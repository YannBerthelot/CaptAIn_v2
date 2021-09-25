import os
import pandas as pd
import time
from configparser import ConfigParser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

import torch as th
from gym_environment import PlaneEnv

th.set_num_threads(1)
parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

TASK = parser.get("task", "TASK")
DELTA_T = float(parser.get("flight_model", "Timestep_size"))
MAX_TIMESTEP = 200 / DELTA_T
N_EPISODES = float(parser.get("task", "n_episodes"))
N_ENVS = int(parser.get("env", "n_envs"))


def test_speed(speeds={"env": "fast", "aerodynamics": "fast"}, n_envs=1):

    wrappable_env = PlaneEnv(task=TASK, speeds=speeds, n_envs=n_envs)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    # for n_envs in [2 ** n for n in range(1, 10)]:
    vec_env = make_vec_env(lambda: wrappable_env, n_envs=n_envs)
    vec_env_eval = make_vec_env(lambda: wrappable_env, n_envs=1)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
    eval_callback = EvalCallback(
        vec_env_eval, callback_on_new_best=callback_on_best, verbose=1
    )
    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[1], vf=[1])])
    model = PPO(
        "MlpPolicy", vec_env, verbose=0, batch_size=1024
    )  # policy_kwargs=policy_kwargs)

    n_timesteps = MAX_TIMESTEP * N_EPISODES
    start_learn = time.process_time()
    start_learn_2 = time.time()

    model.learn(n_timesteps)  # , callback=eval_callback)

    learn_time = time.process_time() - start_learn
    learn_time_2 = time.time() - start_learn_2
    print(learn_time, learn_time_2)
    return learn_time
    # print(n_envs, learn_time / (N_EPISODES), learn_time_2 / (N_EPISODES))


if __name__ == "__main__":
    # fast_slow = ["fast", "slow"]
    # for env in fast_slow:
    #     for aero in fast_slow:
    #         if (env == "fast") and (aero == "slow"):
    #             break
    #         print(f"{env=} {aero=} ")
    #         test_speed(speeds={"env": env, "aerodynamics": aero})
    l = []
    for n_cpu in range(1, os.cpu_count() + 1):
        print(f"{n_cpu=}")
        l.append(test_speed(n_envs=n_cpu))
    pd.Series(l).to_csv("time_vs_cpu.csv", index=False)
    # model.save(f"ppo_plane_{TASK}")
    # env = Monitor(
    #     wrappable_env,
    #     f"videos/batch",
    #     video_callable=lambda episode_id: True,
    #     force=True,
    # )
    # model = PPO.load(f"ppo_plane_{TASK}", env=env)
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=False)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         break
    # env.close()
    # model = PPO.load(f"ppo_plane_{TASK}", env=vec_env)