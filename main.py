import os, logging

from configparser import ConfigParser
from gym.wrappers import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_environment import PlaneEnv
from utils import setup_logger
from stablebaselines_utils import linear_schedule

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)
DELTA_T = float(parser.get("flight_model", "Timestep_size"))
TASK = parser.get("task", "TASK")
wrappable_env = PlaneEnv(task=TASK)


if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    vec_env = make_vec_env(lambda: wrappable_env, n_envs=1)
    LOAD = True & ("ppo_plane_{TASK}.zip" in os.listdir())
    params = {
        "learning_rate": 1e-4,
        "ent_coef": 0,
        "batch_size": 64,
        "gamma": 0.99,
        "n_steps": 2048,
    }

    if LOAD:
        vec_env.reset()
        model = PPO.load(f"ppo_plane_{TASK}", env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=linear_schedule(params["learning_rate"]),
            clip_range=0.1,
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            n_steps=params["n_steps"],
            tensorboard_log="tensorboard_logs",
        )
    MAX_TIMESTEP = 200 / DELTA_T
    n_episodes = 1000
    n_timesteps = MAX_TIMESTEP * n_episodes

    model.learn(n_timesteps, reset_num_timesteps=not (LOAD))
    model.save(f"ppo_plane_{TASK}")
    env = Monitor(
        wrappable_env,
        f"videos/batch_{n}",
        video_callable=lambda episode_id: True,
        force=True,
    )
    model = PPO.load(f"ppo_plane_{TASK}", env=env)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        reward_collector += 0
        if done:
            break
    env.close()
    model = PPO.load(f"ppo_plane_{TASK}", env=vec_env)
