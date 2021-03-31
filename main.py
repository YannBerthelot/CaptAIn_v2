import os
import gym
import pandas as pd
from gym_environment import PlaneEnv
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from utils import timing
from configparser import ConfigParser


parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

DELTA_T = eval(parser.get("flight_model", "Timestep_size"))
TASK = parser.get("task", "TASK")
wrappable_env = PlaneEnv(task=TASK)
os.makedirs("videos", exist_ok=True)
# from stable_baselines3.common.env_checker import check_env

# check_env(env)
vec_env = make_vec_env(lambda: wrappable_env, n_envs=8)
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,
    learning_rate=1e-5,
    clip_range=0.1,
    ent_coef=0.00,
    batch_size=64,
)
n_episodes = 2000


@timing
def learn(number_timesteps):
    return model.learn(number_timesteps)


model = learn(n_episodes * 200 / DELTA_T)
model.save(f"ppo_plane_{TASK}")

env = Monitor(
    wrappable_env, "videos", force=True, video_callable=lambda episode_id: True
)
obs = env.reset()
thrust_log = []
theta_log = []
model = PPO.load(f"ppo_plane_{TASK}")
for i in range(10):

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        theta_log.append(action[0])
        thrust_log.append(action[1])
        obs, reward, done, info = env.step(action)
        if done:
            break
env.close()
fig, ax = plt.subplots()
pd.Series(theta_log).plot(ax=ax, label="Pitch")
pd.Series(thrust_log).plot(ax=ax, label="Thrust", title="Pitch and Thrust vs time")
plt.legend()
plt.show()