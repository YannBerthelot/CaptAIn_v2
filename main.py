import os
import gym
import pandas as pd
from gym_environment import PlaneEnv
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

wrappable_env = PlaneEnv(task="take-off")
os.makedirs("videos", exist_ok=True)
# from stable_baselines3.common.env_checker import check_env

# check_env(env)
vec_env = make_vec_env(lambda: wrappable_env, n_envs=16)
model = PPO(
    "MlpPolicy", vec_env, verbose=0, learning_rate=1e-4, clip_range=0.1, ent_coef=0.001
)
n_episodes = 10000
model.learn(n_episodes * 200)

env = Monitor(wrappable_env, "videos", force=True)
obs = env.reset()
thrust_log = []
theta_log = []
while True:
    action, _states = model.predict(obs)
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