import os, logging
from typing import Callable
from configparser import ConfigParser
from gym.wrappers import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_environment import PlaneEnv
from utils import setup_logger

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

DELTA_T = float(parser.get("flight_model", "Timestep_size"))
TASK = parser.get("task", "TASK")
wrappable_env = PlaneEnv(task=TASK)
os.makedirs("videos", exist_ok=True)
os.makedirs("tensorboard_logs", exist_ok=True)

# create and configure logger
os.makedirs("logs", exist_ok=True)
DEBUG = bool(parser.get("debug", "debug"))
print("debug", DEBUG)
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOG_FORMAT = "%(levelno)s %(asctime)s %(funcName)s - %(message)s"
logger = setup_logger(
    "main_logger", "logs/main.log", level=level, format=LOG_FORMAT, mode="a"
)

# check_env(env)
vec_env = make_vec_env(lambda: wrappable_env, n_envs=8)
LOAD = True
params = {
    "learning_rate": 1e-4,
    "ent_coef": 0,
    "batch_size": 64,
    "gamma": 0.99,
    "n_steps": 2048,
}


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


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
n_episodes = 400
logger.info(params)

N_BATCH = 10
for n in range(N_BATCH):
    print(f"batch {n}/{N_BATCH}")
    model.learn(n_episodes * 200 / DELTA_T, reset_num_timesteps=(n == 0) & (not (LOAD)))
    model.save(f"ppo_plane_{TASK}")
    env = Monitor(
        wrappable_env,
        f"videos/batch_{n}",
        video_callable=lambda episode_id: True,
        force=True,
    )
    obs = env.reset()
    thrust_log = []
    theta_log = []
    model = PPO.load(f"ppo_plane_{TASK}", env=env)
    reward_collector = 0
    for i in range(1):

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=False)
            theta_log.append(action[0])
            thrust_log.append(action[1])
            obs, reward, done, info = env.step(action)
            reward_collector += 0
            if done:
                break
    logger.info("batch %s reward %s", n, reward_collector)
    env.close()
    model = PPO.load(f"ppo_plane_{TASK}", env=vec_env)
