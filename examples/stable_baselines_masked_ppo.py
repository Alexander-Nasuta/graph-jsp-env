import gymnasium as gym
import sb3_contrib
import numpy as np
from stable_baselines3.common.monitor import Monitor

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

jsp = np.array([
    [[1, 2, 0],  # job 0
     [0, 2, 1]],  # job 1
    [[17, 12, 19],  # task durations of job 0
     [8, 6, 2]]  # task durations of job 1
])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=True,
    action_mode='task',  # alternative 'job'
)
env = Monitor(env)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()


env = ActionMasker(env, mask_fn)

model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train the agent
log.info("training the model")
model.learn(total_timesteps=10_000)