import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

jsp = np.array([
    [[1, 2, 0],  # job 0
     [0, 2, 1]],  # job 1
    [[17, 12, 19],  # task durations of job 0
     [8, 6, 2]]  # task durations of job 1
])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,  # see documentation of DisjunctiveGraphJspEnv::get_state for more information
    flat_observation_space=True,  # see documentation of DisjunctiveGraphJspEnv::get_state for more information
    action_mode='task',  # alternative 'job'
    dtype='float32'  # dtype of the observation space
)

terminated = False
info = {}
for i in range(6):
    # get valid action mask. sample expects it to be a numpy array of type int8
    mask = np.array(env.valid_action_mask()).astype(np.int8)
    action = env.action_space.sample(mask=mask)
    state, reward, terminated, truncated, info = env.step(action)
    # chose the visualisation you want to see using the show parameter
    # console rendering
    env.render(show=["gantt_console", "graph_console"])

print(f"makespan: {info['makespan']}")