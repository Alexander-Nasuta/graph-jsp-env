def test_env_constructor():
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
    _ = DisjunctiveGraphJspEnv()


def test_with_stable_baselines3_env_checker(custom_jsp_instance):
    from stable_baselines3.common.env_checker import check_env
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    elite_reward_function_args = {
        "alpha": 13.37,
        "opfa": 42.0,
    }

    def elite_reward_function(s, d, info, G, m, m_, alpha, opfa) -> float:
        info["leet"] = 1337
        return alpha if d else opfa - m + m_

    for left_shift in [True, False]:
        for normalize in [True, False]:
            for flat in [True, False]:
                for mode in ["job", "task"]:
                    for env_transform in [None, 'mask']:
                        for dt in ["float16", "float32", "float64"]:
                            for rew_func, rew_func_args in zip(
                                    [
                                        "nasuta",
                                        "zhang",
                                        "samsonov",
                                        "graph-tassel",
                                        'zero',
                                        'custom'
                                    ],
                                    [
                                        {'scaling_divisor': 1.0},
                                        {},
                                        {'gamma': 1.025, 't_opt': 40},
                                        {},
                                        {},
                                        elite_reward_function_args
                                    ]
                            ):
                                env = DisjunctiveGraphJspEnv(
                                    jps_instance=custom_jsp_instance,
                                    perform_left_shift_if_possible=left_shift,
                                    reward_function=rew_func,
                                    reward_function_parameters=rew_func_args,
                                    custom_reward_function=elite_reward_function,
                                    normalize_observation_space=normalize,
                                    flat_observation_space=flat,
                                    action_mode=mode,
                                    dtype=dt,
                                    env_transform=env_transform,
                                    verbose=1
                                )

                                check_env(env)


def test_trivial_schedule(custom_jsp_instance):
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    env = DisjunctiveGraphJspEnv(jps_instance=custom_jsp_instance)

    iteration_count = 0
    for i in range(env.total_tasks_without_dummies):
        _ = env.step(i)
        iteration_count += 1

    assert iteration_count == 8


def test_reconstruction_after_reset(custom_jsp_instance):
    import numpy as np
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    env = DisjunctiveGraphJspEnv(jps_instance=custom_jsp_instance)

    for i in range(int(env.total_tasks_without_dummies / 2)):
        _ = env.step(i)

    state = env.get_state()
    action_history = env.get_action_history()

    env.reset()

    for a in action_history:
        _ = env.step(a)

    reconstructed_state = env.get_state()

    assert np.array_equal(state, reconstructed_state)


def test_valid_actions(custom_jsp_instance):
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
    env = DisjunctiveGraphJspEnv(jps_instance=custom_jsp_instance)

    valid_actions = env.valid_actions()
    assert valid_actions == {0, 4}
    for i in range(env.total_tasks_without_dummies):
        _ = env.step(i)

    valid_actions = env.valid_actions()
    assert valid_actions == set()



def test_random_rollout(custom_jsp_instance):
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    env = DisjunctiveGraphJspEnv(jps_instance=custom_jsp_instance)

    episode_return, info = env.random_rollout()
    env.render(mode='console')

    assert env.is_terminal()


def test_greedy_machine_utilization_rollout(custom_jsp_instance):
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    env = DisjunctiveGraphJspEnv(
        jps_instance=custom_jsp_instance,
        reward_function="nasuta",
    )

    episode_return, info = env.greedy_machine_utilization_rollout()
    env.render(mode='console')

    assert env.is_terminal()
    assert env.reward_function == "nasuta"

