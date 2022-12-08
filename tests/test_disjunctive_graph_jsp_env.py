def test_env_constructor():
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
    env_instance = DisjunctiveGraphJspEnv()


def test_with_stable_baselines3_env_checker(custom_jsp_instance):
    from stable_baselines3.common.env_checker import check_env
    from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

    for left_shift in [True, False]:
        for normalize in [True, False]:
            for flat in [True, False]:
                for scale_rew in [True, False]:
                    for mode in ["job", "task"]:
                        for env_transform in [None, 'mask']:
                            for dt in ["float16", "float32", "float64"]:

                                env = DisjunctiveGraphJspEnv(
                                    jps_instance=custom_jsp_instance,
                                    perform_left_shift_if_possible=left_shift,
                                    scaling_divisor=None,
                                    scale_reward=scale_rew,
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
