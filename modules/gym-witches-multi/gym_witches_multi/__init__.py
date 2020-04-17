from gym.envs.registration import register

register(
    id='Witches_multi-v1',
    entry_point='gym_witches_multi.envs:WitchesEnvMulti',
)

register(
    id='Witches_test-v1',
    entry_point='gym_witches_multi.envs:WitchesEnvTest',
)
