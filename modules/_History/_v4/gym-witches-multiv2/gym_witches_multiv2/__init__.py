from gym.envs.registration import register

register(
    id='Witches_multi-v2',
    entry_point='gym_witches_multiv2.envs:WitchesEnvMulti',
)
