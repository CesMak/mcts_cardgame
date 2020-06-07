from gym.envs.registration import register

register(
    id='Witches-v0',
    entry_point='gym_witches.envs:WitchesEnv',
)
