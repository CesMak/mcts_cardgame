import numpy as np
import gym
import gym_witches_multiv2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

env_name      = "Witches_multi-v2"
nproc         = 60

def make_env(env_id):
    def _f():
        env =gym.make(env_id)
        return env
    return _f

envs = [make_env(env_name) for s in range(nproc)]
envs = SubprocVecEnv(envs)

xt = envs.reset()
print(len(xt))

print("\nIn a loop:")
for i in range(10):
    ut = np.stack([envs.action_space.sample() for _ in range(nproc)])
    state, rewards, done, info = envs.step(ut)
