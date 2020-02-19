from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='OurHumanoid-v0',
    entry_point='OurEnvs.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)
