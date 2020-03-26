from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

#register(
#    id='OurHumanoidRun-v0',
#    entry_point='OurEnvs.humanoid_run:HumanoidRunEnv',
#    max_episode_steps=1000,
#)

register(
    id='OurHumanoidStand-v0',
    entry_point='OurEnvs.humanoid_stand:HumanoidStandEnv',
    max_episode_steps=1000,
)

register(
    id='OurHumanoidHold-v0',
    entry_point='OurEnvs.humanoid_hold:HumanoidHoldEnv',
    max_episode_steps=1000,
)

register(
    id='OurHumanoidThrow-v0',
    entry_point='OurEnvs.humanoid_throw:HumanoidThrowEnv',
    max_episode_steps=1000,
)
