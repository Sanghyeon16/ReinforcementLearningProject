from gym.envs.registration import registry, register, make, spec


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

register(
    id='OurHumanoidStandToHold-v0',
    entry_point='OurEnvs.humanoid_stand_to_hold:HumanoidStandToHoldEnv',
    max_episode_steps=1000,
)

register(
    id='OurHumanoidHoldToThrow-v0',
    entry_point='OurEnvs.humanoid_hold_to_throw:HumanoidHoldToThrowEnv',
    max_episode_steps=1000,
)


