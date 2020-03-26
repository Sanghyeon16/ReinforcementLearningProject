import numpy as np
from .base_humanoid import HumanoidEnv


class HumanoidStandEnv(HumanoidEnv):
    def __init__(self,
                 xml_file='robots/humanoid_CMU_with_ball.xml',
                 ctrl_cost_weight=0.01,
                 contact_cost_weight=0.0,
                 contact_cost_range=(-np.inf, 10.0),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False,
                 healthy_reward=0.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.5, 2.0)):

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        HumanoidEnv.__init__(self,
                 xml_file,
                 ctrl_cost_weight,
                 contact_cost_weight,
                 contact_cost_range,
                 reset_noise_scale,
                 exclude_current_positions_from_observation)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward


    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done


    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        rewards = self.sim.data.get_geom_xpos("head")[2]
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        observation = self._get_obs()
        done = self.done

        return observation, reward, done, {}

