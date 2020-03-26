import numpy as np
from .base_humanoid import HumanoidEnv


class HumanoidThrowEnv(HumanoidEnv):
    def __init__(self,
                 xml_file='robots/humanoid_CMU_with_ball.xml',
                 ctrl_cost_weight=0.01,
                 contact_cost_weight=0.0,
                 contact_cost_range=(-np.inf, 10.0),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False,
                 healthy_reward=0.0,
                 terminate_when_unhealthy=True,
                 ball_z_range=(0.1, float("inf"))):

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._ball_z_range = ball_z_range
        self.started = False

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
        min_z, max_z = self._ball_z_range
        ball_height = self.sim.data.get_body_xipos("ball")[2]
        is_healthy = min_z < ball_height < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done


    def step(self, action):
        ball_x_before = self.sim.data.get_joint_qpos('ball')[2]
        self.do_simulation(action, self.frame_skip)
        ball_x_after  = self.sim.data.get_joint_qpos('ball')[2]

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        ball_height = self.sim.data.get_body_xipos("ball")[2]
        rewards = ball_x_after - ball_x_before
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        observation = self._get_obs()
        done = self.done
        if not self.started:
            self.started = True
            done = False

        return observation, reward, done, {}

