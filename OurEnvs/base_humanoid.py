import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from . import my_utils


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 3.5,
    'lookat': np.array((0.0, 0.0, 1.2)),
    'elevation': -20.0,
}


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='robots/humanoid_CMU.xml',
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.has_ball = None

        mujoco_env.MujocoEnv.__init__(self, my_utils.abs_path(xml_file), 5)


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost


    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self.has_ball is None:
            self.has_ball ="ball" in self.model.body_names
            self.sim.data.set_joint_qpos("rhumerusry", -1.3)
            self.sim.data.set_joint_qpos("rradiusrx", 1.5)
            self.sim.data.set_joint_qpos("rhandrx", -0.78)
            self.sim.data.set_joint_qpos("rfingersrx", 0.7)
            self.sim.data.set_joint_qpos("rthumbrx", 1.5)
            self.init_qpos = self.sim.data.qpos.ravel().copy()
            self.init_qvel = self.sim.data.qvel.ravel().copy()

        if self.has_ball:
            com_inertia = com_inertia[:-1]
            com_velocity = com_velocity[:-1]
            actuator_forces = actuator_forces[:-6]
            external_contact_forces = external_contact_forces[:-1]
        else:
            position = np.concatenate((position, [0]*7))
            velocity = np.concatenate((velocity, [0]*6))

        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        )).astype(np.float32)

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        if self.has_ball:
            rhand_pos = self.sim.data.get_geom_xpos("rhand").copy()
            rhand_pos[2] += 0.08
            self.sim.data.set_joint_qpos("ball", np.concatenate((rhand_pos, [1, 0, 0, 0])))

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
