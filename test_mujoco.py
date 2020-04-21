import mujoco_py
import time
import numpy as np

#model = mujoco_py.load_model_from_path("OurEnvs/robots/humanoid_CMU.xml")
model = mujoco_py.load_model_from_path("OurEnvs/robots/humanoid_CMU_with_ball.xml")
sim = mujoco_py.MjSim(model)
dim_ctrl = model.actuator_ctrlrange.shape[0]

state = sim.get_state()
print(model.jnt_range[55])

### Right arm position
#state.qpos[54] = -1.3
#state.qpos[56] = 1.5
#state.qpos[59] = -0.78
#state.qpos[60] = 0.7
#state.qpos[62] = 1.5

### Ball position
#state.qpos[63] = -0.48
#state.qpos[64] = 0.1
#state.qpos[65] = 1.78
#sim.data.set_joint_qpos("rhumerusry", -1.3)
#sim.data.set_joint_qpos("rradiusrx", 1.5)
#sim.data.set_joint_qpos("rhandrx", -0.78)
#sim.data.set_joint_qpos("rfingersrx", 0.7)
#sim.data.set_joint_qpos("rthumbrx", 1.5)
##sim.data.set_joint_qpos("ball", [-0.48, 0.1, 1.78, 1, 0 ,0, 0])
sim.data.set_joint_qpos("ball", [-0.4630, 0.09147, 1.67178+0.07, 1, 0 ,0, 0])

#sim.set_state(state)
#sim.forward()

view = mujoco_py.MjViewer(sim)
view.render()

print(sim.data.qpos.shape)
print(sim.data.qvel.shape)
print(sim.data.cinert.shape)
print(sim.data.cvel.shape)
print(sim.data.qfrc_actuator.shape)
print(sim.data.cfrc_ext.shape)
breakpoint()
#while True:
#    view.render()

for i in range(10000):
    sim.data.ctrl[:] = np.random.random(dim_ctrl)*2-1
    sim.step()
    view.render()
    time.sleep(0.01)
    #print(sim.data.qpos[63:])
    #print(sim.data.cinert[-1])
    print(sim.data.qpos[:3])
    #print(sim.data.cvel[-1])
    #print(sim.data.qfrc_actuator[-6:])
    #print(sim.data.cfrc_ext[-1])

