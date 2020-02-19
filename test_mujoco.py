import mujoco_py
import time
import numpy as np

model = mujoco_py.load_model_from_path("OurEnvs/robots/humanoid_CMU.xml")
sim = mujoco_py.MjSim(model)
dim_ctrl = model.actuator_ctrlrange.shape[0]

state = sim.get_state()
#state.qpos[53] = -1.5
#state.qpos[54] = -1.5
#state.qpos[55] = 1.5
#state.qpos[56] = 1.5
print(model.jnt_range[55])
state.qpos[53] = -1.5
state.qpos[56] = 1.5
state.qpos[57] = -1.5
sim.set_state(state)
sim.forward()

view = mujoco_py.MjViewer(sim)
view.render()

while True:
    view.render()
breakpoint()

for i in range(10000):
    sim.data.ctrl[:] = np.random.random(dim_ctrl)*2-1
    sim.step()
    view.render()
    time.sleep(0.01)

