
import mujoco
import mujoco.viewer
import os

#os.chdir("sr_description/mujoco_models")

model = mujoco.MjModel.from_xml_path("sr_ur_hand_e_environment.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
