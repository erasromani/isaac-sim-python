# Credits: The majority of this code is taken from build code associated with nvidia/isaac-sim:2020.2.2_ea with minor modifications.

import time
import os
import numpy as np
import carb.tokens
import omni.kit.settings

from pxr import Usd, UsdGeom, Gf
from collections import deque

from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.motion_planning import _motion_planning
from omni.isaac.samples.scripts.utils import math_utils


# default joint configuration
default_config = (0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75)


# Alternative default config for motion planning
alternate_config = [
    (1.5356, -1.3813, -1.5151, -2.0015, -1.3937, 1.5887, 1.4597),
    (-1.5356, -1.3813, 1.5151, -2.0015, 1.3937, 1.5887, 0.4314),
]


class Gripper:
    """
    Gripper for franka.
    """
    def __init__(self, dc, ar):
        """
        Initialize gripper.

        Args:
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
            ar (int): articulation identifier
        """
        self.dc = dc
        self.ar = ar
        self.finger_j1 = self.dc.find_articulation_dof(self.ar, "panda_finger_joint1")
        self.finger_j2 = self.dc.find_articulation_dof(self.ar, "panda_finger_joint2")
        self.width = 0
        self.width_history = deque(maxlen=50)

    def open(self, wait=False):
        """
        Open gripper.
        """
        if self.width < 0.045:
            self.move(0.045, wait=True)
        self.move(0.09, wait=wait)

    def close(self, wait=False, force=0):
        """
        Close gripper.
        """
        self.move(0, wait=wait)

    def move(self, width=0.03, speed=0.2, wait=False):
        """
        Modify width.
        """
        self.width = width
        # if wait:
        #     time.sleep(0.5)

    def update(self):
        """
        Actuate gripper.
        """
        self.dc.set_dof_position_target(self.finger_j1, self.width * 0.5 * 100)
        self.dc.set_dof_position_target(self.finger_j2, self.width * 0.5 * 100)
        self.width_history.append(self.get_width())

    def get_width(self):
        """
        Get current width.
        """
        return sum(self.get_position())

    def get_position(self):
        """
        Get left and right finger local position.
        """
        return self.dc.get_dof_position(self.finger_j1), self.dc.get_dof_position(self.finger_j2)
    
    def get_velocity(self, from_articulation=True):
        """
        Get left and right finger local velocity.
        """
        if from_articulation:
            
            return (self.dc.get_dof_velocity(self.finger_j1), self.dc.get_dof_velocity(self.finger_j2))

        else:
        
            leftfinger_handle = self.dc.get_rigid_body(self.dc.get_articulation_path(self.ar) + '/panda_leftfinger')
            rightfinger_handle = self.dc.get_rigid_body(self.dc.get_articulation_path(self.ar) + '/panda_rightfinger')
            leftfinger_velocity = np.linalg.norm(np.array(self.dc.get_rigid_body_local_linear_velocity(leftfinger_handle)))
            rightfinger_velocity = np.linalg.norm(np.array(self.dc.get_rigid_body_local_linear_velocity(rightfinger_handle)))
            return (leftfinger_velocity, rightfinger_velocity)

    def is_moving(self, tol=1e-2):
        """
        Determine if gripper fingers are moving
        """
        if len(self.width_history) < self.width_history.maxlen or np.array(self.width_history).std() > tol:
            return True
        else:
            return False

    def get_state(self):
        """
        Get gripper state.
        """
        dof_states = self.dc.get_articulation_dof_states(self.ar, _dynamic_control.STATE_ALL)
        return dof_states[-2], dof_states[-1]

    def is_closed(self, tol=1e-2):
        """
        Determine if gripper is closed.
        """
        if self.get_width() < tol:
            return True
        else:
            return False


class Status:
    """
    Class that contains status for end effector
    """
    def __init__(self, mp, rmp_handle):
        """
        Initialize status object.

        Args:
            mp (omni.isaac.dynamic_control._dynamic_control.DynamicControl): dynamic control interface
            rmp_handle (int): RMP handle identifier
        """
        self.mp = mp
        self.rmp_handle = rmp_handle
        self.orig = np.array([0, 0, 0])
        self.axis_x = np.array([1, 0, 0])
        self.axis_y = np.array([0, 1, 0])
        self.axis_z = np.array([0, 0, 1])

        self.current_frame = {"orig": self.orig, "axis_x": self.axis_x, "axis_y": self.axis_y, "axis_z": self.axis_z}
        self.target_frame = {"orig": self.orig, "axis_x": self.axis_x, "axis_y": self.axis_y, "axis_z": self.axis_z}
        self.frame = self.current_frame

    def update(self):
        """
        Update end effector state.
        """
        state = self.mp.getRMPState(self.rmp_handle)
        target = self.mp.getRMPTarget(self.rmp_handle)
        self.orig = np.array([state[0].x, state[0].y, state[0].z])
        self.axis_x = np.array([state[1].x, state[1].y, state[1].z])
        self.axis_y = np.array([state[2].x, state[2].y, state[2].z])
        self.axis_z = np.array([state[3].x, state[3].y, state[3].z])

        self.current_frame = {"orig": self.orig, "axis_x": self.axis_x, "axis_y": self.axis_y, "axis_z": self.axis_z}
        self.frame = self.current_frame
        self.current_target = {
            "orig": np.array([target[0].x, target[0].y, target[0].z]),
            "axis_x": np.array([target[1].x, target[1].y, target[1].z]),
            "axis_y": np.array([target[2].x, target[2].y, target[2].z]),
            "axis_z": np.array([target[3].x, target[3].y, target[3].z]),
        }


class EndEffector:
    """
    End effector object that controls movement.
    """
    def __init__(self, dc, mp, ar, rmp_handle):
        """
        Initialize end effector.

        Args:
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
            mp (omni.isaac.dynamic_control._dynamic_control.DynamicControl): dynamic control interface
            ar (int): articulation identifier
            rmp_handle (int): RMP handle identifier
        """
        self.dc = dc
        self.ar = ar
        self.mp = mp
        self.rmp_handle = rmp_handle
        self.gripper = Gripper(dc, ar)
        self.status = Status(mp, rmp_handle)
        self.UpRot = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)

    def freeze(self):
        self.go_local(
            orig=self.status.orig, axis_x=self.status.axis_x, axis_z=self.status.axis_z, wait_for_target=False
        )

    def go_local(
        self,
        target=None,
        orig=[],
        axis_x=[],
        axis_y=[],
        axis_z=[],
        required_orig_err=0.01,
        required_axis_x_err=0.01,
        required_axis_y_err=0.01,
        required_axis_z_err=0.01,
        orig_thresh=None,
        axis_x_thresh=None,
        axis_y_thresh=None,
        axis_z_thresh=None,
        approach_direction=[],
        approach_standoff=0.1,
        approach_standoff_std_dev=0.001,
        use_level_surface_orientation=False,
        use_target_weight_override=True,
        use_default_config=False,
        wait_for_target=True,
        wait_time=None,
    ):

        self.target_weight_override_value = 10000.0
        self.target_weight_override_std_dev = 0.03
        if orig_thresh:
            required_orig_err = orig_thresh
        if axis_x_thresh:
            required_axis_x_err = axis_x_thresh
        if axis_y_thresh:
            required_axis_y_err = axis_y_thresh
        if axis_z_thresh:
            required_axis_z_err = axis_z_thresh

        if target:
            orig = target["orig"]
            if "axis_x" in target and target["axis_x"] is not None:
                axis_x = target["axis_x"]
            if "axis_y" in target and target["axis_y"] is not None:
                axis_y = target["axis_y"]
            if "axis_z" in target and target["axis_z"] is not None:
                axis_z = target["axis_z"]

        orig = np.array(orig)
        axis_x = np.array(axis_x)
        axis_y = np.array(axis_y)
        axis_z = np.array(axis_z)
        approach = _motion_planning.Approach((0, 0, 1), 0, 0)

        if len(approach_direction) != 0:
            approach = _motion_planning.Approach(approach_direction, approach_standoff, approach_standoff_std_dev)

        pose_command = _motion_planning.PartialPoseCommand()
        if len(orig) > 0:
            pose_command.set(_motion_planning.Command(orig, approach), int(_motion_planning.FrameElement.ORIG))
        if len(axis_x) > 0:
            pose_command.set(_motion_planning.Command(axis_x), int(_motion_planning.FrameElement.AXIS_X))
        if len(axis_y) > 0:
            pose_command.set(_motion_planning.Command(axis_y), int(_motion_planning.FrameElement.AXIS_Y))
        if len(axis_z) > 0:
            pose_command.set(_motion_planning.Command(axis_z), int(_motion_planning.FrameElement.AXIS_Z))

        self.mp.goLocal(self.rmp_handle, pose_command)

        if wait_for_target and wait_time:
            error = 1
            future_time = time.time() + wait_time

            while error > required_orig_err and time.time() < future_time:
                # time.sleep(0.1)
                error = self.mp.getError(self.rmp_handle)

    def look_at(self, gripper_pos, target):
        # Y up works for look at but sometimes flips, go_local might be a safer bet with a  locked y_axis
        orientation = math_utils.lookAt(gripper_pos, target, (0, 1, 0))
        mat = Gf.Matrix3d(orientation).GetTranspose()

        self.go_local(
            orig=[gripper_pos[0], gripper_pos[1], gripper_pos[2]],
            axis_x=[mat.GetColumn(0)[0], mat.GetColumn(0)[1], mat.GetColumn(0)[2]],
            axis_z=[mat.GetColumn(2)[0], mat.GetColumn(2)[1], mat.GetColumn(2)[2]],
        )


class Franka:
    """
    Franka objects that contains implementation details for robot control.
    """
    def __init__(self, stage, prim, dc, mp, world=None, group_path="", default_config=None, is_ghost=False):
        """
        Initialize Franka controller.

        Args:
            stage (pxr.Usd.Stage): usd stage
            prim (pxr.Usd.Prim): robot prim
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
            mp (omni.isaac.dynamic_control._dynamic_control.DynamicControl): dynamic control interface
            world (omni.isaac.samples.scripts.utils.world.World): simulation world handler
            default_config (tuple or list): default configuration for robot revolute joint drivers
            is_ghost (bool): flag for turning off collision and modifying visuals for robot arm
        """
        self.dc = dc
        self.mp = mp
        self.prim = prim
        self.stage = stage
        # get handle to the articulation for this franka
        self.ar = self.dc.get_articulation(prim.GetPath().pathString)
        self.is_ghost = is_ghost

        self.base = self.dc.get_articulation_root_body(self.ar)

        body_count = self.dc.get_articulation_body_count(self.ar)
        for bodyIdx in range(body_count):
            body = self.dc.get_articulation_body(self.ar, bodyIdx)
            self.dc.set_rigid_body_disable_gravity(body, True)

        exec_folder = os.path.abspath(
            carb.tokens.get_tokens_interface().resolve(
                f"{os.environ['ISAAC_PATH']}/exts/omni.isaac.motion_planning/resources/lula/lula_franka"
            )
        )

        self.rmp_handle = self.mp.registerRmp(
            exec_folder + "/urdf/lula_franka_gen.urdf",
            exec_folder + "/config/robot_descriptor.yaml",
            exec_folder + "/config/franka_rmpflow_common.yaml",
            prim.GetPath().pathString,
            "right_gripper",
            True,
        )
        print("franka rmp handle", self.rmp_handle)
        if world is not None:
            self.world = world
            self.world.rmp_handle = self.rmp_handle
            self.world.register_parent(self.base, self.prim, "panda_link0")

        settings = omni.kit.settings.get_settings_interface()
        self.mp.setFrequency(self.rmp_handle, settings.get("/physics/timeStepsPerSecond"), True)

        self.end_effector = EndEffector(self.dc, self.mp, self.ar, self.rmp_handle)
        if default_config:
            self.mp.setDefaultConfig(self.rmp_handle, default_config)
        self.target_visibility = True
        if self.is_ghost:
            self.target_visibility = False

        self.imageable = UsdGeom.Imageable(self.prim)

    def __del__(self):
        """
        Unregister RMP.
        """
        self.mp.unregisterRmp(self.rmp_handle)
        print("  Delete Franka")

    def set_pose(self, pos, rot):
        """
        Set robot pose.
        """
        self._mp.setTargetLocal(self.rmp_handle, pos, rot)

    def set_speed(self, speed_level):
        """
        Set robot speed.
        """
        pass

    def update(self):
        """
        Update robot state.
        """
        self.end_effector.gripper.update()
        self.end_effector.status.update()
        if self.imageable:
            if self.target_visibility is not self.imageable.ComputeVisibility(Usd.TimeCode.Default()):
                if self.target_visibility:
                    self.imageable.MakeVisible()
                else:
                    self.imageable.MakeInvisible()

    def send_config(self, config):
        """
        Set robot default configuration.
        """
        if self.is_ghost is False:
            self.mp.setDefaultConfig(self.rmp_handle, config)
