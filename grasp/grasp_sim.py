import os
import numpy as np
import tempfile
import omni.kit

from omni.isaac.synthetic_utils import SyntheticDataHelper

from grasp.utils.isaac_utils import RigidBody
from grasp.grasping_scenarios.grasp_object import GraspObject
from grasp.utils.visualize import screenshot, img2vid


default_camera_pose = {
    'position': (142, -127, 56), # position given by (x, y, z)
    'target': (-180, 234, -27)   # target given by (x, y , z)
    }


class GraspSimulator(GraspObject):
    """ Defines a grasping simulation scenario

    Scenarios define planar grasp execution in a scene of a Panda arm and various rigid objects
    """
    def __init__(self, kit, dc, mp, dt=1/60.0, record=False, record_interval=10):
        """
        Initializes grasp simulator
        
        Args:
            kit (omni.isaac.synthetic_utils.scripts.omnikit.OmniKitHelper): helper class for launching OmniKit from a python environment
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
            mp (omni.isaac.dynamic_control._dynamic_control.DynamicControl): dynamic control interface
            dt (float): simulation time step in seconds
            record (bool): flag for capturing screenshots throughout simulation for video recording
            record_interval (int): frame intervals for capturing screenshots
        """
        super().__init__(kit, dc, mp)
        self.frame = 0
        self.dt = dt
        self.record = record
        self.record_interval = record_interval
        self.tmp_dir = tempfile.mkdtemp()
        self.sd_helper = SyntheticDataHelper()
        
        # create initial scene
        self.create_franka()

        # set camera pose
        self.set_camera_pose(default_camera_pose['position'], default_camera_pose['target'])

    def execute_grasp(self, position, angle):
        """
        Executes a planar grasp with a panda arm.

        Args:
            position (list or numpy.darray): grasp position array of length 3 given by [x, y, z]
            angle (float): grap angle in degrees
        
        Returns:
            evaluation (enum.EnumMeta): GRASP_eval class containing two states {GRASP_eval.FAILURE, GRAPS_eval.SUCCESS}
        """
        self.set_target_angle(angle)
        self.set_target_position(position)
        self.perform_tasks()
        # start simulation
        if self._kit.editor.is_playing(): previously_playing = True
        else:                             previously_playing = False

        if self.pick_and_place is not None:

            while True:
                self.step(0)
                self.update()
                if self.pick_and_place.evaluation is not None:
                    break
        evaluation = self.pick_and_place.evaluation
        self.stop_tasks()
        self.step(0)
        self.update()

        # Stop physics simulation
        if not previously_playing: self.stop()

        return evaluation

    def wait_for_drop(self, max_steps=2000):
        """
        Waits for all objects to drop.

        Args:
            max_steps (int): maximum number of timesteps before aborting wait 
        """
        # start simulation
        if self._kit.editor.is_playing(): previously_playing = True
        else:                             previously_playing = False

        if not previously_playing: self.play()
        step = 0
        while step < max_steps or self._kit.is_loading():
            self.step(step)
            self.update()
            objects_speed = np.array([o.get_speed() for o in self.objects])
            if np.all(objects_speed == 0): break
            step +=1

        # Stop physics simulation
        if not previously_playing: self.stop()

    def wait_for_loading(self):
        """
        Waits for all scene visuals to load.
        """
        while self.is_loading():
            self.update()

    def play(self):
        """
        Starts simulation.
        """
        self._kit.play()
        if not hasattr(self, 'world') or not hasattr(self, 'franka_solid') or not hasattr(self, 'bin_solid') or not hasattr(self, 'pick_and_place'):
            self.register_scene()
    
    def stop(self):
        """
        Stops simulation.
        """
        self._kit.stop()

    def update(self):
        """
        Simulate one time step.
        """
        if self.record and self.sd_helper is not None and self.frame % self.record_interval == 0:     

            screenshot(self.sd_helper, suffix=self.frame, directory=self.tmp_dir)
        
        self._kit.update(self.dt)
        self.frame += 1

    def is_loading(self):
        """
        Determine if all scene visuals are loaded.

        Returns:
            (bool): flag for whether or not all scene visuals are loaded
        """
        return self._kit.is_loading()

    def set_camera_pose(self, position, target):
        """
        Set camera pose.

        Args:
            position (list or numpy.darray): camera position array of length 3 given by [x, y, z]
            target (list or numpy.darray): target position array of length 3 given by [x, y, z]
        """
        self._editor.set_camera_position("/OmniverseKit_Persp", *position, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", *target, True)

    def save_video(self, path):
        """
        Save video recording of screenshots taken throughout the simulation.

        Args:
            path (str): output video filename
        """
        framerate = int(round(1.0 / (self.record_interval * self.dt)))
        img2vid(os.path.join(self.tmp_dir, '*.png'), path, framerate=framerate)   
