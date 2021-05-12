# Credits: The majority of this code is taken from build code associated with nvidia/isaac-sim:2020.2.2_ea with minor modifications.

import gc
import carb
import omni.usd

from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
from grasp.utils.isaac_utils import set_up_z_axis


class Scenario:
    """ 
    Defines a block stacking scenario.

    Scenarios define the life cycle within kit and handle init, startup, shutdown etc.
    """

    def __init__(self, editor, dc, mp):
        """
        Initialize scenario.

        Args:
            editor (omni.kit.editor._editor.IEditor): editor object from isaac-sim simulation
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
            mp (omni.isaac.dynamic_control._dynamic_control.DynamicControl): dynamic control interface
        """
        self._editor = editor  # Reference to the Kit editor
        self._stage = omni.usd.get_context().get_stage()  # Reference to the current USD stage
        self._dc = dc  # Reference to the dynamic control plugin
        self._mp = mp  # Reference to the motion planning plugin
        self._domains = []  # Contains instances of environment
        self._obstacles = []  # Containts references to any obstacles in the scenario
        self._executor = None  # Contains the thread pool used to run tasks
        self._created = False  # Is the robot created or not
        self._running = False  # Is the task running or not

    def __del__(self):
        """
        Cleanup scenario objects when deleted, force garbage collection.
        """
        self.robot_created = False
        self._domains = []
        self._obstacles = []
        self._executor = None
        gc.collect()

    def reset_blocks(self, *args):
        """
        Funtion called when block poses are reset.
        """
        pass

    def stop_tasks(self, *args):
        """
        Stop tasks in the scenario if any.
        """
        self._running = False
        pass

    def step(self, step):
        """
        Step the scenario, can be used to update things in the scenario per frame.
        """
        pass

    def create_franka(self, *args):
        """
        Create franka USD objects.
        """
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.asset_path = nucleus_server + "/Isaac"

        # USD paths loaded by scenarios
        
        self.franka_table_usd = self.asset_path + "/Samples/Leonardo/Stage/franka_block_stacking.usd"
        self.franka_ghost_usd = self.asset_path + "/Samples/Leonardo/Robots/franka_ghost.usd"
        self.background_usd = self.asset_path + "/Environments/Grid/gridroom_curved.usd"
        self.rubiks_cube_usd = self.asset_path + "/Props/Rubiks_Cube/rubiks_cube.usd"
        self.red_cube_usd = self.asset_path + "/Props/Blocks/red_block.usd"
        self.yellow_cube_usd = self.asset_path + "/Props/Blocks/yellow_block.usd"
        self.green_cube_usd = self.asset_path + "/Props/Blocks/green_block.usd"
        self.blue_cube_usd = self.asset_path + "/Props/Blocks/blue_block.usd"

        self._created = True
        self._stage = omni.usd.get_context().get_stage()
        set_up_z_axis(self._stage)
        self.stop_tasks()
        pass

    def register_assets(self, *args):
        """
        Connect franka controller to usd assets
        """
        pass

    def task(self, domain):
        """
        Task to be performed for a given robot.
        """
        pass

    def perform_tasks(self, *args):
        """
        Perform all tasks in scenario if multiple robots are present.
        """
        self._running = True
        pass

    def is_created(self):
        """
        Return if the franka was already created.
        """
        return self._created
