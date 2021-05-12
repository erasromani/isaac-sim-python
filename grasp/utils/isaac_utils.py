# Credits: All code except class RigidBody and Camera is taken from build code associated with nvidia/isaac-sim:2020.2.2_ea.

import numpy as np
import omni.kit

from pxr import Usd, UsdGeom, Gf, PhysicsSchema, PhysxSchema


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location):
    """
    Create prim from usd.
    """
    envPrim = stage.DefinePrim(prim_env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(prim_usd_path)  # attach the USD to the given path
    set_translate(envPrim, location)  # set pose
    return stage.GetPrimAtPath(envPrim.GetPath().pathString)


def set_up_z_axis(stage):
    """
    Utility function to specify the stage with the z axis as "up".
    """
    rootLayer = stage.GetRootLayer()
    rootLayer.SetPermissionToEdit(True)
    with Usd.EditContext(stage, rootLayer):
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def set_translate(prim, new_loc):
    """
    Specify position of a given prim, reuse any existing transform ops when possible.
    """
    properties = prim.GetPropertyNames()
    if "xformOp:translate" in properties:
        translate_attr = prim.GetAttribute("xformOp:translate")
        translate_attr.Set(new_loc)
    elif "xformOp:translation" in properties:
        translation_attr = prim.GetAttribute("xformOp:translate")
        translation_attr.Set(new_loc)
    elif "xformOp:transform" in properties:
        transform_attr = prim.GetAttribute("xformOp:transform")
        matrix = prim.GetAttribute("xformOp:transform").Get()
        matrix.SetTranslateOnly(new_loc)
        transform_attr.Set(matrix)
    else:
        xform = UsdGeom.Xformable(prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(Gf.Matrix4d().SetTranslate(new_loc))


def set_rotate(prim, rot_mat):
    """
    Specify orientation of a given prim, reuse any existing transform ops when possible.
    """
    properties = prim.GetPropertyNames()
    if "xformOp:rotate" in properties:
        rotate_attr = prim.GetAttribute("xformOp:rotate")
        rotate_attr.Set(rot_mat)
    elif "xformOp:transform" in properties:
        transform_attr = prim.GetAttribute("xformOp:transform")
        matrix = prim.GetAttribute("xformOp:transform").Get()
        matrix.SetRotateOnly(rot_mat.ExtractRotation())
        transform_attr.Set(matrix)
    else:
        xform = UsdGeom.Xformable(prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(Gf.Matrix4d().SetRotate(rot_mat))


def create_background(stage, background_stage):
    """
    Create background stage.
    """
    background_path = "/background"
    if not stage.GetPrimAtPath(background_path):
        backPrim = stage.DefinePrim(background_path, "Xform")
        backPrim.GetReferences().AddReference(background_stage)
        # Move the stage down -104cm so that the floor is below the table wheels, move in y axis to get light closer
        set_translate(backPrim, Gf.Vec3d(0, -400, -104))


def setup_physics(stage):
    """
    Set default physics parameters.
    """
    # Specify gravity
    metersPerUnit = UsdGeom.GetStageMetersPerUnit(stage)
    gravityScale = 9.81 / metersPerUnit
    gravity = Gf.Vec3f(0.0, 0.0, -gravityScale)
    scene = PhysicsSchema.PhysicsScene.Define(stage, "/physics/scene")
    scene.CreateGravityAttr().Set(gravity)

    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physics/scene"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physics/scene")
    physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
    physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
    physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")


class Camera:
    """
    Camera object that contain state information for a camera in the scene.
    """
    def __init__(self, camera_path, translation, rotation):
        """
        Initializes the Camera object.

        Args:
            camera_path (str): path of camera in stage hierarchy
            translation (list or tuple): camera position
            rotation (list or tuple): camera orientation described by euler angles in degrees
        """
        self.prim = self._kit.create_prim(
            camera_path,
            "Camera",
            translation=translation,
            rotation=rotatation,
        )
        self.name = self.prim.GetPrimPath().name
        self.vpi = omni.kit.viewport.get_viewport_interface

    def set_translate(self, position):
        """
        Set camera position.

        Args:
            position (tuple): camera position specified by (X, Y, Z)
        """
        if not isinstance(position, tuple): position = tuple(position)
        translate_attr = self.prim.GetAttribute("xformOp:translate")
        translate_attr.Set(position)
    
    def set_rotate(self, rotation):
        """
        Set camera position.

        Args:
            rotation (tuple): camera orientation specified by three euler angles in degrees
        """
        if not isinstance(rotation, tuple): rotation = tuple(rotation)
        rotate_attr = self.prim.GetAttribute("xformOp:rotateZYX")
        rotate_attr.Set(rotation)

    def activate(self):
        """
        Activate camera to viewport.
        """
        self.vpi.get_viewport_window().set_active_camera(str(self.prim.GetPath()))

    def __repr__(self):
        return self.name


class Camera:
    """
    Camera object that contain state information for a camera in the scene.
    """
    def __init__(self, camera_path, translation, rotation):
        """
        Initializes the Camera object.

        Args:
            camera_path (str): path of camera in stage hierarchy
            translation (list or tuple): camera position
            rotation (list or tuple): camera orientation described by euler angles in degrees
        """
        self.prim = self._kit.create_prim(
            camera_path,
            "Camera",
            translation=translation,
            rotation=rotation,
        )
        self.name = self.prim.GetPrimPath().name
        self.vpi = omni.kit.viewport.get_viewport_interface

    def set_translate(self, position):
        """
        Set camera position.

        Args:
            position (tuple): camera position specified by (X, Y, Z)
        """
        if not isinstance(position, tuple): position = tuple(position)
        translate_attr = self.prim.GetAttribute("xformOp:translate")
        translate_attr.Set(position)
    
    def set_rotate(self, rotation):
        """
        Set camera position.

        Args:
            rotation (tuple): camera orientation specified by three euler angles in degrees
        """
        if not isinstance(rotation, tuple): rotation = tuple(rotation)
        rotate_attr = self.prim.GetAttribute("xformOp:rotateZYX")
        rotate_attr.Set(rotation)

    def activate(self):
        """
        Activate camera to viewport.
        """
        self.vpi.get_viewport_window().set_active_camera(str(self.prim.GetPath()))

    def __repr__(self):
        return self.name


class RigidBody:
    """
    RigidBody objects that contains state information of the rigid body.
    """
    def __init__(self, prim, dc):
        """
        Initializes for RigidBody object
        
        Args:
            prim (pxr.Usd.Prim): rigid body prim
            dc (omni.isaac.motion_planning._motion_planning.MotionPlanning): motion planning interface from RMP extension
        """
        self.prim = prim
        self._dc = dc
        self.name = prim.GetPrimPath().name
        self.handle = self.get_rigid_body_handle()

    def get_rigid_body_handle(self):
        """
        Get rigid body handle.
        """
        object_children = self.prim.GetChildren()
        for child in object_children:
            child_path = child.GetPath().pathString
            body_handle = self._dc.get_rigid_body(child_path)
            if body_handle != 0:
                bin_path = child_path

        object_handle = self._dc.get_rigid_body(bin_path)
        if object_handle != 0: return object_handle

    def get_linear_velocity(self):
        """
        Get linear velocity of rigid body.
        """
        return np.array(self._dc.get_rigid_body_linear_velocity(self.handle))

    def get_angular_velocity(self):
        """
        Get angular velocity of rigid body.
        """
        return np.array(self._dc.get_rigid_body_angular_velocity(self.handle))

    def get_speed(self):
        """
        Get speed of rigid body given by the l2 norm of the velocity.
        """
        velocity = self.get_linear_velocity()
        speed = np.linalg.norm(velocity)
        return speed

    def get_pose(self):
        """
        Get pose of the rigid body containing the position and orientation information.
        """
        return self._dc.get_rigid_body_pose(self.handle)

    def get_position(self):
        """
        Get the position of the rigid body object.
        """
        pose = self.get_pose()
        position = np.array(pose.p)
        return position
    
    def get_orientation(self):
        """
        Get orientation of the rigid body object.
        """
        pose = self.get_pose()
        orientation = np.array(pose.r)
        return orientation

    def get_bound(self):
        """
        Get bounds of the rigid body object in global coordinates.
        """
        bound = UsdGeom.Mesh(self.prim).ComputeWorldBound(0.0, "default").GetBox()
        return [np.array(bound.GetMin()), np.array(bound.GetMax())]

    def __repr__(self):
        return self.name
