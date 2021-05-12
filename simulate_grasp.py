import os
import argparse

from grasp.grasp_sim import GraspSimulator

from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.synthetic_utils import OmniKitHelper


def main(args):

    kit = OmniKitHelper(
        {"renderer": "RayTracedLighting", "experience": f"{os.environ['EXP_PATH']}/isaac-sim-python.json", "width": args.width, "height": args.height}
    )
    _mp = _motion_planning.acquire_motion_planning_interface()
    _dc = _dynamic_control.acquire_dynamic_control_interface()

    if args.video: record = True
    else:          record = False

    sim = GraspSimulator(kit, _dc, _mp, record=record)

    # add object path
    if args.location == 'local': from_server = False
    else:                        from_server = True

    for path in args.path:
        
        sim.add_object_path(path, from_server=from_server)

    # start simulation
    sim.play()

    for _ in range(args.num):
        
        sim.add_object(position=(40, 0, 10))

    sim.wait_for_drop()
    sim.wait_for_loading()

    evaluation = sim.execute_grasp(args.position, args.angle)

    output_string = f"Grasp evaluation: {evaluation}" 
    print('\n' + ''.join(['#'] * len(output_string)))
    print(output_string)
    print(''.join(['#'] * len(output_string)) + '\n')

    # Stop physics simulation
    sim.stop()
    if record: sim.save_video(args.video) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate Panda arm planar grasp execution in NVIDIA Omniverse Isaac Sim')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-P', '--path', type=str, nargs='+', metavar='', required=True, help='path to usd file or content folder')
    required.add_argument('-p', '--position', type=float, nargs=3, metavar='', required=True, help='grasp position, X Y Z')
    required.add_argument('-a', '--angle', type=float, metavar='', required=True, help='grasp angle in degrees')
    parser.add_argument('-l', '--location', type=str, metavar='', required=False, help='location of usd path, choices={local, nucleus_server}', choices=['local', 'nucleus_server'], default='local')
    parser.add_argument('-n', '--num', type=int, metavar='', required=False, help='number of objects to spawn in the scene', default=1)
    parser.add_argument('-v', '--video', type=str, metavar='', required=False, help='output filename of grasp simulation video')
    parser.add_argument('-W', '--width', type=int, metavar='', required=False, help='width of the viewport and generated images', default=1024)
    parser.add_argument('-H', '--height', type=int, metavar='', required=False, help='height of the viewport and generated images', default=800)
    args = parser.parse_args()

    print(args.path)

    main(args)