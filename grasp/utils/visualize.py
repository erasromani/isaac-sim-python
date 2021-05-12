import os
import ffmpeg
import matplotlib.pyplot as plt


def screenshot(sd_helper, suffix="", prefix="image", directory="images/"):
    """
    Take a screenshot of the current time step of a running NVIDIA Omniverse Isaac-Sim simulation.

    Args:
        sd_helper (omni.isaac.synthetic_utils.SyntheticDataHelper): helper class for visualizing OmniKit simulation
        suffix (str or int): suffix for output filename of image screenshot of current time step of simulation
        prefix (str): prefix for output filename of image screenshot of current time step of simulation
        directory (str): output directory of image screenshot of current time step of simulation
    """
    gt = sd_helper.get_groundtruth(
        [
            "rgb",
        ]
    )

    image = gt["rgb"][..., :3]
    plt.imshow(image)

    if suffix == "":
        suffix = 0

    if isinstance(suffix, int):
        filename = os.path.join(directory, f'{prefix}_{suffix:05}.png')
    else:
        filename = os.path.join(directory, f'{prefix}_{suffix}.png')
    plt.axis('off')
    plt.savefig(filename)


def img2vid(input_pattern, output_fn, pattern_type='glob', framerate=25):
    """
    Create video from a collection of images.

    Args:
        input_pattern (str): input pattern for a path of collection of images
        output_fn (str): video output filename
        pattern_type (str): pattern type for input pattern
        framerate (int): video framerate
    """
    (
        ffmpeg
        .input(input_pattern, pattern_type=pattern_type, framerate=framerate)
        .output(output_fn)
        .run(overwrite_output=True, quiet=True)
    )
