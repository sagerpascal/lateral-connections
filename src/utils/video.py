import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from utils import print_exception, print_info_data


def create_video_from_images_ffmpeg(
        images: Union[Path | str],
        video_fp: Union[Path | str],
        fps: Optional[float] = 2.0,
):
    """
    Create a video from all images inside a folder.
    :param images: Paths to a folder with images.
    :param video_fp: Filepath to the video
    :param fps: Frames per second.
    """
    print_info_data(f"Creating video from images in folder {images}...")

    command = f"ffmpeg -framerate {fps} -pattern_type glob -i '{images}/*.png' -c:v libx264 -pix_fmt yuv420p {video_fp} -y"

    try:
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call([command], shell=True, stdout=devnull, stderr=subprocess.STDOUT)

        print_info_data(f"Video saved to {video_fp}")

    except Exception as e:
        print_exception(e)
