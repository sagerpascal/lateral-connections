import os
from pathlib import Path


def fix_videos():
    file_names = {}
    for file in Path(".").rglob("*.mp4"):
        if file.is_file():
            old_name = file.absolute()
            new_name = (file.parent / (file.name[:-len(file.suffix)] + '__optimized_video' + file.suffix)).absolute()
            command = f"ffmpeg -i {old_name} -vcodec libx264 -f mp4 {new_name}"
            file_names[old_name] = new_name
            os.system(command)

    for old_name, new_name in file_names.items():
        Path(new_name).rename(old_name)


if __name__ == '__main__':
    fix_videos()
