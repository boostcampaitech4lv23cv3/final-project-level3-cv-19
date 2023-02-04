import os
import pathlib
import re
import math
import shlex
import pynvml
from subprocess import check_call, PIPE, Popen

re_metadata = re.compile('Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,.*\n.* (\d+(\.\d+)?) fps')
pynvml.nvmlInit()
# device_count = 0
device_count = pynvml.nvmlDeviceGetCount()  # GPU HW Accel이 가능하면 주석 해제


def get_metadata(filename):
    """
    Get video metadata using ffmpeg
    """
    p1 = Popen(["ffmpeg", "-hide_banner", "-i", filename], stderr=PIPE, universal_newlines=True)
    output = p1.communicate()[1]
    matches = re_metadata.search(output)
    if matches:
        video_length = int(matches.group(1)) * 3600 + int(matches.group(2)) * 60 + int(matches.group(3))
        video_fps = float(matches.group(4))
    else:
        raise Exception("Can't parse required metadata")
    return video_length, video_fps


def split_segment(filename, n, by='size'):
    """
    Split video using segment: very fast but sometimes innacurate
    Reference https://medium.com/@taylorjdawson/splitting-a-video-with-ffmpeg-the-great-mystical-magical-video-tool-%EF%B8%8F-1b31385221bd
    """
    assert n > 0
    assert by in ['size', 'count']
    split_size = n if by == 'size' else None
    split_count = n if by == 'count' else None

    # Parse meta data
    video_length, video_fps = get_metadata(filename)
    # Calculate split_count
    if split_size:
        split_count = math.ceil(video_length / split_size)
        if split_count == 1:
            raise Exception("Video length is less than the target split_size.")
    else:
        # Use split_count
        split_size = round(video_length / split_count)
    pth, ext = filename.rsplit(".", 1)
    uploaded_path, session_id, fn = pth.rsplit("/", 2)
    save_path = str(pathlib.Path(uploaded_path).parent.joinpath("tmp").joinpath(session_id))
    cmd = f'ffmpeg -nostdin -hide_banner -loglevel panic -i "{filename}" -c copy -an -map 0 -segment_time {split_size} -reset_timestamps 1 -g {round(split_size * video_fps)} -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*{split_size})" -f segment -y "{save_path}/{fn}-%04d.{ext}"'
    check_call(shlex.split(cmd), universal_newlines=True)
    return os.listdir(save_path)


def split_cut(filename, n, by='size'):
    """
    Split video by cutting and re-encoding: accurate but very slow
    Adding "-c copy" speed up the process but causes imprecise chunk durations
    Reference: https://stackoverflow.com/a/28884437/1862500
    """
    assert n > 0
    assert by in ['size', 'count']
    split_size = n if by == 'size' else None
    split_count = n if by == 'count' else None

    # Parse meta data
    video_length, video_fps = get_metadata(filename)
    # Calculate split_count
    if split_size:
        split_count = math.ceil(video_length / split_size)
        if split_count == 1:
            raise Exception("Video length is less than the target split_size.")
    else:
        # Use split_count
        split_size = round(video_length / split_count)
    output = []
    for i in range(split_count):
        split_start = split_size * i
        pth, ext = filename.rsplit(".", 1)
        uploaded_path, session_id, fn = pth.rsplit("/", 2)
        save_path = str(pathlib.Path(uploaded_path).parent.joinpath(session_id))
        output_path = f'{save_path}/{fn}-{i + 1 :04}.{ext}'
        cmd = f'ffmpeg -nostdin -i "{filename}" -ss {split_start} -t {split_size} -vcodec copy -an -y "{output_path}"'
        check_call(shlex.split(cmd), universal_newlines=True)
        output.append(output_path)
    return output


def concatenate(file_path: str, dst_file: str):
    txt_fp = os.path.join(file_path, "fileList.txt")
    with open(txt_fp, "w+", encoding='utf-8') as f:
        for fn in sorted(os.listdir(file_path)):
            if fn.endswith("mp4"):
                f.write(f"file '{fn}'\n")
    cmd = f"ffmpeg -nostdin -f concat -safe 0 -i {txt_fp} -c copy {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)


def h264_encoding(file_path: str, dst_file: str):
    vcodec = "h264_nvenc" if device_count != 0 else "libx264"
    cmd = f"ffmpeg -nostdin -y -i {file_path} -vcodec {vcodec} -profile:v high -preset slow -pix_fmt yuv420p -src_range 1 -dst_range 1 -g 30 -bf 2 -an -movflags faststart {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)


def video_preprocessing(file_path: str, dst_file: str, resize_h=None, tgt_framerate=None):  # For increasing Video Process Performance
    resizing_cmd = ""
    framerate_chg_cmd = ""
    if resize_h is None and tgt_framerate is None:
        print("Both 'resize_h' and 'tgt_framerate' should not be None, Designate at least one parameter")
        raise Exception

    if isinstance(resize_h, int):
        resizing_cmd = f" -vf scale=-1:{resize_h}"
    elif resize_h is not None:
        print("Something wrong in parameter 'resize_h', please check again")
        raise TypeError

    if isinstance(tgt_framerate, int):
        framerate_chg_cmd = f" -r {tgt_framerate}"
    elif tgt_framerate is not None:
        print("Something wrong in parameter 'tgt_framerate', please check again")
        raise TypeError
    vcodec = "h264_nvenc" if device_count != 0 else "libx264"
    cmd = f"ffmpeg -nostdin -y -i {file_path}{resizing_cmd} -vcodec {vcodec} -an -movflags faststart{framerate_chg_cmd} -y {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)


def combine_video_audio(img_dir: str, audio_file_path: str, dst_file: str, fps=15):
    img_list = [f"file '{file_name}'\n" for file_name in sorted(os.listdir(img_dir)) if file_name.endswith(".jpg")]
    file_list_path = f"{img_dir}/filelist.txt"
    with open(file_list_path, 'w+', encoding="utf-8") as f:
        f.writelines(img_list)
    vcodec = "h264_nvenc" if device_count != 0 else "libx264"
    cmd = f"ffmpeg -nostdin -y -f concat -safe 0 -r {fps} -i {file_list_path} -i {audio_file_path} -vcodec {vcodec} {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)
