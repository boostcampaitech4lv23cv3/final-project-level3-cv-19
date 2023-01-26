import os
import re
import math
import shlex
import ffmpeg
from subprocess import check_call, PIPE, Popen

re_metadata = re.compile('Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,.*\n.* (\d+(\.\d+)?) fps')


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
    _, pth = pth.rsplit("/", 1)
    cmd = f'ffmpeg -nostdin -hide_banner -loglevel panic -i "{filename}" -c copy -an -map 0 -segment_time {split_size} -reset_timestamps 1 -g {round(split_size * video_fps)} -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*{split_size})" -f segment -y "{pth}-%d.{ext}"'
    check_call(shlex.split(cmd), universal_newlines=True)
    return [f'app/tmp/{pth}-{i}.{ext}' for i in range(split_count)]


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
        _, pth = pth.rsplit("/", 1)
        output_path = f'app/tmp/{pth}-{i + 1 :04}.{ext}'
        cmd = f'ffmpeg -nostdin -i "{filename}" -ss {split_start} -t {split_size} -vcodec copy -an -y "{output_path}"'
        check_call(shlex.split(cmd), universal_newlines=True)
        output.append(output_path)
    return output


def concatenate(file_path: str, dst_file: str):
    txt_fp = os.path.join(file_path, "fileList.txt")
    with open(txt_fp, "w+", encoding='utf-8') as f:
        for fn in os.listdir(file_path):
            if fn.endswith("mp4"):
                f.write(f"file '{fn}'\n")
    cmd = f"ffmpeg -nostdin -f concat -safe 0 -i {txt_fp} -c copy {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)


def h264_encoding(file_path: str, dst_file: str):
    cmd = f"ffmpeg -i {file_path} -vcodec libx264 -profile:v high -preset slow -pix_fmt yuv420p -src_range 1 -dst_range 1 -crf 18 -g 30 -bf 2 -an -movflags faststart {dst_file}"
    check_call(shlex.split(cmd), universal_newlines=True)
