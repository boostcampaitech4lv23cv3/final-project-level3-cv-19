import os
import pathlib
import json
from app.utils import dir_func

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)
    return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d},{int(millisecond):03d}"


def json2srt(session_id: str, json_str: str, fps=10, dst_fn="subtitle", save: bool = True):
    frame_time_in_sec = 1/fps
    ts_start = .0
    ts_end = ts_start + frame_time_in_sec
    subtitle_arr = []
    js_dict = json.loads(json_str)
    for frame_no, frame_items in sorted(js_dict.items(), key=lambda x: x[0], reverse=False):
        if len(frame_items):
            seq_line = f"{int(frame_no)}"
            time_line = f"{second_to_timecode(ts_start)} --> {second_to_timecode(ts_end)}"
            caption_line = "\n".join([f"IDX: {obj_id} // WARNING: {obj_data['warning_lv']} "
                                      f"// LOCATION: {obj_data['location']} // CLASS: {obj_data['class']}"
                                      f"" for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False)])
            break_line = ""
            subtitle_arr.append("\n".join([seq_line, time_line, caption_line, break_line]))

            ts_start = ts_end
            ts_end += frame_time_in_sec

    parsed_str = "\n".join(subtitle_arr)
    if save:
        save_path = os.path.join(APP_PATH, "srt", session_id)
        dir_func(save_path, rmtree=False, mkdir=True)
        with open(os.path.join(save_path, dst_fn+".srt"), 'w+', encoding='utf-8') as f:
            f.writelines(parsed_str)
    return parsed_str


def get_html(session_id: str, srt_fname: str = "subtitle") -> tuple:
    video_path = os.path.join(APP_PATH, "result", session_id)
    html_path = os.path.join(APP_PATH, "html", session_id)
    srt_path = os.path.join(APP_PATH, "srt", session_id)
    srt_fp = os.path.join(srt_path, srt_fname+".srt")
    dir_func(html_path, rmtree=True, mkdir=True)
    html_str = f"""
<!DOCTYPE html>
<html>
<head>
  <link href="https://cdn.jsdelivr.net/npm/video.js@7.9.3/dist/video-js.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/video.js@7.9.3/dist/video.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/videojs-sublime-skin@3.3.2/dist/videojs-sublime-skin.min.css"></script>
</head>
<body>
  <video id="my-video" class="video-js vjs-sublime-skin" controls preload="auto" width="640">
    <source src="{os.path.join(video_path, os.listdir(video_path)[0])}" type"video/mp4">
    <track src="{srt_fp}" kind="subtitles" srclang="ko" label="한국어">
  </video>
  <script>
    var player = videojs("my-video");
  </script>
</body>
</html>
    """
    html_fp = os.path.join(html_path, "subs.html")
    with open(html_fp, 'w+', encoding='utf-8') as f:
        f.writelines(html_str)
    return html_fp, html_str
