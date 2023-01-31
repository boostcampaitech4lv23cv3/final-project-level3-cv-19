import pathlib
import os
from app.main import dir_func

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)
    return f"{hour:.2d}:{minute:.2d}:{second:.2d},{millisecond:.3d}"


def json2srt(src: dict, fps=10, session_id=None, dst_fn="subtitle", save: bool = False):
    frame_time_in_sec = 1/fps
    ts_start = .0
    ts_end = ts_start + frame_time_in_sec
    subtitle_arr = []

    for frame_no, frame_items in sorted(src.items(), key=lambda x: x[0], reverse=False):
        seq_line = f"{frame_no+1}"
        time_line = f"{second_to_timecode(ts_start)} --> {second_to_timecode(ts_end)}"
        caption_line = "\n".join([f"IDX: {obj_id} // WARNING: {obj_data['warning_lv']} // LOCATION: {obj_data['location']} // CLASS: {obj_data['class']}" for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False)])
        break_line = "\n"
        subtitle_arr.append("\n".join([seq_line, time_line, caption_line, break_line]))

        ts_start = ts_end
        ts_end += frame_time_in_sec

    parsed_str = "\n".join(subtitle_arr)
    if save:
        save_path = os.path.join(APP_PATH, "srt", session_id, dst_fn)
        dir_func(save_path, rmtree=False, mkdir=True)
        with open(save_path, 'w+', encoding='utf-8') as f:
            f.writelines(parsed_str)
    return parsed_str


def get_html(session_id: str, srt_fname: str = "subtitle", sub_json_input=None) -> str:
    srt_path = os.path.join(APP_PATH, "srt", session_id, srt_fname+".srt")
    video_path = os.path.join(APP_PATH, "result", session_id)
    if sub_json_input is not None:
        with open(srt_path, 'w+', encoding='utf-8') as f:
            f.writelines(json2srt(sub_json_input))
    html_path = os.path.join(APP_PATH, "html", session_id)
    dir_func(html_path, rmtree=False, mkdir=True)
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
    <source src="{os.path.join(video_path, os.listdir(video_path)[0])}" type='video/mp4'>
    <track src="{srt_path}" kind="subtitles" srclang="ko" label="한국어">
  </video>
  <script>
    var player = videojs('my-video');
  </script>
</body>
</html>
    """
    with open(html_path+"subs.html", 'w+', encoding='utf-8') as f:
        f.writelines(html_str)
    return html_path+"subs.html"
