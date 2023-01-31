import pathlib
import os

ROOT_PATH = pathlib.Path.root


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)
    return f"{hour:.2d}:{minute:.2d}:{second:.2d},{millisecond:.3d}"


def json2srt(src: dict, fps=10, dst=None, save: bool = False):
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
        with open(dst, 'w+', encoding='utf-8') as f:
            f.writelines(parsed_str)
    return parsed_str


def get_html_with_subvideo(session_id: str, sub_input=None) -> str:
    if sub_input is not None:
        srt = json2srt(sub_input)
    else:
        srt = open(os.path.join(ROOT_PATH, "srt", session_id, "subtitle.srt"), encoding="utf-8").read()

    html_path = os.path.join(ROOT_PATH, "html", session_id, "subtitle.html")
    return html_path
