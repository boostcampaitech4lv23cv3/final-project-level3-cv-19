import os
import pathlib
import json
from utils import dir_func
from pydub import AudioSegment

PRJ_ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
APP_PATH = os.path.join(PRJ_ROOT_PATH, "app")

def json2audio(session_id: str, json_str: str, fps=10, dst_fn="synthesizedaudio", save: bool = True) -> str:

    class_names_index = {'bicycle':0,'bus':1,'car':2,'carrier':3,'cat':4,
    'dog':5,'motorcycle':6,'movable_signage':7,'person':8,'scooter':9,
    'stroller':10,'truck':11,'wheelchair':12,'barricade':13,'bench':14,
    'bollard':15,'chair':16,'fire_hydrant':17,'kiosk':18,'parking_meter':19,
    'pole':20,'potted_plant':21,'power_controller':22,'stop':23,'table':24,
    'traffic_light':25,'traffic_light_controller':26,'traffic_sign':27,'tree_trunk':28}

    frame_time_in_sec = 1/fps
    frame_time_in_ms = 1000/fps
    ts_start = .0
    ts_end = ts_start + frame_time_in_sec
    ts_audio = .0
    js_dict = json.loads(json_str)
    silent_1sec = AudioSegment.from_wav("./app/wav/silent_1-second.wav")
    silent_1frame = silent_1sec[:int(frame_time_in_ms)] # (slicing is done by milliseconds)
    synthesizedaudio = silent_1sec[0] # 1ms
    foundobj = False
    for frame_no, frame_items in sorted(js_dict.items(), key=lambda x: x[0], reverse=False):
        foundobj = False
        if (ts_start >= ts_audio):
            if len(frame_items):
                for obj_id, obj_data in sorted(frame_items.items(), key=lambda x: x[0], reverse=False):
                    if (obj_data['warning_lv'] == "1"):
                        #warning->location->class 순으로 인덱스 부여
                        #클래스 29개 x 위치 3개 x 경고수준 2개 = 174개 wav
                        wavindex = 2 * int(obj_data['location']) + 6 * class_names_index[obj_data['class']]
                        if (wavindex <= 173) and (wavindex >= 0):
                            obj_warnaudio = AudioSegment.from_wav("./app/wav/" + str(wavindex) + ".wav")
                            gapframe_in_ms = int(frame_time_in_ms) - (len(obj_warnaudio) % int(frame_time_in_ms))
                            obj_warnaudio_1frame = obj_warnaudio + silent_1sec[:gapframe_in_ms-1]
                            synthesizedaudio += obj_warnaudio_1frame
                            ts_audio += len(obj_warnaudio_1frame) / 1000.0 # ms -> sec
                            foundobj = True
                            break

            if (foundobj == False) and (ts_start <= ts_audio):
                synthesizedaudio += silent_1frame
                ts_audio += len(silent_1frame) / 1000.0 # ms -> sec

        ts_start = ts_end
        ts_end += frame_time_in_sec

    if save:
        save_path = os.path.join(APP_PATH, "audio", session_id)
        dir_func(save_path, rmtree=False, mkdir=True)
        save_path = os.path.join(save_path, "synthesizedaudio.wav")
        synthesizedaudio.export(save_path, format="wav")
    return save_path

#with open("./app/preprocessed.json", "rb") as f:
#    print(json2audio("1234", f, fps=10, dst_fn="synthesizedaudio", save = True))
