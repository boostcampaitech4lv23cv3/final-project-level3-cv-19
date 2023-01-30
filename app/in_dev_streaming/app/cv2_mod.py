import cv2


def get_stream_video():
    cam = cv2.VideoCapture(0)  # cam 정의
    while True:
        success, frame = cam.read()  # cam에서 데이터 불러오기

        if not success:
            break
        else:
            # 수신한 영상의 Frame을 Byte 형식으로 변경 후 인코딩
            ret_value, mat_buffer = cv2.imencode('.jpg', frame)
            frame = mat_buffer.tobytes()
            # yield로 각 프레임 데이터를 전달
            yield b'--PNPframe\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n'

