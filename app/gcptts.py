"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16
)

# 읽을 텍스트 생성
class_names = ['자전거','버스','자동차','손수레','고양이','개','오토바이','간판','사람',
        '스쿠터','유모차','트럭','휠체어','바리케이드','벤치','볼라드','의자','소화전',
        '키오스크','주차요금정산기','기둥','화분','전력제어함','정류장','탁자','신호등',
        '신호등제어기','교통표지판','가로수']
class_postpositions = ['가', '가', '가', '가', '가', '가', '가', '이', '이',
                        '가', '가', '이', '가', '가', '가', '가', '가', '이', 
                        '가', '가', '이', '이', '이', '이', '가', '이',
                        '가', '이', '가' ]        
location_names = ['왼쪽','중앙','오른쪽']
warning_names = ['가까이 ','']

#warning->location->class 순으로 인덱스 부여
fileindex = 0
for class_name_index, class_name in enumerate(class_names):
    classstr = class_name + class_postpositions[class_name_index] + " "
    for location_name in location_names:
        locationstr = location_name + "에 "
        for warning_name in warning_names:
            warningstr = warning_name + "있습니다."

            # 읽어줄 문장
            textstr = classstr + locationstr + warningstr

            print(textstr + "\n")
            
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=textstr)

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # The response's audio_content is binary.
            with open(str(fileindex) + ".wav", "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)
                fileindex = fileindex + 1
      