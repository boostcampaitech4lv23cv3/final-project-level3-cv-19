# 통행약자 인도 보행 시 장애물 안내 서비스
[네이버커넥트 부스트캠프 AI Tech 4기 CV-19조 최종 프로젝트]

## 1. 프로젝트 개요
### 📙 프로젝트 주제 

- 동영상으로 입력받은 보행자 시각 정보를 바탕으로 장애물 정보를 추출하여 사용자에게 전달하는 서비스 개발
- 📆 **프로젝트 기간** : 2023.01.25. ~ 2023.02.06. / 2주
- 🛠 **개발 환경**
  - 시스템 환경 : Tesla V100, Docker, GCP, Python(3.8.13), PyTorch(1.7.1), FFmpeg(3.4.11 & 5.1)
  - 개발 환경 : VSCode, PyCharm, Jupyter Notebook, GPU(Tesla V100)
  - 협업 Tools : GitHub, Notion, Zoom

## 2. 팀원 소개 

<table>
  <tr>
    <td align="center"><a href="https://github.com/RADM90"><img src="https://avatars.githubusercontent.com/u/69555670?v=4" width="100px;" alt=""/><br /><sub><b>박제원_T4092<br></b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/hdak95"><img src="https://avatars.githubusercontent.com/u/37134920?v=4" width="100px;" alt=""/><br /><sub><b>백하닮_T4103<br></b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/JunghoYoo"><img src="https://avatars.githubusercontent.com/u/10891644?v=4" width="100px;" alt=""/><br /><sub><b>유정호_T4138<br></b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ths3847"><img src="https://avatars.githubusercontent.com/u/46395571?v=4" width="100px;" alt=""/><br /><sub><b>서성관_T4239<br></b></sub></a><br /></td>
    </tr>
</table>


### 👨‍👨‍👦‍👦 팀원 역할

|  팀원  |                    역할                   |
| :----: |:----------------------------------------|
| 박제원 |Team Leader, Web App(BE & FE), Video Handling w/ FFmpeg, Subtitle Function|
| 유정호 |Docker w/ TensorRT, Video Handling w/ FFmpeg, TTS(GCP, Audio Mixing)|
| 백하닮 |AI Modeling (YOLOv8), Detection Result Processing w/ OpenCV|
| 서성관 |AI Modeling (YOLOv7)|


## 3. Project Description

### 📌 Process

1. **Problem Definition**
   - 시각장애인을 포함한 통행약자에게 볼라드, 자전거 등 통행에 장애가 되는 요인이 산재
   - 동영상 등 시각정보를 바탕으로 이동에 장애가 되는 물체를 식별, 비문자적 수단(음성 등)을 통해 사용자에게 해당 객체에 대한 정보를 전달하여 안전에 도움이 될 수 있는 소프트웨어 개발 필요

2. **Application Design**
   - 인도보행시 충돌위험이 존재하는 이동체와 고정체를 포함한 29종 장애물 탐지
   - 식별된 Object들에 대한 위치 정보(Pixel base) 분석
   - 장애물과의 충돌 위험 정도를 상기 위치 정보로 구분
   - 안전 / 주의 / 위험 3단계의 경고 수준 정의
   - 위험 수준의 장애물 정보를 TTS(Text-to-Speech) 활용 비문자 데이터(음성) + 자막으로 전달

3. **AI Modeling**
   - AI-Hub의 인도보행 영상 데이터셋에서 제한된 시간과 자원을 고려, 학습에 유리한 데이터를 선별하여 사용
   - 동영상 데이터의 볼륨과 모바일 환경 확장 고려, 낮은 복잡도와 빠른 추론 속도를 제공하는 `YOLOv8` 중 가장 가벼운 `YOLOv8n` 선정
   
4. **Product Serving**
   - 충돌 위험 수준을 3단계로 정의하고, 모델에서 탐지한 결과를 필터링하여 `JSON` 형태로 TTS, 자막 처리 모듈에 전송
   - `Streamlit` 라이브러리를 사용하여 Web Application을 생성하고, 처리된 동영상(음성 포함)과 자막을 웹페이지에 전시하기 위해 `FastAPI`로 파일 호스팅
   - 동영상(전처리 및 후처리)과 음성 처리에 사용되는 소프트웨어는 `FFmpeg`와 `OpenCV`, `Google Cloud Platform`
   - 프로세스는 다음과 같음
     1) 동영상 업로드
     2) 동영상 전처리
     3) Object Detection
     4) Detection Result Processing (`JSON` to TTS & Subtitle)
     5) Data Combining (IMG files + TTS Audio)
     6) File Hosting
     7) 동영상 전시


### 💻 Structure
```
final-project-level3-cv-19
├─ app                  # Web Application Python Code Files
│   ├─ wav              # Static TTS(Text-to-Speech) `.wav` files Dir
│   ├─ audio_func.py    # JSON Data to Audio Mixing Function
│   ├─ ffmpeg_func.py   # Video Data Handler (FFmpeg functions)
│   ├─ subtitle_func.py # JSON Data to Subtitle Converter
│   ├─ gcptts.py        # TTS Voice Data Generator
│   ├─ live_server.py   # (Result) File Hosting Server
│   └─ main.py          # Main Streamlit(FE & BE) file
├─ EDA                  # Exploratory Data Analysis `.ipynb` files Dir
├─ Model                # ML/DL Related files Dir
│   ├─ onnx_tensorrt    # TensorRT Ported Implementation
│   ├─ yolov8           # YOLOv8 Weight File Dir
│   └─ detector.py      # Main Object Detection Function Python file
├─ Makefile             # Web Application Starting File
├─ REAMDME.md           # Project Markdown
├─ requirements.txt     # Required Libraries (Dependencies)
└─ .gitmodules          # Submodules Defined
```

### ✏️Product Description
- Object Detection을 활용한 장애물 식별
- 사용자의 Requests에 기반한 On-Demand Machine Learning
- 사용자 입력 동영상으로 GPU 서버에서 데이터를 처리하는 Application
- TTS(Text To Speech) 기술을 활용한 비문자 정보 전달


### 🎞 Demonstration
> 데모영상 보러가기 : [YouTube Link](https://youtu.be/wYuviCwY80c)


### ❕ Dataset
- AI-Hub 인도보행 영상 : [Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189)


### ❗ License
- Dataset : CC-BY-SA
- `Streamlit` : Apache 2.0
- `PyTorch` : Copyright on Facebook
- `YOLOv8` : GPL 3.0
- `Pydub` : MIT
- `FastAPI` : MIT
- `FFmpeg` : GPLv2 | LGPL 2.1


---
# HOW-TO-USE

### *Disclaimer*
> *Following Steps are based on the status that `CUDA Toolkit` and `CuDNN` has been installed.*

> ***You can also use pre-set Docker file on `Model/onnx_tensorrt/Dockerfile` to avoid following messy installation.***

---
## Step 1. ***Installations***

### Step 1-1. ***Install `Make`***

- On Windows

  > [안경잡이개발자(나동빈)님 블로그 링크 참고](https://ndb796.tistory.com/381)

- On Linux (Ubuntu base)

  ```shell
  apt-get install gcc make
  ```
  

### Step 1-2. ***Install `FFmpeg` and `NVIDIA nv-codec-headers`***

- On Windows

  > [천동이님 블로그 링크 참고](https://m.blog.naver.com/chandong83/222095346417)

- On Linux
  ```shell
  make step2
  ```

### Step 1-3. ***Install compatible version of `PyTorch`***
> See the [PyTorch Official Installation Method](https://pytorch.org/get-started/locally/#start-locally)


### Step 1-4. ***Install required libraries in `requirements.txt`***
  ```shell
  cd final-project-level3-cv-19
  python -m pip install -r requirements.txt
  ```

---
## Step 2. Run "2" of Applications
### ***Step 2-1. Run Streamlit Application***
  ```shell
  make run
  ```
### ***Step 2-2. Execute Another Terminal***
### ***Step 2-3. Run Hosting Server on `Step 2-2` Terminal***
- This server runs for hosting local Video & Subtitle files to Internet with URL. 
```shell
make server
```
---
## Step 3. Upload the Video file on Streamlit Page, and Press `Start Process` Button
> Have Fun XD