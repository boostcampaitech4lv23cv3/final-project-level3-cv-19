# 시각장애인 보행 시 장애물 안내 서비스
[네이버커넥트 부스트캠프 AI Tech 4기 CV-19조 최종 프로젝트]

## 1. 프로젝트 개요
### 📙 프로젝트 주제 

- 동영상으로 입력받은 보행자 시각 정보를 바탕으로 장애물 정보를 추출하여 사용자에게 전달하는 서비스 개발
- 📆 **프로젝트 기간** : 2023.01.25. ~ 2023.02.06. / 2주
- 🛠 **개발 환경**
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
| 박제원 |   Web Application (Backend & Frontend)  |
| 백하닮 |AI Modeling (YOLOv8) & Data Pre-processing|
| 유정호 | Server Environment & Project Management |
| 서성관 |AI Modeling (YOLOv7) & Data Pre-processing|


## 3. Project Description

### 📌 Process

1. Problem Definition

2. Application Design

3. AI Modeling

4. Product Serving


### 💻 Structure
```
final-project-level3-cv-19
├─ yolov8           # YOLOv8
├─ app              # Web Application Python Code Files
├─ assets           # Web Application Static Files
├─ Makefile         # Web Application Starting File
├─ REAMDME.md       # Project Markdown
├─ requirements.txt # Required Libraries (Dependencies)
└─***
```

### ✏️ Product Description
- Object Detection을 활용한 장애물 식별
- 사용자의 Requests에 기반한 On-Demand Machine Learning
- 사용자 단말에서 접속하여 GPU 서버에서 데이터 처리하는 (Semi-)Realtime Application
- TTS(Text To Speech) 기술을 활용한 비문자 정보 전달

### ❕ Dataset
- AI Hub **인도보행 영상** : [Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189)


### ❗ License
- Dataset : CC-BY-SA
- Application Icons : CC-BY
- PyTorch (Library) : Copyright on Facebook