# FurEmotion App

> "반려동물 감정 및 상황 해석을 위한 울음 감지 및 분석 어플리케이션" 프로젝트의 Cry-Detect AI Repository입니다.

201904008 곽재원, 202204174 권순우

<br>

## 프로젝트 소개

본 프로젝트는 대한민국 반려동물 시장의 성장에 비해 반려동물에 대한 이해와 전문적 인프라가 부족한 문제를 해결하기 위해 기획되었습니다. 매년 많은 반려동물이 유기되고 보호소에서 자연사하거나 안락사되는 상황을 개선하고자, AI 기술을 활용해 반려동물의 울음소리를 실시간으로 감지하고 분석하여 감정 및 상황적 정보를 파악하는 것이 핵심 목표입니다. 이를 통해 반려인이 반려동물의 감정을 더 잘 이해하고 적절히 대응할 수 있도록 돕고, 나아가 반려동물의 건강 관리와 행동 교정에도 기여하고자 합니다. FastAPI 기반의 비동기 실시간 처리와 직관적인 스마트폰 애플리케이션을 통해 반려동물의 상태를 모니터링하고, 반려동물과 반려인 간의 소통 강화 및 복지 증진, 유기 및 학대 문제 감소를 목표로 합니다.

## Cry-Detect AI Repository 소개

### 레포지토리 사용 주의사항(중요!)

반려동물의 울음 감지 AI는 Google에서 개발한 Yamnet을 fine-tuning하여 개발하였습니다. yamnet의 라이센스에 따라 **모델 파일은 공개할 수 없습니다.** 따라서, 이 레포지토리에는 **모델 파일이 포함되어 있지 않습니다.**. 아래 제시된 순서를 통해 모델 파일을 다운로드 받아야 하며 **제시된 정확한 버전의 라이브러리를 사용**해야 합니다.

1. 현재 레포지토리를 가져옵니다.

```bash
git clone https://github.com/FurEmotion/FurEmotion-Cry-Detect-AI.git
cd FurEmotion-Cry-Detect-AI
```

2. 새로운 가상환경 생성 및 활성화

```bash
conda create -n fur_emotion python=3.9.12
conda activate fur_emotion
```

3. 필요한 라이브러리 설치(다음 명시되지 않은 라이브러리들은 버전 상관없이 최신 버전으로 설치)

```bash
pip install tensorflow==2.15.1
pip install keras==2.15.0
pip install tf-keras==2.15.1
pip install tensorflow-io==0.36.0
pip install resampy pysoundfile
```

4. yamnet 모델을 받아옵니다.

```bash
# tensorflow/models 레포지토리를 클론.
git clone https://github.com/tensorflow/models.git
cd models/research/audioset/yamnet

# yamnet 가중치 파일을 다운로드.
curl -O https://storage.googleapis.com/audioset/yamnet.h5

# 정상 설치 확인
python yamnet_test.py

# 다시 루트 디렉토리로 이동
cd ../../../..
```

5. Clone한 yamnet 폴더에 본 레포지토리의 `yamnet_new.py`와 `main.py`, `cat.wav` 파일을 복사합니다.

```bash
# 파일 복사
cp yamnet_new.py main.py cat.wav models/research/audioset/yamnet

# yamnet 폴더로 이동
cd models/research/audioset/yamnet
```

6. `main.py`를 실행하여 파인튜닝을 수행합니다.

```bash
python main.py
```

7. 모델이 정상적으로 학습되었는지 확인합니다.

```bash
# cat이라고 뜨면 정상.
python inference.py cat.wav
```

### 시스템 구조

![시스템 구조](https://raw.githubusercontent.com/FurEmotion/FurEmotion-Backend/refs/heads/main/static/system_architecture.png)

- App은 **Android**와 **IOS**를 동시에 개발할 수 있는 크로스 플랫폼 **Flutter**를 사용합니다.
- Backend과 **Restful API**로 통신합니다.
- **Tensorflow lite**를 사용하여 **AI 모델**을 개발합니다.
- **Firebase Auth**를 사용하여 소셜 로그인을 구현합니다.
- **JWT**를 사용하여 사용자 인증을 구현합니다.

<br>

### 시스템 플로우

![시스템 플로우](https://raw.githubusercontent.com/FurEmotion/FurEmotion-Backend/refs/heads/main/static/systm_flow.png)

전체 시스템 플로우는 위와 같습니다.

1. 어플리케이션에서 사용자가 울음감지 버튼을 누르면 **실시간으로 울음을 감지** 시작합니다.
2. 들은 울음을 **AI 모델**을 통해 **반려동물의 울음소리 여부**를 분석합니다.
3. 만약 반려동물의 울음소리가 맞다면 Backend로 **울음소리 데이터**를 전송합니다.
4. Backend에서 울음소리 데이터를 **분석**하여 **반려동물의 감정 및 상황**을 분석합니다.
5. 분석된 결과를 **어플리케이션**으로 전송하여 사용자에게 **감정 및 상황**을 보여줍니다.

이때 울음소리가 판별될 경우 사용자의 웨어러블 디바이스로 **알림**을 전송하여 감지 여부를 알려줍니다.

<br>

### 개발 플로우

- **MVC 패턴**: **Model**, **Service**, **Controller**로 구분하여 개발합니다.
- **에자일 방법론**: **스프린트** 단위로 개발을 진행합니다.
- **Trunk-based Development**: **Main 브랜치**를 기반으로 개발을 진행합니다.

<br>

## 개발 일정

### 주요 일정 소개

- **프로젝트 기획**: 프로젝트 목표 및 범위를 설정하고, 기능 명세를 작성함.
  - **1주차**: 프로젝트 준비 및 요구사항 정의: 고객 및 반려동물 보호자의 요구사항을 바탕으로 주요 기능 및 성능 목표 설정.
  - **2~3주차**: 반려동물 울음소리 데이터 수집(공개 데이터셋 활용) 및 전처리. 데이터 정제 및 필터링 작업 수행.
- **1차 중간평가**: 울음 감지 모델을 어플리케이션에 탑재 및 시연.
  - **4~5주차**: MobileNet 기반 on-device AI 모델 개발 및 초기 테스트(on-device 구현 유효성 판단). 울음소리 감지를 위한 실시간 분류 시스템 구축.
  - **6~7주차**: 푸리에 변환 및 멜 필터뱅크 적용하여 울음소리 데이터를 3D 이미지로 변환. Computer Vision 기술(CNN 또는 ViT) 적용 및 훈련.
  - **8주차**: Flutter 기반 모바일 애플리케이션 UI/UX 설계 및 프로토타입 개발. 음성 데이터 실시간 감지 기능 통합.
- **2차 중간평가**: Backend와 연동된 전체 서비스 프로토타입 시연.
  - **9~10주차**: FastAPI 기반 백엔드 시스템 개발. TensorFlow 또는 PyTorch와 연동해 AI 모델과 실시간 데이터 처리 연동.
  - **11주차**: 모바일 기기 성능 최적화 작업(발열, 지연 문제 해결). on-device AI 모델 성능 개선 및 모바일 환경에 맞춘 경량화 작업.
  - **12주차**: 모바일 애플리케이션과 백엔드 시스템 간의 통신 안정화. 테스트 환경에서의 통합 테스트 진행.
- **최종평가**: 어플리케이션 및 Backend 서버를 배포함.
  - **13~14주차**: 추가 데이터 수집 및 다양한 모델 아키텍쳐 적용을 통한 성능 향상. 데이터 증강 기법을 적용해 모델의 정확도 및 세부 감정/상황 분류 성능 향상.
  - **15주차**: 최종 통합 및 사용자 테스트. 실사용자 피드백 반영하여 최종 버전 개선 및 배포 준비.
  - **최종발표**: 프로젝트 결과 및 성과를 발표함.

<br>

### 세부 일정 및 목표 소개

#### 1차 중간평가

1차 중간평가까지는 Flutter를 이용한 App과 울음감지 모델을 연동한 Frontend 개발이 주요 목표입니다.

따라서 **울음 감지 및 유저 보안 로직(JWT)를 탑재한 어플리케이션이 1차 중간평가의 프로도타입으로 제작될 예정**입니다.

이에 따라 App 1차 중간평가까지 다음 내용을 개발할 예정입니다.

- 울음 감지 모델 Fine-tuning: 울음 감지 모델을 Fine-tuning하여 정확로 80% 이상 달성

<br>

#### 2차 중간평가

2차 중간평가까지는 Backend와 Frontend를 연동한 전체 서비스 프로토타입을 제작하는 것이 주요 목표입니다.

따라서 **Backend와 연동하여 울음 분석 결과를 사용자에게 보여주는 어플리케이션이 2차 중간평가의 프로토타입으로 제작될 예정**입니다.

<br>

#### 최종평가

최종평가까지는 Backend와 Frontend를 완성한 서비스를 배포하는 것이 주요 목표입니다.

따라서 **배포를 위한 서버 환경 설정 및 최적화, 울음 기록 분석 및 시각화 기능을 추가한 어플리케이션 배포**입니다.
