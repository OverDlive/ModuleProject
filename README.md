# ModuleProject
![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=10&height=200&text=Module%20Project&fontSize=50&animation=twinkling&fontAlign=68&fontAlignY=36)
## 프로젝트 개요

**실시간 얼굴 인식 접근 제어 시스템**은 얼굴 인식 기술을 활용하여 보안 구역에 대한 간단하고 효율적인 접근 제어를 제공하는 시스템입니다. 이 시스템은 MediaPipe AI API를 사용하여 실시간으로 얼굴을 탐지하고 인증하며, 데이터베이스와 비교하여 접근 권한을 관리합니다.

## 프로젝트 팀

- 팀명: FaceID
- 팀원 및 역할:

### Team
|<img src="https://avatars.githubusercontent.com/u/66999301?s=400&v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/60040347?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/55913669?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/191064967?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/191064925?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/144879167?v=4" width="150" height="150"/>
|:-:|:-:|:-:|:-:|:-:|:-:|
|한동혁<br/>[@OverDlive](https://github.com/OverDlive)<br/>PM, DB 구축|이정민<br/>[@sillage13](https://github.com/sillage13)<br/>모델 학습|송재섭<br/>[@Ddabong](https://github.com/songjae44)<br/>UI 및 모델 학습|석주원<br/>[@JW6022](https://github.com/JW6022)<br/>모델 학습|이채은<br/>[@LCEnetworksecurity](https://github.com/LCEnetworksecurity)<br/>데이터 수집|김준영<br/>[@yfhjhgk](https://github.com/yfhjhgk)<br/>데이터 수집


## 주요 목표

- 실시간 얼굴 인식을 통한 안전하고 신뢰할 수 있는 접근 제어 구현
- 기존 키카드나 비밀번호 접근 방식을 대체하는 사용자 친화적 시스템 제공

## 주요 기능

1. **실시간 얼굴 탐지**: MediaPipe Face Detection API를 사용하여 얼굴을 즉시 탐지하며, 이 API는 빠른 처리 속도와 높은 정확도를 제공하여 실시간 애플리케이션에 적합합니다.
2. **얼굴 인증**: 탐지된 얼굴을 데이터베이스와 비교하여 사용자를 인증.
3. **접근 제어**:
   - 얼굴 인증 성공 시 접근 허용.
   - 인증 실패 시 경고 알림(소리, 화면 경고 등).
4. **보안 알림**: 비인가 접근 시 실시간 경고 제공.
5. **활동 로그**: 시간 및 결과를 포함한 접근 시도 기록.

## 기술 스택

- **프로그래밍 언어**: Python
- **웹 프레임워크**: Streamlit
- **AI 라이브러리**: MediaPipe, OpenCV
- **데이터베이스**: SQLite
- **플랫폼**: Windows
- **하드웨어**: 웹캠 또는 기타 비디오 입력 장치

## 설치 및 실행 방법

1. **필수 라이브러리 설치**:
   ```bash
   pip install -r requirements.txt
   ```
2. **데이터베이스 설정**:
   ```bash
   python init_db.py
   ```
3. **애플리케이션 실행**:
   ```bash
   streamlit run app.py
   ```
4. **웹 인터페이스 접속**:
   - 브라우저에서 `http://localhost:8501`로 접속

## 시스템 아키텍처

1. **입력**: 카메라로부터 실시간 비디오 피드 수신.
2. **처리**:
   - MediaPipe를 사용하여 얼굴 특징 탐지.
   - 데이터베이스와 탐지된 얼굴 비교.
3. **출력**:
   - 인증 결과에 따른 접근 허용 또는 차단.
   - 비인가 접근 시 경고 알림 작동.
4. **저장**: 사용자 얼굴 데이터를 데이터베이스에 유지.

## 구현 단계

1. **환경 설정**: 필수 라이브러리 설치 및 구성.
2. **얼굴 탐지 모듈**: 카메라 피드를 통한 실시간 얼굴 탐지 개발.
3. **데이터베이스 통합**: 사용자 데이터를 저장하는 데이터베이스 설계.
4. **인증 로직 구현**: 얼굴 데이터 비교 알고리즘 개발.
5. **접근 제어 인터페이스**: 인증 결과와 경고를 표시하는 Streamlit 기반 UI 설계.
6. **테스트**: 다양한 조건에서 시스템 검증.

## 활용 사례

- **사무실 출입 관리**: 직원만 접근 가능하도록 보안 강화.
- **제한 구역 보호**: 민감한 구역에 대한 비인가 접근 차단.
- **스마트 홈 보안**: 가족 구성원 인증 및 미확인 인물 접근 차단.

## 향후 발전 가능성

- 다중 인증 통합으로 보안 강화.
- 대규모 데이터베이스 확장 지원.
- 마스크 감지 및 감정 인식 추가 기능 구현.

---

본 프로젝트는 개인 정보 보호를 준수하며, 사용자의 데이터 안전을 최우선으로 고려합니다.

