   # GAN 강의노트
GAN(Generative Adversarial Networks)을 역사적 맥락과 실습으로 배우는 정리 노트입니다.<br>
강의 내용과 소스코드는 pytoch를 중심으로 작성되어있으며, 정리 노트는 중급자 난이도로 구성했습니다. 

* pytorch의 모델 구성 및 torch 그리고 pytorch 훈련과정을 다루지만, 자세한 설명을 하지 않습니다.
## 목표
- GAN의 이론적 기초 이해
- 실습으로 생성 모델 직접 구현


## **Contents**
0. [GAN이란: 생성 모델의 시작](./docs/01_.md)
1. [GAN의 기본 구조와 학습 원리](./docs/01_.md)
2. [MNIST로 생성해보는 숫자 이미지](./docs/01_.md)
3. [CIFAR-10으로 생성해보는 컬러 이미지](./docs/01_.md)
4. [GAN의 응용: 흑백 이미지를 컬러로 변환하기](./docs/01_.md)
5. [Latent Space 탐구: GAN의 숨겨진 공간 이해하기](./docs/01_.md)
6. [GAN의 한계와 개선 방향](./docs/01_.md)
7. 
...(작성중)

서론: 생성 모델과 GAN의 등장
1.1 생성 모델이란 무엇인가?
1.2 GAN의 기본 개념과 역사
1.3 GAN이 바꾼 인공지능의 패러다임
GAN의 이론적 기초
2.1 생성자(Generator)와 판별자(Discriminator)의 역할
2.2 손실 함수와 게임 이론
2.3 훈련 과정의 수학적 이해
2.4 GAN의 안정성과 수렴 문제
첫 번째 GAN 구현
3.1 기본 GAN 구조 설계
3.2 MNIST 데이터셋으로 실습하기
3.3 훈련 팁과 디버깅 방법
GAN의 변형과 발전
4.1 DCGAN: 딥 컨볼루션 GAN
4.2 Conditional GAN: 조건 기반 생성
4.3 CycleGAN: 이미지 변환의 혁신
4.4 WGAN: Wasserstein 거리와 안정성 개선
4.5 StyleGAN: 스타일 기반 고품질 생성
GAN의 실습: 고급 프로젝트
5.1 CIFAR-10으로 컬러 이미지 생성
5.2 얼굴 이미지 생성 (CelebA 데이터셋 활용)
5.3 텍스트-이미지 변환 (Text-to-Image GAN)
5.4 음악 생성과 GAN (예: AudioGAN)
GAN의 응용 분야
6.1 이미지 보정 및 초해상도 (Super-Resolution)
6.2 데이터 증강(Data Augmentation)
6.3 의료 영상 생성
6.4 예술과 디자인에서의 활용 (예: AI 아트)
GAN의 한계와 해결책
7.1 모드 붕괴(Mode Collapse)와 그 대응
7.2 훈련 불안정성과 하이퍼파라미터 튜닝
7.3 윤리적 문제: 딥페이크와 오용 방지
최신 트렌드와 미래 전망
8.1 최신 GAN 연구 동향 (2025년 기준)
8.2 트랜스포머와 GAN의 융합
8.3 자율주행, 메타버스 등에서의 GAN
8.4 GAN의 다음 단계는?
부록
9.1 주요 GAN 논문 리스트
9.2 PyTorch와 TensorFlow로 구현한 코드 예제
9.3 용어 사전
9.4 추가 학습 리소스
## 기여
피드백과 PR 환영합니다!
