# Greatest-Classify-Group
Respiratory Diseases Classification Using Audio Data  

## 프로젝트 목표

[ICBHI 2017 Challenge Respiratory Sound Database](https://www.kaggle.com/vbookshelf/respiratory-sound-database)을 이용한 질병 분류 작업을 진행하였습니다.  

정확도 향상을 목표로 기존 작업물([Base논문](https://eden.dei.uc.pt/~ruipedro/publications/Conferences/ICBHI2017a.pdf), [관련작업](https://www.kaggle.com/eatmygoose/cnn-detection-of-wheezes-and-crackles))과 다른 전처리 과정 및 학습법을 적용하였습니다. 

---

## 목차 
- [데이터 전처리](#데이터-전처리)
- [학습과정](#학습과정)
- [검증](#검증)
- [추가작업](#추가작업)
- [회의록 모음](#회의록-모음)

---

## 데이터 전처리

* STFT Filter
  * 개발환경 구축, 음원 데이터 분류 및 시각화, 피처추출, CNN 모델 설계, 학습에 필요한 전처리, 피처추출 과정 모듈화(완료)
  * 187개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * 정확도 향상을 위한 다양한 시도와 성능 비교(잡음 섞인 데이터셋 추가, 다른 Filter 사용, SVM 학습법 적용 (진행중)
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)
  
* MFCC Filter
  * MFCC 
  * MFCC

* MFCC Delta Filter
  * MFCC 

* 이미지 생성 모듈화
  * 이미지 생성 모듈화

---

## 학습과정
  
* CNN 학습 진행
  * 187개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)

---

## 검증

* 질분 유무 판단(진행중)
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)

---

## 추가작업

* 다중 필터 사용, SVM 학습 진행
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * 정확도 향상을 위한 다양한 시도와 성능 비교(잡음 섞인 데이터셋 추가, 다른 Filter 사용, SVM 학습법 적용 (진행중)


---
## [회의록 모음](https://github.com/Hongiee2/Greatest-Classify-Group/issues/2)
