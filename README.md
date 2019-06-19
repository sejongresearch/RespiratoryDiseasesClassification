# Greatest-Classify-Group
Respiratory Diseases Classification Using Audio Data  
  
  
  
## 목차 
- [프로젝트 목표](#프로젝트-목표)
- [진행경과](#진행경과)
- [회의록](#회의록)

---

## 프로젝트 목표

* ICBHI 2017 Challenge Respiratory Sound Database을 사용한 질병 유무 판단의 정확도 향상
* ICBHI 2017 Challenge Respiratory Sound Database을 사용한 질병 분류의 정확도 향상


---

## 진행경과

* 질분 유무 판단(진행중)
  * 개발환경 구축, 음원 데이터 분류 및 시각화, 피처추출, CNN 모델 설계, 학습에 필요한 전처리, 피처추출 과정 모듈화(완료)
  * 187개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * 정확도 향상을 위한 다양한 시도와 성능 비교(잡음 섞인 데이터셋 추가, 다른 Filter 사용, SVM 학습법 적용 (진행중)
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)
  
* 질병 분류 작업(예정)
---

## [회의록 모음](https://github.com/Hongiee2/Greatest-Classify-Group/issues/2)
