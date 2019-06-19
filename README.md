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

* STFT Filter - [Code]()
  * 데이터에서 시간에 대해 구간을 짧게 나누어 나누어진 여러 구간의 데이터를 각각 Fourier Transform 하는 방법 - 참고자료
<div>
https://user-images.githubusercontent.com/46617803/59761766-9f997d00-92d0-11e9-872c-c91f694e1bd4.png
https://user-images.githubusercontent.com/46617803/59761767-a0caaa00-92d0-11e9-9f3c-730bda576c2b.png
https://user-images.githubusercontent.com/46617803/59761770-a2946d80-92d0-11e9-9715-bcc53344c31f.png
</div>
  
* MFCC Filter - [Code]()
  * MFCC 설명 - 참고자료
  * 이미지

* MFCC Delta Filter - [Code]()
  * MFCC Delta 설명 - 참고자료
  * 이미지

* 이미지 생성 모듈화 - [Code]() //오류안난 것으로 업로드 할 것
  * CNN 학습에 필요한 이미지를 자동 생성하도록 모듈화 시켰습니다. 이미지를 생성하기 위해 Wav 폴더에 음원 파일을 넣고 해당 폴더에 대한 정보를 담은 CSV 파일이 요구됩니다. 이후 원하는 필터를 선택해 코드를 실행하면 자동으로 Train, Test 폴더에 설정한 비율에 따라 라벨링된 이미지들이 생성됩니다.

---

## 학습과정
  
* CNN 학습 진행
  * 187개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * 필터거친 이미지

---

## 검증

* 질분 유무 판단(진행중)
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)
  * 결과 정리후 관련 결과물들 링크처리
---

## 추가작업

* 다중 필터 사용, SVM 학습 진행
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * 정확도 향상을 위한 다양한 시도와 성능 비교(잡음 섞인 데이터셋 추가, 다른 Filter 사용, SVM 학습법 적용 (진행중)


---
## [회의록 모음](https://github.com/Hongiee2/Greatest-Classify-Group/issues/2)
