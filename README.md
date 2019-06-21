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

---

##  결과물
- [텀프로젝트 논문]()
- [텀프로젝트 발표 PPT]()

---

## [회의일지](https://github.com/Hongiee2/Greatest-Classify-Group/issues/2)

---

## 데이터 전처리

* STFT(Short-Time Fourier Transform) Filter
  *Fourier Transform이란?
  * 데이터에서 시간에 대해 구간을 짧게 나누어 나누어진 여러 구간의 데이터를 각각 Fourier Transform 하는 방법 - 참고자료
      

  
* MFCC Filter
  * 입력된 소리 전체를 대상으로 하는 것이 아닌 일정 구간(Short time)으로 나누어, 이 구간에 대한 스펙트럼을 분석하여 특징을 추출. 
  * 특징을 추출하기 위해 단계별로 진행됨[[참고자료](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/03.%20Feature%20extraction/MFCC%EB%9E%80.docx)]
  <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59762719-db354680-92d2-11e9-975e-c0e3d6c8ccba.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59762731-dec8cd80-92d2-11e9-9b9a-c855c969862b.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59762755-e8eacc00-92d2-11e9-9f5e-6c1a80cf71c5.png">
</p>
</div>

* MFCC Delta Filter
  * MFCC Delta 설명 - 참고자료
  <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59761766-9f997d00-92d0-11e9-872c-c91f694e1bd4.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59761767-a0caaa00-92d0-11e9-9f3c-730bda576c2b.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59761767-a0caaa00-92d0-11e9-9f3c-730bda576c2b.png">
</p>
</div>
  

* 이미지 생성 모듈화 - [[Code]()] //오류안난 것으로 업로드 할 것
  * CNN 학습에 필요한 이미지를 자동 생성하도록 모듈화 시켰습니다. 이미지를 자동 생성하기 위해선 Wav 폴더에 해당 음원 파일을 넣고 파일에 대한 정보를 CSV 파일에 갱신해주는 작업이 필요합니다. 이후 원하는 필터를 선택해 코드를 실행하면 설정된 비율에 따라 자동으로 Train, Test 폴더에 라벨링된 이미지들이 생성됩니다.

---

## 학습과정
  
* CNN 학습 진행 - [[Code]()]
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도
  * layer 개수, Learning_rate, batch_size, train_test rate 변경해가며 정확도 측정 (예정)
* 질병 유무 판단 - [[Code]()]
* 질병 분류 작업 - [[Code]()]
  * 학습모델 이미지

---

## 검증

* Train Data와 Test Data 8:2 비율로 질분 유무, 분류 작업 진행
  * 920개 호흡 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도

* 질병 유무 판단 결과 
  * 128X128 사이즈, 64X64 사이즈
* 질병 분류 작업 결과
  * 128X128 사이즈, 64X64 사이즈

* 결과 정리후 관련 결과물들 링크처리

---

## 추가작업

<p align="center">
<img width="300" src="https://user-images.githubusercontent.com/46617803/59861489-dd270480-93bb-11e9-844b-008ea200264f.png">
</p>

* SVM(Support Vector Machine) 학습 진행 - [[Code]()]

  * 지도 학습 방식의 대표 분류 기법인 SVM을 사용했습니다. SVM은 데이터를 벡터공간으로 표현 후 서포터 벡터(각 분류의 경계선에 가장 가까이 있는 벡터)간의 거리를 최대화하여 데이터들을 분류합니다.

  * SVM은 선형분류와 더불어 비선형 분류에도 사용됩니다. 비선형 분류를 하기 위해 주어진 데이터를 고차원 특징 공간으로 만드는 작업이 필요합니다. SVM은 kernel function이란 개념을 도입하여 특징 공간을 접어버리거나 꼬아버려 비선형 데이터를 선형으로 분류할 수 있게 만듭니다.

  * SVM는 차원 수 > 데이터 수 일대 효과적인 장점이 있습니다. 단점으로는 데이터가 너무 많으면 속도가 느리고 메모리 소모가 크다는 점이 있습니다.

<p align="center">
<img width="400" src="https://user-images.githubusercontent.com/46617803/59865181-655cd800-93c3-11e9-969e-e856405d6130.png">
</p>

<p align="center">
사용한 커널에 따라 다른 정확도를 보인다.
</p>

* 피쳐추출 - 설명
  * 다중 필터 적용 - [[Code]()]

* 차원축소 - 이유
  * 

* C, G 값 설정 - 우리꺼 그래프 가져오기, rbf 이유
  * SVM에서 Over fitting을 막기 위해서는 Parameter인 C, gamma, kernel paramter를 잘 조절해야 합니다. C 값은 
  * kernel을 rbf로 사용하게 될 경우 training data set의 수를 줄여 먼저 C parameter 값을 최적화 시키고 그 다음 원래 training data set을 가지고 fitting 시키면 최적화된 SVM을 얻습니다.

<div>
<p align="center">
<img width="400" src="https://user-images.githubusercontent.com/46617803/59864925-c6d07700-93c2-11e9-90a2-bbe85cb5427b.png">
<img width="400" src="https://user-images.githubusercontent.com/46617803/59864929-c801a400-93c2-11e9-917b-cfd1e7710b8a.png">
</p>
</div>

* 질병 유무 판단 결과 - [[Code]()]

* 질병 분류 작업 결과 - [[Code]()]

* 결론
  * CNN 학습대비 더 적은 데이터로도 높은 분류 정확도를 보였습니다. 하지만 데이터가 많아질 경우 ~~단점을 보임.
---

  * [자료정리중](https://blog.naver.com/slykid/221183951057)[자료정리중2](https://blog.naver.com/genesis717/220657502933)
  * [자료정리중](http://www.itdaily.kr/news/articleView.html?idxno=69618)[자료정리중](https://m.blog.naver.com/sbd38/221371546928)
