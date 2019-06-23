# Greatest-Classify-Group
Respiratory Diseases Classification Using Audio Data  

## 프로젝트 목표

[ICBHI 2017 Challenge Respiratory Sound Database](https://www.kaggle.com/vbookshelf/respiratory-sound-database)을 이용한 질병 분류 작업을 진행하였습니다.  

정확도 향상을 목표로 기존 작업물([Base논문](https://eden.dei.uc.pt/~ruipedro/publications/Conferences/ICBHI2017a.pdf), [관련작업](https://www.kaggle.com/eatmygoose/cnn-detection-of-wheezes-and-crackles))과 다른 전처리 과정 및 학습법을 적용하였습니다. 

---

## 목차 
- [데이터 전처리](#데이터-전처리)
- [CNN 학습과정](#학습과정)
- [검증](#검증)
- [SVM 학습과정](#SVM-학습과정)
- [결론](#결론)

---

##  결과물
- [텀프로젝트 논문]()
- [텀프로젝트 발표 PPT]()

---

## [회의일지](https://github.com/Hongiee2/Greatest-Classify-Group/issues/2)

---

## 데이터 전처리

* STFT(Short-Time Fourier Transform) Filter - [[참고자료](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/07.%20Reference/STFT.pdf)]
  * FT(Fourier Transform)을 실제 녹음된(유한의) 소리에 적용하기 위해 만든 것.  
  * 데이터에서 시간에 대해 구간을 짧게 나누어 나누어진 여러 구간의 데이터를 각각 FT(Fourier Transform)적용.  
        
        * FT(Fourier Transform)이란?  
         * 'frequency domain의 함수','하나의 함수를 다른 함수로 변환하는 과정'.  
         * 연속시간의 아날로그 파형을 infinite Fourier series(무한 퓨리에 급수)의 합으로 만드는 것.  
         * infinite Fourier series는 특정 amp와 phase를 가지는 사인파로 변환.  
         * 입력 신호를 대응하는 스펙트럼으로 전환.         
    <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59903930-3daa5600-943d-11e9-9c27-95d74c0c76a6.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59903938-41d67380-943d-11e9-9e1a-ae1d24b6c517.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59903942-4438cd80-943d-11e9-8e7d-5605e47e6d9b.png">
</p>
</div>
      

  
* MFCC Filter(Mel-Frequency-Cepstral Coefficent)
  * 입력된 소리 전체를 대상으로 하는 것이 아닌 일정 구간(Short time)으로 나누어, 이 구간에 대한 스펙트럼을 분석하여 특징을 추출. 
  * 특징을 추출하기 위해 6단계로 진행됨 - [[단계별 특징추출](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/03.%20Feature%20extraction/MFCC%EB%9E%80.docx)]
  * 6단계가 끝나면 12~13개의 Coefficient들을 얻을 수 있음.(Mel Frequency Cepstral Coefficient)
  <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59901168-5a429000-9435-11e9-89d0-a79eceae5142.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59901172-5ca4ea00-9435-11e9-8e23-14204306c857.png">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59901173-5dd61700-9435-11e9-9dcf-9f6076ab2a9c.png">
</p>
</div>

* MFCC Delta Filter
  * MFCC필터에서 성능을 올리기 위한 추가적인 방법.(약 20%정도의 성능이 올라감.)
  * MFCC 적용 후 미분을 함으로써 음성의 변화율을 스팩트럼 이미지를 생성.  
  <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/48382704/59901019-ee602780-9434-11e9-8dfd-437d2bfa47c0.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59761766-9f997d00-92d0-11e9-872c-c91f694e1bd4.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59761767-a0caaa00-92d0-11e9-9f3c-730bda576c2b.png">
</p>
</div>
  

* 이미지 생성 모듈화 - [[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/03.%20Feature%20extraction/Respiratory_Sound_Feature_save.ipynb)] 
  * CNN 학습에 필요한 이미지를 자동 생성하도록 모듈화.  
  * 이미지를 자동 생성하기 위해선 Wav 폴더에 해당 음원 파일을 넣고 파일에 대한 정보를 CSV 파일에 갱신해주는 작업이 필요.  
  * 이후 원하는 필터를 선택해 코드를 실행하면 설정된 비율에 따라 자동으로 Train, Test 폴더에 라벨링된 이미지들이 생성.

---

## 학습과정
  
* CNN 학습 진행

<p align="center">
<img  src="https://user-images.githubusercontent.com/46617803/59962802-352d4a80-9525-11e9-80a2-0f3c2e4c734e.png">
</p>
    
   * 질병 유무(8:2 학습비율/3 layers)
   
         * MFCC  
           * 186개 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 72% 정확도(56X56)
           * 920개 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 97% 정확도(112X112)  
           * 920개 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 96% 정확도(56X56)          
           * 920개 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 97% 정확도(14X14)           
         * MFCC+delta  
           * 920개 음성파일 MFCC+delta를 통한 Feature 추출 후 CNN 학습 진행 > 약 97% 정확도(112X112)      
           * 920개 음성파일 MFCC+delta를 통한 Feature 추출 후 CNN 학습 진행 > 약 96% 정확도(56X56)   
           * 920개 음성파일 MFCC+delta를 통한 Feature 추출 후 CNN 학습 진행 > 약 96% 정확도(14X14)  
	   
   * 질병 분류(8:2 학습비율/3 layers)
        
         * MFCC    
           * 920개 음성파일 MFCC를 통한 Feature 추출 후 CNN 학습 진행 > 약 86% 정확도(112X112)    
         * MFCC+delta  
           * 920개 음성파일 MFCC+delta를 통한 Feature 추출 후 CNN 학습 진행 > 약 88% 정확도(112X112)   
	 * 질병분류 정확도가 낮아 추가적인 검증을 실행 
	 
---

## 검증

*  train_test rate,Learning_rate, batch_size,train, test set 변경해가며 정확도 측정 
  (batch_size = 100 ,learning_rate = 0.001 , layers =3)

  * train_test rate 변경(mfcc,mfcc_delta 평균)
	* 9:1(질병 여부 분류)
	
	  |Size    |정확도 |   Code  |
 	  |:------:|:----:|:--------:|
	  | 112X112| 84.9%|[[Code]()]|
	  |   56X56|85.45%|[[Code]()]|
 	  |   28X28|85.95%|[[Code]()]|
 	  |   14X14| 86.3%|[[Code]()]|
	  
	
	* 7:3(질병 여부 분류)  
	 
	  |Size    |정확도 |   Code  |
 	  |:------:|:----:|:--------:|
	  | 112X112| 84.2%|[[Code]()]|
	  |   56X56| 89.1%|[[Code]()]|
 	  |   28X28|   82%|[[Code]()]|
 	  |   14X14| 83.75%|[[Code]()]|

 * size가 56X56 training_rate 8:2가 가장 높아 비교값으로 사용
 * Learning_rate 변경(mfcc,mfcc_delta 평균)
 learning_rate = 0.01,0.001,0.0001

 |Size | 0.01|0.001| 0.0001|  avg|
 |:---:|:---:|:---:|:-----:|:---:|
 |56X56|85.5%|85.5%|  85.5%|85.5%|
  
 * batch_size 변경(mfcc,mfcc_delta 평균)
 batch_size = 100,200,300,400

 |Size | 100| 200| 300| 400|   avg|
 |:---:|:--:|:--:|:--:|:---:|:---:|
 |56X56|84.6%|84.6%|84.6%|84.6%|84.6%|
 * train, test set 변경(mfcc,mfcc_delta 평균)
 
 |Size | case1| case2| case3|  avg|
 |:---:|:----:|:----:|:----:|:---:|
 |56X56|   88%| 85.8%|86.35%|84.6%|
 
 * 결과
   * CNN는 데이터 갯수가 많을수록 높은 정확도를 보임.
   * 데이터 셋확보가 매우 중요함
---

## SVM 학습과정

<p align="center">
<img width="300" src="https://user-images.githubusercontent.com/46617803/59861489-dd270480-93bb-11e9-844b-008ea200264f.png">
</p>

* SVM(Support Vector Machine) 학습 진행 

  * 지도 학습 방식의 대표 분류 기법인 SVM을 사용. 
  * SVM은 데이터를 벡터공간으로 표현 후 서포터 벡터(각 분류의 경계선에 가장 가까이 있는 벡터)간의 거리를 최대화하여 데이터들을 분류.

  * SVM은 선형분류와 더불어 비선형 분류에도 사용.  
  * 비선형 분류를 하기 위해 주어진 데이터를 고차원 특징 공간으로 만드는 작업이 필요.     
  * SVM은 kernel function이란 개념을 도입하여 특징 공간을 접어버리거나 꼬아버려 비선형 데이터를 선형으로 분류할 수 있게 함.  

        * 장점 : SVM는 차원 수 > 데이터 수 인 경우 효과적.   
        * 단점 : 데이터가 너무 많으면 속도가 느리고 메모리 소모가 커짐.

* 피쳐추출 
  * librosa.feature를 통해 음원의 특징을 추출하였습니다. 특징 추출에는 MFCC, zero_crossing_rate, spectral_rolloff, spectral_centriod, spectral_contrast, spectral_bandwidth 등 총 6가지 방식이 사용되었습니다. 각 음원파일마다 6가지 특징을 뽑아내어 정규화시킨 후 [numpy.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html)을 이용해 특징을 정렬하여 [pd.Series](https://magnking.blog.me/221333137412)로 묶어 반환하도록 하였습니다.
  * Zero crossing rate는 음성 신호의 smoothness를 측정하여 신호를 구별하는 방법입니다. 소리에 따라 파형의 형태가 다르기 때문에 smoothness를 측정하여 소리를 구분할 수 있습니다. Spectral roll off는 스펙트럼 magnitude 분포의 80%가 집중되어 있는 주파수를 나타냅니다. 스펙트럼의 형태와 낮은 주파수 영역에 신호의 에너지가 얼마나 집중되어 있는지 보여줍니다. Spectral Centroid는 STFT의 magnitude 스펙트럼의 중심을 뜻합니다. Centroid는 스펙트럼의 형태를 측정하는 방법 중의 하나입니다. Spectral Contrast 매 프레임마다 6개의 구역으로 나누어 스펙트럼의 peak, valley의 차이점을 계산합니다. Spectral bandwidth는 주파수의 대역폭을 측정하는 방법 중 하나입니다.
  

<div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902749-0c7c5680-943a-11e9-89bb-b72fdeb1b6dd.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902751-0dad8380-943a-11e9-830d-6c0202ba04a0.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902753-0f774700-943a-11e9-95c8-0fa5b477efd3.png">
</p>
</div>

  <div>
<p align="center">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902755-10a87400-943a-11e9-9091-022c28450acf.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902756-11d9a100-943a-11e9-9bf4-364316a203b0.png">
<img width="250" src="https://user-images.githubusercontent.com/46617803/59902761-12723780-943a-11e9-8ef0-8da20c70d673.png">
</p>
</div>


* 차원축소
  * 특징이 많으면 기계학습 모델이 잘 훈련되지 않거나 과적합을 일으키고, 훈련된 모델을 해석하여 용이한 정보를 얻기 힘듬. 또한 고차원 데이터는 시각화가 불가능하기에 분석결과를 공유하는 것 또한 쉽지 않음.
  * 차원을 줄이는 방법은 크게 특징 선택(feature selection)과 특징 추출(feature extraction)으로 구분. 특징 선택은 전문가 지식이나 데이터 밖의 데이터를 이용하여 일부를 골라내는 작업이고, 특징 추출은 주어진 특징들을 조합하여 새로운 특징값을 계산하는 작업.
  * 특징 추출의 보편적인 방법 주성분분석(PCA) 방법을 사용
  
* RBF 커널 사용
  * 선형에서 구분하지 못하는 구조를 rbf kernel을 사용해 데이터 변환
  * Over fitting을 막기 위해서는 Cost, Gamma 값들의 조절이 필요. 
  * C의 값이 작을때는 제약이 큰 모델을 만들고 각 포인트의 영향력이 작음, 반대로 C값이 증가할수록 각 포인트들이 모델에 큰 영향을 주며 결정 경계를 휘어서 정확하게 분류.
  * Gamma 값이 커짐에 따라 결정 경계는 하나의 포인트에 더 민감해진다.
  * C 값은 kernel을 rbf로 사용하게 될 경우 training data set의 수를 줄여 먼저 C parameter 값을 최적화. 그 후 원래 training data set을 가지고 fitting 시켜 최적화된 SVM의 결과를 획득


* 정확도 측정
  * 85개의 데이터셋을 기준으로 Train data 개수를 100개, 200개, 400개로 바꿔가며 정확도를 측정하였습니다. 각 실행마다 C, G값은 가장 높은 정확도를 나타내는 값으로 설정하여 진행하였습니다.  
  
  
* 질병 유무 판단 결과  

Train Data 수 | 정확도 | Code 
:---:|:---:|:---:|
100 개|0.95%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_100_85_95.ipynb)]
200 개|0.95%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_200_095.ipynb)]
300 개|0.98%|[[Code]()]
400 개|0.97%|[[Code]()]


* 질병 분류 작업 결과  

 Train Data 수 | 정확도| Code 
 :---:|:---:|:---:|
 100 개|0.85%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_multi%20100_85_85.ipynb)]
 200 개|0.95%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_multi%20200_85_95.ipynb)]
 300 개|0.95%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_multi_300_85_95.ipynb)]
 400 개|0.93%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_multi_400_85_93..ipynb)]
 500 개|0.89%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_multi_89%25.ipynb)] 
 600 개|0.93%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_using mfcc_multi_600_85_93..ipynb)] 
 700 개|0.87%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_using mfcc_multi_700_87..ipynb)] 
 800 개|0.87%|[[Code](https://github.com/Hongiee2/Greatest-Classify-Group/blob/master/Project/05.%20Validation/SVM_using_mfcc_multi_800_87..ipynb)]  
 
* 결과
  * SVM의 질병 유무 판단의 결과를 보았을 때 약간의 오차는 있지만 일반적으로 train data의 개수가 올라갈수록 더 높은 정확도를 보이는 것으로 나타났다. 하지만 질병 분류 작업의 경우 train data의 개수가 올라가도 정확도가 감소하는 등 변화가 나타난다. 이는 train data와 test data의 비율을 8:2로 잡았는데 라벨링되어 있는 데이터들이 균일하지 않게 분포하다보니 test 데이터를 어떻게 잡느냐에 따라 정확도에 영향을 미친 것으로 추측된다. 따라서 800개를 train한 모델이 가장 신뢰도 높은 모델이라 할 수 있으며 이러한 문제는 총 데이터가 많으면 해결될 것이라고 판단한다.
---

## 결론
  * 일률적인 학습법의 비교는 어렵지만 질병 유무 판단의 경우 CNN에서 920개 데이터를 학습했을 때 기준 96%의 정확도를, SVM에서 100개 학습기준 95%의 정확도를 보였다. 또한 질병 분류 판단의 문제에서는 CNN이 920개의 데이터를 학습했을 때 85%의 정확도를, SVM에서는 100개 기준 85%, 200개 기준 200% 등의 결과를 얻었다. 하지만 모델이 갖는 신뢰도를 따져보았을 때 SVM은 질병분류 문제에서 800개 학습 시 도출한 87% 정확도를 갖는다고 말하는 것이 적절하며 이는 CNN 대비 2% 정도 더 나은 결과를 도출했다고 말할 수 있다.
