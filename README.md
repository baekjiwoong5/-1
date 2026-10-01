# 키·몸무게 회귀분석 (Height–Weight Regression)

키(cm) 하나로 몸무게(kg)를 예측하는 회귀 모델 네 가지를 만들고 비교합니다.
데이터처리개론 수업의 3강(선형회귀) 내용을 바탕으로, 선형회귀를 비선형회귀까지 넓혀 봤습니다.

**[온라인 계산기 열기](https://baekjiwoong5.github.io/height-weight-regression/)** ← 키를 넣으면 네 모델의 예측을 한 번에 보여줍니다.

![회귀곡선](images/fit.png)

## 모델

| 모델 | 식 | 종류 | 푸는 방법 |
|---|---|---|---|
| 선형회귀 | y = b₀ + b₁·x | 선형 | 최소제곱법 공식 (경사하강법으로 검증) |
| 2차 다항회귀 | y = c₀ + c₁·x + c₂·x² | 계수에 대해 선형 | `np.polyfit` |
| 거듭제곱 | y = a · xᵇ | **비선형** | `scipy.optimize.curve_fit` |
| 지수 | y = a · e^(b·x) | **비선형** | `scipy.optimize.curve_fit` |

2차 다항식은 그래프가 곡선이지만 계수 c₀, c₁, c₂에 대해서는 선형이라 선형회귀로 풀립니다.
거듭제곱·지수 모델은 계수가 지수 자리에 있어 공식으로 바로 풀 수 없고, 반복 계산으로 오차제곱합을 최소화합니다.
반복 계산의 초기값은 양변에 log를 취한 직선의 계수로 정했습니다.

## 결과

데이터 199명, 키 148~197cm.

| 모델 | 식 | R² | RMSE (kg) |
|---|---|---:|---:|
| 선형 | y = −130.7470 + 1.1492·x | 0.5940 | 8.481 |
| 2차 다항 | y = 105.3749 − 1.6114·x + 0.008047·x² | 0.5977 | 8.442 |
| 비선형(거듭제곱) | y = 1.3779×10⁻⁵ · x^2.9894 | 0.5973 | 8.446 |
| 비선형(지수) | y = 3.4208 · e^(0.01722·x) | 0.5961 | 8.459 |

- 키가 1cm 늘면 몸무게는 평균 약 1.15kg 늘어납니다 (선형회귀 기울기).
- 거듭제곱 모델의 지수가 약 3으로, 몸무게가 키의 세제곱에 비례한다는 결과입니다. 몸을 부피로 보면 자연스럽습니다.
- 네 모델의 R² 차이는 0.004 이내입니다. 이 데이터에서 키와 몸무게의 관계는 거의 직선이고, 몸무게 변동의 약 40%는 키 이외의 요인에서 나옵니다.

### 잔차분석

![잔차분석](images/residuals.png)

## 실행 방법

```bash
git clone https://github.com/baekjiwoong5/height-weight-regression.git
cd height-weight-regression
pip install -r requirements.txt

python regression.py                # 학습, 결과 출력, images/에 그래프 저장
python regression.py --height 175   # 키 175cm의 예측 몸무게
python regression.py --show         # 그래프 창 띄우기
```

`--height 175` 실행 결과

```
키 175 cm의 예측 몸무게
  선형             70.4 kg
  2차 다항          69.8 kg
  비선형(거듭제곱)      69.9 kg
  비선형(지수)        69.6 kg
```

## 폴더 구조

```
height-weight-regression/
├── regression.py          # 모델 학습·평가·예측·그래프
├── requirements.txt
├── data/
│   ├── height_weight_only.xlsx
│   └── height_weight_only.csv
├── images/                # regression.py가 생성하는 그래프
│   ├── fit.png
│   └── residuals.png
├── index.html             # 웹 계산기 (GitHub Pages)
└── .nojekyll
```

## 웹 계산기 켜기 (GitHub Pages)

1. 저장소가 **Public**인지 확인 (무료 계정은 Private 저장소에서 Pages를 쓸 수 없음)
2. **Settings → Pages**로 이동
3. Source: **Deploy from a branch**, Branch: **main**, 폴더: **/ (root)** 선택 후 Save
4. 1~2분 뒤 https://baekjiwoong5.github.io/height-weight-regression/ 에서 열립니다
   (진행 상황은 저장소의 **Actions** 탭에서 확인)

## 주의

학습 데이터 범위(148~197cm) 밖의 키는 예측을 믿기 어렵습니다. 특히 지수 모델은 범위를 벗어나면 값이 빠르게 커집니다.
