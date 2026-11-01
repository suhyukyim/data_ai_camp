# 2주차

## 학습 내용

1주차에 구축한 데이터셋과 리서치 결과(YOLO26 채택)를 바탕으로, `02_yolo_모델_학습_코드.ipynb`를 실제로 완성함. 기존에 공유받은 노트북 템플릿이 옛날 버전인 yolov5(git clone + 커맨드라인 스크립트 실행) 방식으로 짜여있어서, 이를 YOLO26(ultralytics 패키지 기반 파이썬 API) 방식으로 전면 수정하는 작업을 진행함 ([code/02_yolo_모델_학습_코드.ipynb](../code/02_yolo_모델_학습_코드.ipynb) 전 섹션 완료). 이후 Colab에서 실제로 학습을 실행하며 발생한 이슈들(라이브러리 import 순서, 구글 드라이브 I/O 속도)도 함께 대응함.

## 진행한 작업

### 1. yolov5 → YOLO26(ultralytics) 전환이 필요한 이유 파악

기존 템플릿은 `git clone`으로 yolov5 레포 전체를 받아와서 그 안의 `train.py`/`val.py` 스크립트를 커맨드라인으로 실행하는 방식이었음. 반면 YOLO26은 `ultralytics`라는 pip 패키지로 제공되고, 파이썬 코드 안에서 `YOLO` 객체를 만들어 메서드(`.train()`, `.val()`, `.predict()`)를 호출하는 방식임. `!yolo train model=yolo26n.pt ...`처럼 파일 경로 없이 실행되는 명령어라는 점에서 pip install 기반이라는 것을 유추함.

### 2. 설치 & 환경설정 섹션

```python
cd /content/drive/MyDrive/comento/1   # 데이터셋 위치로 이동 (git clone 목적이 아니라, 이후 상대경로 사용 편의 + 결과물을 드라이브에 보존하기 위함)
!pip install ultralytics               # ultralytics(YOLO26) 패키지 설치
```

- `pip install`은 파이썬 환경 전체에 패키지를 설치하는 것이라 `cd` 위치와 무관하게 동작하고, `cd`는 이후 학습 데이터 경로/결과 저장 경로를 짧게 쓰기 위한 것이라는 걸 정리함 — 두 코드가 순서상 붙어있지만 서로 독립적인 준비 작업이라는 점을 확인
- 기존 yolov5 관련 코드(git clone, Pillow 업그레이드 등)를 전부 제거

### 3. 학습(train) 코드 작성

```python
model = YOLO('yolo26n.pt')
train_results = model.train(data='datasets_yolo/dataset.yaml', epochs=10, imgsz=224, batch=20, name='train_result', project='train_results')
```

- `YOLO(...)` 자리엔 "어떤 모델(가중치)을 쓸지", `.train(data=...)` 자리엔 "어떤 데이터로 학습시킬지"가 들어간다는 역할 구분을 명확히 함 (`YOLO('dataset.yaml')`처럼 반대로 넣는 실수를 몇 번 하면서 확실히 이해)
- 모델 스케일 표기(`yolo26n.pt`)가 yolov5와 달리 버전에 `v`가 붙지 않는다는 것을 확인
- `epochs=10, imgsz=224, batch=20`은 공식 튜토리얼 예제값(coco8 데모용, `epochs=100, imgsz=640`)이 아니라, 기존 yolov5 템플릿에서 이전 멘토님이 우리 데이터셋 크기와 Colab 무료 GPU 환경에 맞춰 이미 잡아둔 값을 그대로 가져다 씀
- `project=`(상위 폴더)/`name=`(하위 폴더)로 결과 저장 위치 지정, `train_results`라는 변수명을 써서 이후 `val_results`와 값이 덮어씌워지지 않게 함
- `timeit`으로 학습 소요 시간을 측정하는 코드 유지 (Colab 무료 GPU 시간 제한을 고려해 학습 시간을 기록해두기 위함)

### 4. 평가(val) 코드 작성

```python
val_results = model.val(data='datasets_yolo/dataset.yaml', imgsz=224, batch=60, name='val_result', project='val_results')
```

- `train()`/`val()` 메서드 이름 패턴에서 "평가하다"에 해당하는 메서드명을 유추
- val은 역전파(그래디언트 계산)가 없어 메모리 여유가 있으므로, train보다 큰 batch(60)를 쓸 수 있다는 것을 이해

### 5. 추론(inference) 코드 작성

```python
trained_model = YOLO('train_results/train_result/weights/best.pt')

test_img_path = 'datasets_yolo/images/test/syn_01921.png'
img = Image.open(test_img_path)
infer_results = trained_model.predict(source=img, save=True, conf=0.25, imgsz=224, name='inference_result', project='inference_results')

result = infer_results[0]
print(len(result))          # 탐지된 차량 개수
result.show()
plt.imshow(result.plot())
plt.axis('off')
plt.show()
```

- 학습 결과 가중치 파일이 `project/name/weights/best.pt` 경로에 저장된다는 것을, 학습 코드의 `project=`/`name=` 설정값으로부터 역산해서 찾음
- `predict()`의 반환값은 리스트이고, 리스트 안의 `Results` 객체가 `.show()`(외부 이미지 뷰어, Colab에선 잘 안 보일 수 있음), `.plot()`(matplotlib용 numpy 배열 반환), `.boxes`(탐지 정보) 등을 갖고 있다는 것을 확인
- yolov5식 `.pandas().xyxy[0]`, `.render()` 문법은 ultralytics(YOLO26)의 `Results` 객체엔 없다는 것을 확인 — 라이브러리(yolov5 vs ultralytics)마다 반환 객체 구조가 다르다는 것을 체감
- `len(result)`로 탐지된 차량 개수를 바로 셀 수 있다는 것 확인 (`Results` 객체가 `__len__`을 구현)

### 6. 실제 Colab 실행 중 발견한 이슈

- 라이브러리 import 셀에 `from ultralytics import YOLO`를 넣었는데, `!pip install ultralytics`가 그보다 뒤 셀에 있어서 `ModuleNotFoundError` 발생 → 노트북은 위에서 아래로 순서대로 실행되므로, import보다 install이 먼저 와야 한다는 실행 순서 문제를 확인하고 위치를 바로잡음
- 학습 시작 시 `WARNING: Slow image access detected` + 라벨 스캔 단계에서만 30분 이상 소요 예상되는 문제 발견 → 원인은 구글 드라이브(네트워크 마운트 저장소)의 느린 I/O(읽기 속도 3.2MB/s 수준)
- 대응 방안 두 가지 확인: ① `model.train(..., fraction=0.2)`처럼 학습 데이터 중 일부만 랜덤으로 사용해서 처리량 자체를 줄이는 방법 ② 데이터를 Colab 로컬 디스크(`/content/`)로 복사해서 읽기 속도 자체를 개선하는 방법(세션 종료 시 사라지는 단점) — 우선 ①번(`fraction` 축소)으로 진행하기로 함

## 새로 배운 개념 / 궁금해서 짚어본 것들

- **pip install vs git clone**: 전자는 파이썬 환경 전체에 패키지를 설치(현재 위치와 무관하게 어디서든 import 가능), 후자는 특정 폴더에 레포 소스 전체를 복사. `yolo`처럼 파일 경로 없이 실행되는 CLI 명령어는 pip install로 등록된 것이라는 신호로 볼 수 있음
- **YOLO(모델) vs .train(data=데이터)**: 괄호 안에 뭘 넣느냐가 완전히 다른 역할 — "어떤 뇌를 쓸지"와 "그 뇌를 뭘로 학습시킬지"는 구분해서 생각해야 함
- **epoch / batch / imgsz**: epoch=전체 데이터를 몇 바퀴 돌릴지, batch=한 번에 몇 장씩 묶어서 가중치를 업데이트할지, imgsz=모델에 입력하는 이미지 리사이즈 크기. 공식 튜토리얼의 예제값은 데모용 데이터셋 기준이라 우리 프로젝트 상황(데이터 크기, GPU 시간 제한)에 맞게 별도로 판단해야 함
- **train/val/test와 evaluation/inference는 다른 축의 개념**: train/val/test는 "데이터를 용도별로 나눈 것"이고, evaluation(`.val()`)과 inference(`.predict()`)는 "모델에게 시키는 행동"임. `predict`라는 함수 이름은 어떤 데이터(train/val/test 무엇이든)를 넣었는지와 무관하게, "모델이 새 입력에 대해 결과를 뽑아내는 행동" 자체를 가리키는 이름
- **평가(val) vs 추론(inference)의 목적 차이**: 평가는 정답 라벨과 비교해서 점수(mAP)를 매기는 것이고, 추론은 정답 유무와 상관없이 모델이 실제로 뭘 예측하는지 결과 자체를 확인하는 것. 3주차에 만들 FastAPI 서비스가 바로 이 "추론"을 실제 서비스로 감싸는 작업이라는 걸 이해함
- **프로젝트 전체 그림과 지금 작업의 연결**: 업무요청서 1~4차 구조상, 1차(데이터+모델조사) → 2차(baseline 모델 학습, 지금 여기) → 3차(FastAPI로 차량 개수 세는 서비스화) → 4차(전체 과정 보고)로 이어짐. 2주차 자체는 "교통 정체 해소"가 아니라 "차를 정확히 찾아내는 탐지 엔진"을 완성하는 단계이고, 이 엔진이 3주차에서 차량 카운팅 API로 연결되면서 비로소 "정체 파악"이라는 실제 서비스 가치로 이어진다는 걸 정리함
- **API 문서/소스 직접 검증하는 습관**: `docs.ultralytics.com`의 Modes(Train/Val/Predict) 페이지, 또는 GitHub `ultralytics/engine/model.py` 소스코드에서 실제 메서드가 어떻게 정의되어 있는지 직접 확인하는 방법을 활용함

### 7. 학습 데이터 분포 사전 분석 (01번 노트북 Section 8)

학습을 시작하기 전에, 데이터 자체의 특성을 먼저 파악하는 작업을 진행함 (2주차 과제피드백의 "학습 전 데이터 특성 파악" 항목 반영).

- 이미지당 객체(차량) 수 분포
- bbox 크기/위치 분포 (화면 어디에 차가 몰려있는지, 크기가 균일한지)
- `calculate_iou`로 같은 이미지 내 차량 간 겹침 정도 계산
- train 이미지 16장 랜덤 샘플로 조명/배경/시점 다양성 육안 확인

이 분석 결과(각 이미지에 차량 위치가 특정 패턴으로 몰려있는 경향)가 이후 "새 이미지 일반화 테스트"에서 실제로 문제가 된 지점과 연결됨.

### 8. Baseline 학습 결과 + 학습 곡선 확인

`epochs=10, imgsz=224, batch=20`으로 baseline을 학습한 결과: **mAP50=0.977, mAP50-95=0.94, Precision=0.939, Recall=0.901**.

`results.png`/`results.csv`로 epoch별 train/val loss와 mAP 추이를 확인 — val loss가 다시 오르는 구간 없이 잘 수렴하는 것을 확인 (과적합 아님).

### 9. TP / FN / FP 시각적 확인 (IoU 기반 자동 판정)

- 01번 노트북의 `draw_boxes_on_image`를 참고해, 정답(파랑)과 예측(빨강) 박스를 한 이미지에 같이 그리는 `draw_gt_vs_pred` 함수를 새로 작성
- `calculate_iou` + `match_gt_pred`(그리디 매칭, IoU≥0.5 기준 — mAP50과 동일한 기준)로, 눈으로만 보는 게 아니라 **TP/FP/FN 개수를 자동으로 집계**하도록 구현
- test 이미지 중 frac 비율만큼 샘플링해서 baseline 성능 재확인: **TP=99, FP=7, FN=4 (Precision=0.934, Recall=0.961)**

### 10. 실패 케이스 원인 분석 — FN과 FP는 서로 다른 원인이었음

FP/FN이 있었던 이미지 9장의 GT/Pred 박스 좌표를 직접 출력해서 원인을 구분함.

- **FP(오탐) 7개 중 6개**: 좌표가 거의 동일한(IoU≈0.99) 박스 쌍이 반복 발견됨 → **같은 차를 중복 예측**한 것. ultralytics의 NMS(`iou` 파라미터)가 이 모델(YOLO26)에는 극단값(0.1)을 줘도 전혀 영향이 없다는 것도 별도로 확인 — NMS-free 계열 구조로 추정됨.
- **FN(놓침) 나머지**: 해당 위치에 예측 박스 자체가 없음(확신도 낮아서 걸러진 게 아니라 후보 자체가 없음). 놓친 박스 크기도 다른 박스들과 비슷해서 "작아서 놓쳤다"는 설명은 안 맞음 → **학습 부족(undertraining)**으로 추정.

### 11. 하이퍼파라미터 실험

- **실패한 시도**: `fraction=0.1`(데이터 10%만 사용)로 재학습 → **TP=0, 완전 실패**. 원인은 "1 epoch당 가중치 업데이트 횟수 = 데이터 개수 / batch 크기" 공식에서, 데이터를 10%로 줄이면서 epoch는 그대로 둬서 총 업데이트 횟수가 1/10로 줄었기 때문. 이후 실험은 전부 `fraction` 없이(=1.0, 전체 데이터) 진행.
- **실험1 (epochs=30)**: FN 가설(학습 부족) 검증 → **TP=103, FP=1, FN=0** (Recall 1.000). 학습 곡선도 재확인해서 과적합 아님을 검증.
- **실험2 (conf 조정, 재학습 없음)**: FP 가설(NMS 실패) 검증. `nms_iou`를 0.7→0.5로 낮춰도 예측 박스 개수가 전혀 안 바뀌는 것을 확인(모델이 NMS-free 구조라는 근거 재확인). 대신 `conf`를 0.25→0.45로 올리자 **exp1 기준 남은 FP 1개까지 사라짐 (Precision=1.000, Recall=1.000)**.
- 여러 모델(가중치 경로)을 같은 기준으로 비교하기 위해 `evaluate_model_on_samples()` 함수로 평가 로직을 재사용 가능하게 정리함.

### 12. train/val/test 이미지 중복(leakage) 확인

`train_test_split`은 파일 리스트를 나누는 방식이라 구조적으로 같은 파일명이 두 세트에 동시에 들어갈 수 없지만, synthetic 데이터 특성상 "파일명은 다른데 내용이 같은" 이미지가 섞여 있을 가능성을 확인함. 전체 1995장(train 1197 + val 399 + test 399)의 MD5 해시를 비교한 결과, **train/val/test에 걸쳐 중복된 이미지 그룹 0개** — leakage 없음을 확인.

### 13. 학습에 안 쓰인 새 이미지(실제 사진)로 추론 테스트 — synthetic→실제 일반화 한계 발견

Unsplash에서 항공뷰/거리뷰 사진 6장을 구해 exp1(epochs=30, conf=0.45)로 추론한 결과, **차량이 여러 대 보이는 사진에서도 0~2대밖에 탐지 못함** (synthetic test에서의 Precision/Recall 1.0과 극명하게 대비).

원인을 구분하기 위해 `imgsz`를 224→1280으로 올려서 재확인:
- 일부 사진(각도가 비교적 수직에 가까운 것)은 탐지 개수가 늘어남 → **해상도 문제**가 일부 원인 (원본이 3000~6900px급으로 커서, `imgsz=224`로 리사이즈하면 차가 몇 픽셀로 쪼그라듦)
- 반면 각도가 많이 비스듬한 사진(거리뷰)은 해상도를 올려도 그대로 → **각도(시점) domain gap**이 더 근본적인 원인

결론: 학습 데이터가 전부 수직에 가까운 항공뷰 synthetic 이미지로만 구성돼서, 그와 다른 각도의 실제 사진에는 일반화가 잘 안 됨. `imgsz`, `conf` 같은 추론 파라미터로는 각도 문제 자체를 해결할 수 없음(모델이 애초에 학습 때 못 본 시각 패턴이라서).

### 14. Perspective/Affine Augmentation 실험 (실험3, 각도 일반화 시도)

2주차 과제피드백 Q&A(김시우님 질문 — "라벨 위치/시점이 특정 패턴에 몰려 있을 때, 추가 데이터 확보가 어려운 상황에서 일반화 성능을 높이는 방법") 답변을 참고. 새 데이터를 모으고 라벨링하는 대신, **기존 이미지를 학습 중에 무작위로 회전/기울임/원근왜곡시키는 augmentation**으로 각도 다양성을 인위적으로 넓히는 시도.

- ultralytics 기본값은 `perspective=0.0, shear=0.0, degrees=0.0`(전부 꺼져 있음) — 이번 실험에서 `perspective=0.0005, shear=5.0, degrees=10.0`으로 켬
- 나머지 조건은 실험1과 동일(epochs=30, imgsz=224, batch=20)

**결과**:

| 비교 | exp1 (augmentation 없음) | exp3 (augmentation 추가) |
|---|---|---|
| synthetic test (39장) | Precision=1.000, Recall=1.000 | Precision=0.990, Recall=1.000 (FP 1개 재발생) |

새 이미지(실제 사진 6장) 탐지 개수 비교:

| 파일명 | exp1 | exp3 |
|---|---|---|
| john-arano | 1 | 2 |
| miraxh-tereziu | 2 | 2 |
| ryan-besgrove | 2 | 2 |
| tony-hsu | 1 | 3 |
| warren-valentine | **0** | **1** |
| yoel-j-gonzalez | 2 | 1 |

**해석**: augmentation을 추가하니 synthetic test에서는 Precision이 아주 살짝 떨어졌지만(1.000→0.990, 중복 예측 1건 재발생), 실제 사진 6장 중 3장은 개선, 2장은 동일, 1장만 악화됨. 특히 **exp1이 아예 0개를 탐지했던 사진(warren)이 exp3에서는 1개라도 탐지**하기 시작한 게 의미 있음 — "각도 다양성 augmentation이 완벽하진 않아도 실제 일반화에는 어느 정도 도움이 된다"는 근거로 볼 수 있음. 다만 표본이 6장뿐이라 통계적으로 확정하기엔 이르고, synthetic 성능과의 트레이드오프가 있다는 점도 같이 기록해둠.

### 15. 최종 모델 선정

**추론 속도 / 모델 크기 측정**

baseline / exp1(epochs=30) / exp3(augmentation) 세 모델을 같은 이미지 10장 기준으로 측정.

| 모델 | 1차 측정 | 2차 측정(순서 반대로) |
|---|---|---|
| baseline | 115.46ms | 153.82ms |
| exp1 | 115.88ms | 115.08ms |
| exp3 | **163.14ms** | 117.14ms |

1차에서는 exp3가 유독 느리게 나왔지만, 측정 순서를 반대로(exp3→exp1→baseline) 돌려보니 **이번엔 baseline이 느리게 나옴** — 즉 "어떤 모델이든 그 측정에서 가장 마지막에 실행된 모델이 느리다"는 패턴. 세 모델 모두 구조(yolo26n)와 파라미터 개수가 동일하므로(augmentation은 학습 데이터에만 영향, 모델 구조는 안 바꿈), 이 차이는 실제 모델 성능 차이가 아니라 Colab GPU를 오래 쓰면서 생기는 세션 내 자원 변화(발열 스로틀링 등) 때문으로 판단. **결론: 세 모델의 추론 속도(~115ms/image)와 모델 크기(5.08~5.09MB)는 사실상 동일** — 속도/크기는 모델 선택의 기준이 되지 못함.

**최종 비교표**

| 기준 | baseline (epochs=10) | exp1 (epochs=30) | exp3 (epochs=30 + augmentation) |
|---|---|---|---|
| synthetic Precision | 0.934 | 1.000 | 0.990 |
| synthetic Recall | 0.961 | 1.000 | 1.000 |
| 실제 사진(6장) 탐지 | - | 기준 | 3장 개선 / 2장 동일 / 1장 악화 |
| 추론 속도 | ~115ms | ~115ms | ~115ms |
| 모델 크기 | 5.08MB | 5.09MB | 5.09MB |

**최종 선정: exp3 (epochs=30 + perspective/shear/degrees augmentation)**

- 속도/크기는 세 모델이 동일해서 변별력이 없고, 실질적 차이는 정확도/일반화 두 축에서만 발생함.
- synthetic test 기준으로는 exp1이 근소하게 우세하지만(Precision 1.000 vs 0.990), 이 차이는 1000장당 오탐 1건 수준의 작은 손해임.
- 반면 실제 사진(synthetic과 분포가 다른 데이터)에서는 exp3가 대부분 동일하거나 개선되었고, exp1이 아예 탐지하지 못했던 사진(0개)에서도 exp3는 1개를 잡아냄.
- 실제 서비스(3주차 FastAPI)에서 입력될 이미지는 synthetic test셋보다 이런 실제 사진에 훨씬 가까울 것이므로, **synthetic 성능의 작은 손해를 감수하고 실전 일반화를 택함**. 이는 2주차 과제피드백의 개별 Q&A(김시우님 질문 — 추가 데이터 없이 일반화 성능을 높이는 방법)에 대한 직접적인 답이기도 함.
- 한계: 실제 사진 검증 표본이 6장뿐이라 통계적으로 확정적인 결론은 아니며, 3주차 서비스 운영 중 실제 입력 이미지가 쌓이면 재검증이 필요함. augmentation은 각도 domain gap을 **완화**했을 뿐 **해소**한 건 아님(6장 중 1장은 오히려 악화) — "해결됨"이 아니라 "완전 해결에는 못 미치지만, 추가 라벨링 없이 시도해볼 수 있는 방향 중 하나"로 정리함.
- **다음 단계(시간이 더 있다면)**: ① `perspective`/`degrees` 강도를 더 올려서 재실험, ② 실제 각도 사진을 소량이라도 직접 라벨링해서 fine-tuning, ③ 3주차 FastAPI 서비스 문서에 이 한계(비스듬한 각도 입력에서는 탐지율이 떨어질 수 있음)를 명시해 사용자가 알고 쓰도록 함.

**최종 선정 가중치 경로**: `runs/detect/train_results/exp3_augment/weights/best.pt` (권장 추론 설정: `conf=0.45, imgsz=224`)

## 느낀 점 / 어려웠던 점

- **`fraction`으로 빠른 실험을 하려다 오히려 잘못된 결론을 낼 뻔함**: Colab GPU 시간을 아끼려고 데이터의 10%만 써서 학습했는데 결과가 TP=0으로 완전히 망가졌다. 처음엔 "왜 이렇게 잘 안 되지" 싶었는데, "1 epoch당 가중치 업데이트 횟수 = 데이터 개수 / batch 크기"라는 걸 다시 짚어보니 데이터를 줄이면서 epoch는 그대로 둬서 총 학습량 자체가 1/10로 줄어든 거였다. **"빠르게 실험한다"는 게 무조건 데이터를 줄이는 게 아니라, 뭘 고정하고 뭘 줄여야 공정한 비교가 되는지부터 따져야 한다**는 걸 몸으로 배움.
- **경로 하나 잘못 적어서 한참 헤맴**: `trained_model = YOLO('train_results/train_result/weights/best.pt')`처럼 경로 앞에 `runs/detect/`가 빠져 있던 걸 못 알아채고, 한동안 baseline과 fraction 실험 모델을 헷갈려서 평가함. `glob.glob('**/best.pt', recursive=True)`로 실제 파일이 어디 있는지 직접 찾아보고 나서야 정리됨. 실행이 에러 없이 되더라도 "진짜 내가 의도한 파일을 불러온 게 맞는지"는 별도로 확인해야 한다는 걸 느낌.
- **NMS 파라미터가 안 먹혀서 당황함**: FP 대부분이 중복 예측이라는 걸 확인하고 `iou`(NMS 임계값)를 낮추면 해결될 줄 알았는데, 극단적으로 0.1까지 낮춰도 예측 박스 개수가 전혀 안 바뀌었다. 처음엔 내 코드가 잘못된 줄 알았는데, 한 이미지로 직접 박스 개수를 비교해보고 나서야 "이 모델(YOLO26) 자체가 전통적인 NMS 후처리를 안 쓰는 구조일 수 있다"는 결론에 도달함. 라이브러리 옵션이 항상 기대한 대로 동작하는 게 아니라는 걸 확인.
- **synthetic test에서 Precision/Recall 1.0이 나왔을 때 마냥 좋아할 수만은 없었음**: 처음엔 완벽한 수치에 신났는데, "표본이 작다", "이 test 결과를 보고 conf 값을 골랐다"는 한계를 짚고 나니 이 숫자를 그대로 믿기보다 leakage 확인 + 새 이미지 테스트로 한 번 더 검증해야겠다는 생각이 들었음.
- **실제 사진으로 테스트했을 때 성능이 확 떨어진 게 제일 인상 깊었음**: synthetic test에서는 완벽했던 모델이, 실제 항공뷰/거리뷰 사진에서는 차가 여러 대 보여도 0~2대밖에 못 잡았다. `imgsz`를 키워서 해상도 문제는 어느 정도 완화됐지만, 각도가 많이 다른 사진은 그래도 안 잡혔다. **"test set 성능이 좋다"는 게 "실전에서도 통한다"는 뜻이 아니라는 걸 숫자가 아니라 직접 눈으로 확인하고 나서야 체감**함. 이 경험 덕분에 2주차 과제피드백에서 왜 "새로운 데이터로 반드시 추론해보라"고 강조했는지 이해가 됨.
- **각도 문제를 어떻게 풀어야 할지 고민했던 부분**: 처음엔 "새 데이터를 모아서 라벨링하고 재학습해야 한다"고만 생각했는데, 피드백 문서의 Q&A(김시우님 질문)를 보고 나서야 `perspective`/`shear`/`degrees` 같은 augmentation으로 기존 데이터 안에서도 각도 다양성을 어느 정도 흉내낼 수 있다는 걸 알게 됨. 문제 해결 방법이 "데이터 추가"만 있는 게 아니라는 걸 배움.
