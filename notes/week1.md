# 1주차

## 학습 내용

도로 위의 차량 객체탐지(YOLO) 프로젝트를 위해, 데이콘 원본 데이터셋(차종별 라벨)을 "차(car)" 단일 클래스의 YOLO 포맷으로 변환하고, train/valid/test로 분리해서 학습용 폴더 구조 + `data.yaml`까지 만드는 데이터셋 구축 과정을 진행함 ([code/01*데이터셋*구축\_코드.ipynb](../code/01_데이터셋_구축_코드.ipynb) 1~8번 섹션 완료).

## 진행한 작업

### 1. Google Drive 마운트

Colab은 매번 새로 뜨는 임시 가상머신이라 로컬 디스크가 유지되지 않음. 구글 드라이브의 데이터셋에 접근하려면 `drive.mount('/content/drive')`로 마운트가 필요함. 로컬 진행 시엔 생략.

### 2. 라이브러리 임포트

```python
import os, cv2, glob, shutil, warnings
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml
```

- `os`: 파일/폴더 경로 다루기, 폴더 생성 여부 확인, 폴더 생성(`os.mkdir`/`os.makedirs`) 등
- `cv2` (OpenCV): 이미지 읽기/색상 변환, bbox 그리기, `.shape`로 이미지 가로/세로 크기 확인
- `glob`: 특정 패턴(`*.txt` 등)에 맞는 파일 목록을 한 번에 가져오기
- `shutil`: 파일 복사/이동 (`shutil.copy`로 train/val/test 폴더에 이미지·라벨 옮길 때 사용)
- `warnings`: 불필요한 경고 메시지 숨기기 (`warnings.filterwarnings(action='ignore')`)
- `matplotlib.pyplot`: 이미지 및 bbox 시각화
- `tqdm`: 반복문 돌 때 진행률 바 표시 (파일 수백~수천 개 처리할 때 유용)
- `train_test_split`: 데이터를 무작위로 두 그룹으로 분리 (`test_size`로 비율 조정)
- `from PIL import Image` (Pillow): 이미지를 열어서 `.size`로 가로/세로 픽셀 크기 확인 — 원래 임포트 목록에 빠져 있었는데, 5번 섹션 라벨 검증 코드(`draw_boxes_on_image`)에서 `NameError`가 나서 뒤늦게 추가함
- `import yaml`: 파이썬 딕셔너리 ↔ YAML 파일 형식 변환 (`yaml.dump`로 저장, `yaml.full_load`로 읽기) — 이것도 원래 목록에 빠져 있어서 8번 섹션 `data.yaml` 생성할 때 추가함

### 3. 데이터셋 준비

원본은 차종(소나타, 아반떼 등)으로 라벨링되어 있지만, 목표는 "차 여부"만 탐지하는 것이므로 라벨을 전부 `car` 하나로 통일해야 함. 원본 라벨 경로(`origin_dir`)와 수정본 저장 경로(`edit_label_dir`)를 미리 정의.

### 4. YOLO 포맷으로 라벨 변환

원본 라벨은 Labelme 형태 `<classId, x1,y1,x2,y2,x3,y3,x4,y4>` (네 꼭짓점, 절대 픽셀 좌표). 이걸 YOLO 포맷 `classId x_center y_center width height` (0~1로 정규화)로 변환해야 함. 함수 3개로 분리해서 구현:

```python
def initialize_label_data(label_txt):        # 라벨 파일 하나 읽어서 float 리스트로 변환
def generate_new_label_data(lines, img_width, img_height):  # 꼭짓점 -> bbox -> 정규화된 center/width/height, classId=0 고정
def save_new_label_data(edit_lines, label_txt):  # edit_label_dir에 원본과 같은 파일명으로 저장
```

메인 반복문에서 각 라벨 파일마다 대응하는 이미지를 찾아(`.replace('.txt','.png')`) `cv2.imread().shape`로 이미지 크기를 구하고, 위 세 함수를 순서대로 호출.

### 5. 라벨 검증

`draw_boxes_on_image()`로 변환된 라벨을 이미지 위에 bbox로 그려서 육안으로 확인. (`from PIL import Image` 임포트가 누락되어 있던 것을 추가해서 해결)

### 6. train/valid/test 분리 (6:2:2)

`train_test_split`은 한 번에 두 그룹으로만 나뉘므로, 두 번 나눠서 3분할함

```python
temp, testset = train_test_split(label_txts, test_size=0.2, random_state=42)      # 전체의 20%를 test로
trainset, validset = train_test_split(temp, test_size=0.25, random_state=42)      # 남은 80%의 25% = 전체의 20%를 valid로
```

### 7. train/valid/test 폴더 구성

`images/{train,val,test}`, `labels/{train,val,test}` 폴더를 `os.makedirs`로 생성하고, 각 분리셋에 대해 원본 이미지와 **변환된(edit_label_dir) 라벨**을 `shutil.copy`로 복사. YOLO 라이브러리가 이미지 경로의 `"images"`를 `"labels"`로 자동 치환해서 라벨을 찾기 때문에 폴더 구조/이름을 맞추는 게 중요함.

### 8. `data.yaml` 생성

클래스 정보(`names`, `nc`)와 train/val/test 이미지 폴더 경로를 담은 딕셔너리를 `yaml.dump()`로 파일에 저장. 이 파일이 다음 노트북(`02_yolo_모델_학습_코드.ipynb`)에서 YOLO 모델 학습 시 데이터셋 설정으로 사용됨.

```python
input_yaml = {'names': ['car'], 'nc': 1, 'path': '/content/images',
              'test': '/content/images/test', 'train': '/content/images/train', 'val': '/content/images/val'}
with open('/content/dataset.yaml', 'w') as f:
    yaml.dump(input_yaml, f)
```

(이 버전은 이후 팀장님 피드백을 받고 9번에서 수정됨)

### 9. 1주차 피드백 반영 (데이터셋 보완)

ylee 팀장님 피드백에서 지적받은 부분들을 보완함.

**(1) 여러 이미지 무작위 시각화 검증**

기존엔 이미지 1장(`syn_00068`)만 확인해서, 전체 데이터셋의 라벨 변환이 다 맞는지는 보장이 안 됐음. `random.sample(label_txts, 10)`으로 10개를 무작위로 뽑아 반복문으로 `draw_boxes_on_image`를 호출하도록 개선.

**(2) 이미지-라벨 개수/누락 검증 코드 추가**

`origin_dir`에서 `*.png`, `*.txt`를 각각 `glob`으로 불러와 파일명(확장자 제외)을 `set`으로 만들고, 양방향 차집합(`set(label) - set(image)`, `set(image) - set(label)`)으로 짝이 안 맞는 파일이 있는지 확인. 둘 다 빈 `set()`으로 나와 이상 없음 확인.

라벨→이미지 방향은 사실 섹션 4에서 대응 이미지가 없으면 `cv2.imread()`가 `None`을 반환해 `.shape`에서 에러가 나므로 간접적으로 이미 보장되던 부분이었음. 반면 섹션 4 루프는 `label_txts` 기준으로만 도니까, "라벨이 아예 없는 이미지"(이미지→라벨 방향)는 지금까지 전혀 검증되지 않고 있었음 — 이번 검증 코드가 그 빈틈을 메꿈.

**(3) `dataset.yaml` 상대경로로 정리**

기존엔 `path`, `train`, `val`, `test`가 전부 절대경로라 `path`가 사실상 안 쓰이고 값이 중복됐음. Ultralytics 관례대로 `path`를 루트로 잡고, `train`/`val`/`test`는 `images/train`처럼 `path` 기준 상대경로로 수정.

```python
input_yaml = {'names': ['car'], 'nc': 1, 'path': dataset_root,
              'test': 'images/test', 'train': 'images/train', 'val': 'images/val'}
```

**(4) 데이터셋 저장 위치를 Google Drive로 이전**

`/content`는 Colab 세션이 끊기면 통째로 사라지는 임시 저장소라, 2차 과제(모델 학습)에서 재사용하려면 Drive에 영구 저장해야 함. `cell-22`의 복사 대상 경로와 `cell-24`/`cell-25`의 yaml 저장·조회 경로를 전부 `dataset_root = '/content/drive/MyDrive/comento/1/datasets_yolo'` 기준으로 변경. `dataset.yaml` 파일 자체도 `dataset_root` 밑에 같이 저장하도록 수정함.

## 새로 배운 개념 / 궁금해서 짚어본 것들

- **정규화(normalize)**: bbox 픽셀 좌표를 이미지 가로/세로 크기로 나눠서 0~1 비율로 바꾸는 이유 = 이미지 크기와 무관하게 라벨을 표현하기 위함
- **`os.mkdir` vs `os.makedirs`**: 중간 경로가 없을 때 `os.makedirs`만 한 번에 다 만들어줌
- **YAML과 파이썬 딕셔너리**: YAML은 딕셔너리를 사람이 읽기 좋게 표현한 파일 포맷. `yaml.dump(dict, file)` / `yaml.load(file)`로 서로 변환됨. 코드 셀에 YAML 문법을 그대로 쓰면 안 되고 파이썬 dict로 만들어야 함 (dict의 key는 문자열이라 따옴표 필수)
- **YOLO data.yaml에 라벨(txt) 경로가 따로 없는 이유**: `images`를 `labels`로 문자열 치환해서 자동으로 라벨 경로를 찾는 ultralytics 관례 때문 → 그래서 `images/`, `labels/` 폴더 이름과 하위 구조를 맞추는 게 중요
- **절대경로 vs 상대경로**: `/`로 시작하면 절대경로(루트부터 시작), 없으면 기준이 되는 `path` 뒤에 이어붙는 상대경로. yaml의 `train`에 실수로 `/`를 붙이면 `path`가 무시되고 엉뚱한 경로(`/train`)를 가리키게 됨
- **`images`↔`labels` 자동 치환이 성립하는 조건**: 폴더 구조가 `path/images/{train,val,test}`, `path/labels/{train,val,test}`처럼 같은 상위 폴더 밑에 나란히 있어야 함. `path/{train,val,test}/images`처럼 순서가 바뀌면 yaml에 적은 상대경로(`images/train`)와 실제 폴더 위치가 어긋나서 학습 시 파일을 못 찾음
- **`random.sample(list, k)`**: 리스트에서 중복 없이 k개를 무작위로 추출
- **`set` 차집합으로 데이터 짝 검증**: `set(A) - set(B)`로 A에만 있고 B엔 없는 원소를 뽑아낼 수 있음 → 이미지/라벨처럼 짝이 맞아야 하는 두 파일 목록의 무결성 확인에 활용

## 느낀 점 / 어려웠던 점

비록 아직은 데이터 모델을 사용하지 않고 데이터만 구축해서 아직은 복습하는 느낌이 더 강했다하지만 학교에서 배우던 데이터 전처리 과정이 이렇게 쓰이겠구나 라는 것을 느꼈던 것 같다

또한 과제에서 추천한 YOLO 말고도 다른 모델을 사용하는 것에 대해 추가적으로 알아봐야겠다는 생각이 들었다.

YAML이 뭔지 YOLO가 뭔지 또 그것을 쓰기 위해 뭘 해야하는지 대한 지식이 없어서 찾아보고 하는 과정이 이번 과제에서 어려웠다.
