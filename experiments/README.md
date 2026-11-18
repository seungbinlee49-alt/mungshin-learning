# 로컬 연구 자료를 사용하는 재현 코드

특징 추출·정규화·PCA·교차곱·분류기 함수는 기존 최종 공유 패키지에서 복사했습니다. `SOURCE_MAP.json`에 원 함수 출처를 기록했습니다. `run_local.py`는 공개 저장소용 실행기이며 기존 고정 함수로 학습·평가합니다.

## 자료가 없는 경우

사진·좌표·특징 캐시는 공개하지 않습니다. 저장소를 clone하는 것만으로 학습할 수 있는 공개 데이터셋이 제공되는 것은 아닙니다. `../site/data/`의집계 결과와 `../tools/check_report.py`는 데이터 없이 확인할 수 있습니다.

## 팀의 로컬 자료를 가진 경우

기존 공유 패키지에서 다음 파일을 **본인의 컴퓨터에서만** `data/`로 복사하세요.

- `features.npz`: person/dog 각각 global(200,3,2724), eye(200,3,20)
- `plan.json`, `photo_selection.json`은 고정 계획으로 이미 포함돼 있습니다.
- 원본 이미지 특징 재검사가 필요하면 추가로 `학습용_200세트.zip`, `eye_annotations.json`을 복사합니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_local.py --seed 20261101 --folds 0 --sizes 160 --photos 3 --models logistic
```

전체 분류기·학습량·장수 비교는 아래 명령입니다. CPU에서 시간이 걸리며, 보고서 확인만을 위해 실행할 필요는 없습니다.

```bash
python run_local.py --seed 20261101
python run_local.py --seed 20261102
python run_local.py --seed 20261103
python run_local.py --analyze
```

학습 완료 checkpoint는 `outputs/`에 저장하고 재사용합니다. 원본 자료나 공개 집계표를 덮어쓰지 않습니다. `--analyze`는 필요한 전체 조합이 완료됐을 때만 실행합니다.

`prepare_features.py --check-samples`는 원본 이미지와 좌표가 있는 로컬 환경에서 실제 특징14장을 재계산해 캐시와 비교합니다. 원본 이미지 없이 특징을 새로 생성하는 기능은 아닙니다.

공개 보고서 기준 모델은 수동 전처리·전통 특징을 사용합니다. 자동 누끼/눈 검출 시연의 사전학습 모델은 여기에 포함되지 않습니다.
