# BDG2 공식 배포처·원자료 설명 검토

- **검토일:** 2026-09-23
- **대상 저장소:** [buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2)
- **검토 범위:** 공식 GitHub README, GitHub Wiki, 저장소 파일 페이지 및 저장소 LICENSE
- **수행하지 않은 작업:** 대용량 CSV 다운로드, Git LFS 설치, CSV 본문 열람, 데이터 전처리, 패키지 설치, 모델 학습

## 1. 공식 배포처와 파일 목록

저장소 README는 BDG2를 1,636개 비주거 건물의 3,053개 에너지 계량기, 2016·2017년 두 개의 완전한 연도, 시간별 측정으로 설명한다. 데이터 디렉터리는 `metadata`, `weather`, `meters/raw`, `meters/cleaned`, `meters/kaggle`로 구성된다. 저장소의 모든 `.csv` 파일은 Git LFS로 관리되며, 각 파일 페이지에서 Raw/Download raw 링크가 제공된다.

파일 크기는 **GitHub 파일 페이지에 표시된 값**이며, 로컬 파일의 실제 바이트 수나 내용 검증 결과가 아니다.

### 1.1 건물 메타데이터와 기상 데이터

| 경로 및 공식 파일 페이지 | 형식 | 페이지 표시 크기 | 공식 문서상 역할 |
|---|---:|---:|---|
| [`data/metadata/metadata.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/metadata/metadata.csv) | CSV | 266 KB | 건물·사이트·용도·면적·위치·시간대·계량기 종류 보유 여부 등을 담는 메타데이터 |
| [`data/weather/weather.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/weather/weather.csv) | CSV | 18.6 MB | 19개 사이트의 시간별 ISD-Lite 기상 자료 |

### 1.2 원시 계량 데이터 (`data/meters/raw`)

| 공식 파일 페이지 | 형식 | 페이지 표시 크기 | 파일명으로 표시된 계량기 종류/역할 |
|---|---:|---:|---|
| [`chilledwater.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/chilledwater.csv) | CSV | 75.2 MB | chilled water 계량 |
| [`electricity.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/electricity.csv) | CSV | 166 MB | electricity 계량 |
| [`gas.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/gas.csv) | CSV | 19.2 MB | gas 계량 |
| [`hotwater.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/hotwater.csv) | CSV | 23.8 MB | hot water 계량 |
| [`irrigation.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/irrigation.csv) | CSV | 3.01 MB | irrigation 계량 |
| [`solar.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/solar.csv) | CSV | 726 KB | solar 계량 |
| [`steam.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/steam.csv) | CSV | 44.6 MB | steam 계량 |
| [`water.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/raw/water.csv) | CSV | 14.9 MB | water 계량 |

공식 Wiki의 **Meters data features** 설명에 따르면 원시 계량 파일은 계량기 종류별로 하나씩 있으며, `timestamp`와 건물별 열을 포함하고 시간당 한 개의 읽기값을 갖는다. 이는 파일 구조에 대한 문서 설명이며, 실제 열 이름·값·결측 상태를 이번 단계에서 직접 확인한 것은 아니다.

### 1.3 정제 계량 데이터 (`data/meters/cleaned`)

| 공식 파일 페이지 | 형식 | 페이지 표시 크기 | 공식 문서상 역할 |
|---|---:|---:|---|
| [`chilledwater_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/chilledwater_cleaned.csv) | CSV | 73.4 MB | chilled water 정제본 |
| [`electricity_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/electricity_cleaned.csv) | CSV | 167 MB | electricity 정제본 |
| [`gas_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/gas_cleaned.csv) | CSV | 21 MB | gas 정제본 |
| [`hotwater_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/hotwater_cleaned.csv) | CSV | 23.2 MB | hot water 정제본 |
| [`irrigation_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/irrigation_cleaned.csv) | CSV | 2.93 MB | irrigation 정제본 |
| [`solar_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/solar_cleaned.csv) | CSV | 719 KB | solar 정제본 |
| [`steam_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/steam_cleaned.csv) | CSV | 43.3 MB | steam 정제본 |
| [`water_cleaned.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/water_cleaned.csv) | CSV | 14.7 MB | water 정제본 |

Wiki는 정제 과정에서 S-H-ESD로 검출한 이상치, 24시간을 초과하는 연속 0값, electricity의 0값을 `NaN`으로 바꾼다고 설명한다. 따라서 `cleaned`라는 이름만으로 완전성이나 물리적 타당성이 보장된다고 해석하지 않는다.

### 1.4 GEPIII/Kaggle 부분집합

| 공식 파일 페이지 | 형식 | 페이지 표시 크기 | 공식 문서상 역할 |
|---|---:|---:|---|
| [`data/meters/kaggle/kaggle.csv`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/kaggle/kaggle.csv) | CSV | 855 MB | ASHRAE Great Energy Predictor III 공개 리더보드에 사용된 2017년 선택 건물·계량기 부분집합 |

Wiki는 Kaggle 부분집합이 electricity, chilled water, hot water, steam을 포함하며, 모든 값이 kWh로 완전히 변환된 것은 아니라고 명시한다. 사이트와 계량기 종류에 따라 kBTU, tons, mmBTU, MJ, kWh 등의 단위와 변환계수가 표로 제시된다.

## 2. 전력·건물 메타데이터·기상 파일의 구성

공식 저장소 구조는 다음과 같다([README의 Repository structure](https://github.com/buds-lab/building-data-genome-project-2#repository-structure)).

```text
data/
├─ metadata/metadata.csv
├─ weather/weather.csv
└─ meters/
   ├─ raw/       (계량기 종류별 원시 CSV 8개)
   ├─ cleaned/   (계량기 종류별 정제 CSV 8개)
   └─ kaggle/    (2017년 GEPIII/Kaggle 부분집합 1개)
```

- **전력 계량 데이터:** `electricity.csv` 및 `electricity_cleaned.csv`가 별도 제공된다. 다른 에너지·용량 계량은 파일명별 CSV로 분리된다.
- **건물 메타데이터:** `metadata.csv`는 공식 설명상 30개 feature로 1,636개 건물과 3,053개 계량기를 설명한다.
- **기상 데이터:** `weather.csv`는 19개 사이트의 NOAA/NCEI ISD-Lite 자료를 하나의 CSV로 제공한다. 공식 문서는 10개 기상 feature를 열거한다.
- **파일 형식:** 모두 CSV이며, 저장소에서는 `.csv`를 Git LFS로 제공한다. 이번 검토에서는 CSV 본문을 열지 않았다.

## 3. 건물 전체 전력 계량기와 기타 계량기를 구분할 수 있는 메타데이터

### 공식 자료에서 확인되는 것

`Metadata features` Wiki는 다음 필드를 공식적으로 열거한다.

- 식별·연결: `building_id`, `site_id`, `building_id_kaggle`, `site_id_kaggle`
- 건물 설명: `primaryspaceusage`, `sub_primaryspaceusage`, `sqm`, `sqft`, `lat`, `lng`, `timezone` 등
- 계량기 종류 보유 여부: `electricity`, `hotwater`, `chilledwater`, `steam`, `water`, `irrigation`, `solar`, `gas`
- 계량기 종류 보유 여부 필드는 문서상 해당 종류가 있으면 `Yes`, 없으면 `NaN`이다.

저장소 README와 인용 논문 설명은 BDG2를 비주거 건물의 **whole-building energy meter data**로 소개하며, 전기·냉수/온수·증기·태양광·물·관개 등 여러 종류의 계량기를 포함한다고 설명한다. 따라서 `electricity`라는 **계량기 종류의 존재 여부**를 확인할 수 있는 메타데이터는 있다.

### 이번 공식 자료 검토에서 확인하지 못한 것

- `metadata.csv`의 보유 여부 필드만으로 특정 electricity 열이 건물 전체 총전력인지, 하위계량기인지, 어떤 물리적 경계를 포함하는지 판정하는 필드는 확인하지 못했다.
- per-meter 물리적 계량기 ID, 주계량기/하위계량기 구분, 계량기 범위·위치·합산 규칙을 설명하는 메타데이터 필드는 공식 feature 목록에서 확인하지 못했다.
- 저장소 문서의 “whole-building” 표현은 데이터셋의 일반적 설명이며, 각 건물·각 열의 실제 계량 범위를 직접 검증한 결과가 아니다.
- 따라서 연구의 예측 대상에 `electricity.csv`를 바로 건물 전체 총소비전력으로 사용할 수 있다고 확정하지 않는다. 원자료의 열, 원 출처 설명, 계량 범위를 후속 단계에서 직접 검수해야 한다.

## 4. 공식 문서의 기간·간격·시간대·단위

| 항목 | 공식 자료에서 확인한 내용 | 출처 |
|---|---|---|
| 계량 관측 기간 | 2016년과 2017년의 두 완전한 연도 | [README](https://github.com/buds-lab/building-data-genome-project-2#data-set-description), [Meters data description](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-description) |
| 계량 간격 | 시간별 1개 읽기값 | [Meters data features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features) |
| 계량 timestamp 형식 | `YYYY-MM-DD hh:mm:ss`, local timezone | [Meters data features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features) |
| 계량 DST 처리 | 공식 계량 설명에서 명시적인 DST 처리 규칙을 확인하지 못함 | [Meters data features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features) |
| 기상 timestamp | `YYYY-MM-DD hh:mm:ss`, local timezone; DST는 무시한다고 명시 | [Weather features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Weather-features) |
| 사이트 시간대 | metadata의 `timezone` 필드. 공식 설명의 범주 예시는 US/Eastern, Europe/London, US/Mountain, US/Pacific, US/Central, Europe/Dublin | [Metadata features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Metadata-features), [Metadata description](https://github.com/buds-lab/building-data-genome-project-2/wiki/Metadata-description) |
| 원시·정제 계량 단위 | 공식 Wiki는 모든 energy meter를 kWh, 모든 volume meter를 litre로 설명 | [Meters data features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features) |
| Kaggle 부분집합 단위 | kWh로 완전히 통일되지 않았으며 사이트·계량기별 kBTU, tons, mmBTU, MJ, kWh 등이 표에 제시됨 | [Meters data features](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features) |

여기서 **데이터 관측 기간(2016–2017)**과 **데이터셋 수집·정리 작업 기간(2017년 9월–2019년 5월)**을 구분한다. 후자의 기간은 여러 사이트의 원자료를 수집·통합한 프로젝트 작업 기간이며, 관측값이 그 기간에 측정되었다는 뜻이 아니다([Building data collection](https://github.com/buds-lab/building-data-genome-project-2/wiki/Building-data-collection)).

## 5. 데이터 사용 조건과 공식 배포처에서 확인하지 못한 사항

### 5.1 사용 조건 및 출처 표기

1. 저장소의 [`LICENSE`](https://github.com/buds-lab/building-data-genome-project-2/blob/master/LICENSE)는 **Creative Commons Attribution-ShareAlike 4.0 Unported (CC BY-SA 4.0)**이다.
2. LICENSE는 배포 시 저작권·라이선스 고지를 유지하고, 라이선스 URI 또는 사본을 포함하며, 변경한 경우 변경 사실을 표시하고, 원 저작자·제목·URI 등 합리적인 방식의 attribution을 제공하도록 요구한다.
3. Adaptation을 배포할 때에는 동일한 License Elements(Attribution, ShareAlike)를 유지해야 하며, 수신자에게 추가적인 제한을 부과할 수 없다.
4. README는 BDG2 논문을 인용하도록 안내한다: [Miller et al., Scientific Data (2020), DOI 10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x).
5. 기상 자료는 NOAA/NCEI ISD에서 왔다. Wiki는 NOAA/NCEI가 정확성·신뢰성·완전성을 보증하지 않으며, 사용자가 목적에 대한 적합성과 결과에 책임을 진다고 명시한다([Weather data collection](https://github.com/buds-lab/building-data-genome-project-2/wiki/Weather-data-collection)).
6. 저장소의 모든 CSV는 Git LFS로 관리된다([Git LFS Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Git-LFS)). 이는 후속 다운로드 시 Git LFS가 필요할 수 있다는 뜻이며, 이번 단계에서는 설치하지 않았다.

### 5.2 공식 배포처 검토에서 확인하지 못한 사항

- 현재 `master` 기준 파일의 **버전 고정 릴리스, per-file SHA-256 checksum, 다운로드 아카이브의 검증용 manifest**는 이번 공식 저장소·Wiki 검토에서 찾지 못했다.
- GitHub 파일 페이지의 표시 크기는 확인했지만, 실제 파일을 다운로드해 바이트 수·행 수·열 수·압축 여부·CSV 인코딩·헤더를 검증하지 않았다.
- 공식 문서는 결측·이상치 처리 절차를 설명하지만, 각 파일·각 건물의 결측률, 중복 timestamp, 시간대 전환, 계량기 초기화, 물리적 이상값이 없다고 보증하지 않는다. Wiki의 screening 설명도 raw data를 사용한 별도 품질 탐색임을 밝힌다([Meters data screening](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-screening)).
- 19개 사이트의 원자료가 모두 같은 계량 경계·같은 계량기 정의를 사용한다고 공식 문서만으로 확정할 수 없다. 공식 수집 문서는 7개 사이트는 온라인 공개 자료였고, 나머지 12개 사이트는 시설관리 담당자와 수작업으로 수집했으며 그 원시 파일은 저장소에 포함되지 않았다고 설명한다([Building data collection](https://github.com/buds-lab/building-data-genome-project-2/wiki/Building-data-collection)).
- 저장소 LICENSE가 BDG2 저장소 작업물에 적용된다는 사실과, NOAA 및 각 원자료 제공 사이트의 별도 이용 조건은 구분해야 한다. 모든 upstream source의 개별 법적 조건이 저장소 문서 하나에 통합되어 있다고 해석하지 않는다.
- metadata의 계량기 보유 여부만으로 건물 전체 총전력, 하위계량기, 계통 순전력, 태양광 발전량을 구분할 수 있는 공식 매핑은 확인하지 못했다.

## 6. 이번 단계의 결론

BDG2 공식 저장소는 Git LFS로 제공되는 CSV 기반 자료이며, 계량기는 원시·정제·Kaggle 부분집합으로 나뉘고, 별도의 건물 메타데이터와 19개 사이트의 시간별 기상 자료가 제공된다. 그러나 현재 공식 문서만으로는 연구 논문에서 요구하는 **건물 전체 실제 전력 소비량의 물리적 계량 범위**를 건물별로 확정할 수 없다. 다음 단계에서 데이터를 사용하려면 먼저 선택한 건물과 `electricity` 계량 열의 범위·단위·시간대·결측 상태를 원자료와 출처 문서로 검수해야 한다.

