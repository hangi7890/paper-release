# BDG2 메타데이터·전력 품질 결합 및 계량 범위 검토

- 결합 키: `building_id`
- 입력: `data/raw/bdg2/metadata.csv`, `reports/bdg2_electricity_quality.csv`, `reports/bdg2_zero_value_audit.csv`
- 출력 표: `reports/bdg2_metadata_quality_join.csv`
- 원본과 기존 결과 파일은 수정하지 않았으며, 건물 자동 제외·결측 처리·기상 데이터 다운로드·학습은 수행하지 않았다.

## 1. 결합 확인

- metadata 행: 1,636개
- 기존 전력 품질 행: 1,578개
- 기존 0값 감사 행: 1,578개
- 결합 성공 전력 건물: **1,578개**
- 결합 오류: 0건
- 전력 품질 결과의 1,578개 `building_id`가 모두 metadata와 1:1로 결합되도록 검사했다.

## 2. 전력 건물의 사이트별 분포

| site_id | 전력 건물 수 | 숫자 기록 모두 0 | 숫자 기록 없음 |
|---|---:|---:|---:|
| Rat | 305 | 0 | 1 |
| Hog | 152 | 3 | 0 |
| Lamb | 146 | 0 | 0 |
| Fox | 137 | 0 | 0 |
| Bull | 123 | 0 | 0 |
| Cockatoo | 117 | 0 | 0 |
| Eagle | 106 | 1 | 1 |
| Panther | 105 | 0 | 0 |
| Bear | 92 | 0 | 0 |
| Gator | 74 | 0 | 0 |
| Robin | 52 | 0 | 0 |
| Peacock | 45 | 0 | 0 |
| Wolf | 36 | 0 | 0 |
| Bobcat | 35 | 0 | 0 |
| Swan | 19 | 0 | 0 |
| Moose | 13 | 0 | 0 |
| Shrew | 9 | 0 | 0 |
| Mouse | 7 | 0 | 0 |
| Crow | 5 | 0 | 0 |

## 3. 전력 건물의 용도별 분포

| primaryspaceusage | 전력 건물 수 | 숫자 기록 모두 0 | 숫자 기록 없음 |
|---|---:|---:|---:|
| Education | 604 | 0 | 0 |
| Office | 296 | 0 | 0 |
| Entertainment/public assembly | 203 | 4 | 0 |
| Public services | 166 | 0 | 1 |
| Lodging/residential | 149 | 0 | 1 |
| Healthcare | 27 | 0 | 0 |
| Other | 26 | 0 | 0 |
| Parking | 23 | 0 | 0 |
| [missing] | 19 | 0 | 0 |
| Warehouse/storage | 14 | 0 | 0 |
| Manufacturing/industrial | 11 | 0 | 0 |
| Retail | 11 | 0 | 0 |
| Services | 9 | 0 | 0 |
| Technology/science | 7 | 0 | 0 |
| Food sales and service | 6 | 0 | 0 |
| Utility | 4 | 0 | 0 |
| Religious worship | 3 | 0 | 0 |

## 4. 결합 표의 요약 지표

- 결측률(전체 기록 기준) 건물별 최소/중앙값/최대: 0.000000 / 0.001482 / 1.000000
- 0값 비율(전체 기록 기준) 건물별 최소/중앙값/최대: 0.000000 / 0.000342 / 1.000000
- 숫자 기록 모두 0: 4개
- 숫자 기록 없음: 2개

결합 CSV에는 `site_id`, `primaryspaceusage`, `sqm`, `timezone`, 기존 품질 수치, 결측률·0값 비율, 최장 연속 유효 기록 길이, 최장 연속 0구간 및 검토 상태를 함께 기록했다.

## 5. BDG2 공식 자료에서 확인되는 계량 범위 설명

### 공식 자료에서 확인된 사실

1. BDG2 README는 데이터셋을 1,636개 건물의 3,053개 에너지 계량기로 설명하고, 2016~2017년의 시간별 electricity, heating/cooling water, steam, irrigation 계량값을 제공한다고 명시한다.
2. README의 인용 논문 초록은 건물별 하나 이상의 계량기가 whole building electrical, heating/cooling water, steam, solar energy, water, irrigation 등을 측정한다고 데이터셋 수준에서 설명한다.
3. Meters data description Wiki는 raw meter 자료가 2016~2017년 시간당 한 건의 kWh 계량값이며, 계량기 종류별 데이터셋이 있다고 설명한다.
4. Metadata 문서는 `electricity`를 전력 계량기 보유 여부를 나타내는 메타데이터 특징으로 설명하지만, 개별 열의 물리적 계량 경계나 submeter 여부를 나타내는 필드는 제시하지 않는다.

### 개별 건물의 총소비전력 여부에 관한 미확인 사항

- 데이터셋 전체 설명의 `whole building` 표현만으로 `electricity.csv`의 각 열이 건물 전체의 실제 총소비전력이라고 확정할 수 없다.
- 이번 세 파일에는 계량기 식별자, 계량 지점, 물리적 경계, main meter/submeter 구분, 전력 열별 공식 단위·범위 매핑이 없다.
- 따라서 결합 표의 모든 전력 건물에 `meter_scope_status=확인 필요`를 기록했다. 숫자 기록이 모두 0인 4개와 숫자 기록이 없는 2개도 자동 제외하지 않았다.
- 개별 건물의 총소비전력 여부는 시설별 원자료 설명이나 별도 계량기 매핑 없이는 확인할 수 없다.

## 6. 공식 출처 URL

- [BDG2 공식 README](https://github.com/buds-lab/building-data-genome-project-2)
- [Meters data description Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-description)
- [Meters data features Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features)
- [Metadata features Wiki](https://github.com/buds-lab/building-data-genome-project-2/wiki/Metadata-features)
- [BDG2 데이터 논문 DOI](https://doi.org/10.1038/s41597-020-00712-x)
