# final_v1 Software addendum 001

## 발견 시각과 오류

- 기록 시각(UTC): `2026-09-25T03:48:26.304254+00:00`
- 오류: 공식 `primaryspaceusage`가 비어 있는 Swan 사이트 18개가 탐색 특성 분석의 pandas `groupby` 기본 `dropna=True` 때문에 제외됨.
- 최초 표본: 해당 탐색 상관의 완전 사례 435개.
- 수정 표본: 453개. 결측 용도는 값을 추정하지 않고 문자열 범주 `<missing>`으로 명시.

## 영향 범위

- 영향 있음: 탐색적 특성 상관표, 용도별 탐색 그룹표, 탐색 보고서의 상관 수치와 표본 수.
- 영향 없음: raw prediction, 독립 MAE/RMSE, 데이터 효율성, F1~F4, H1~H3, bootstrap, cohort, temporal split, 모델 checkpoint.
- 모델 재실행·재학습: 없음.
- 가설·설정·기간·건물 변경: 없음.

## 보존·수정

최초 파일과 코드는 `reports/final_v1/addendum_pre_characteristic_fix/` 및 `scripts/final_v1/addendum_pre_characteristic_fix/`에 보존했다. 분석 함수는 결측 용도를 `<missing>` 범주로 보존하도록 수정했고 탐색 파일만 재계산했다.

| artifact | before SHA-256 | after SHA-256 |
|---|---|---|
| characteristic associations | `568b8670526935016af0e6a7251e2b4371a281820275abb34995eb8e4a246fbe` | `823f4fc353902d2536555203e74b337ace7f83023346c5dcb1d4adb0a9cec627` |
| characteristic group summary | `68f1ea111e4c0d57d94f1617a444c200cfd177026d8621f5caece87db52bb383` | `27817ff1add9f5318d6da13cde6eff405138058393b7a9f8d3ede56bf923a6a4` |
| characteristic report | `bdd20f1b9cdfecf6d867cd96b06f1363e8de7d5ac0ebb63fd1c6d9e1b6fe1170` | `9b7f8215743936062c025af199384c528aa6eb0fef6ed11c141687b2b3120f97` |
| final report | `62da14f1602beed0ae8657b47f4261fa138a01b3349913d8f2cc7ea57396fe2e` | `9604c593c81767c19eb7f49017d2867f8978afde55ffa1d98d8073e967499ee0` |
| pre-fix analysis code | `75c8cd39c5863ca5734828ba6b3599b3e36c75bb3e3217e202f27b5778c2b5b9` | `2c7900d44e1549f846992d6669c074893e92eae150cc1758b2287b5b8bc7d73c` |

이 수정은 전향적으로 고정된 최종 endpoint의 값이나 판정을 바꾸지 않았다.
