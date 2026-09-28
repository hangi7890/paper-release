# BDG2 다건물 개발 실험 선정 및 시간 분할 v2

## 고정 결과

- Source 출발 학습 건물: **20개**, 4개 사이트에서 각 5개.
- 개발·검증 건물: **12개**, 6개 사이트에서 각 2개.
- 최종 시험 사이트: ID와 기존 `alternative_group` 배정만 새 cohort manifest에 보존했다. 해당 전력 열은 읽지 않았다.
- 선정과 시간 분할은 모델 실행 전에 저장했으며 모델 성능을 사용하지 않았다.

## 사전 선정 기준

1. 기존 `alternative_group`이 Source 또는 개발 그룹일 것.
2. 기존 `preliminary_eligible=True`, 숫자 기록 부재·전부 0 건물이 아닐 것.
3. 보고된 최장 연속 유효 기록이 90일 이상일 것.
4. 30일 이상 연속 0값 검토 대상이 아닐 것.
5. 음수값이 0개이고 전체 결측률이 20% 이하일 것.
6. 사이트별 고정 할당량 안에서 용도별 현재 선택 수가 적은 후보를 우선하고, 동률이면 결측률 오름차순, 최장 연속 기록 내림차순, 0 비율 오름차순, 용도·건물 ID 순으로 선택했다.
7. 선택 후 raw의 해당 32개 열만 읽어 실제 유한값·1시간 연속 최장 구간이 90일 이상인지 다시 확인했다. 실패 시 대체 선정 없이 중단하도록 구현했다.

## 사이트별 선택 수

```
alternative_group                site_id 
development_validation_proposal  Bear        2
                                 Cockatoo    2
                                 Fox         2
                                 Panther     2
                                 Peacock     2
                                 Wolf        2
source_train_proposal            Crow        5
                                 Lamb        5
                                 Rat         5
                                 Robin       5
```

## 용도별 선택 수

```
alternative_group                primaryspaceusage            
development_validation_proposal  Education                        6
                                 Entertainment/public assembly    1
                                 Lodging/residential              2
                                 Office                           2
                                 Public services                  1
source_train_proposal            Education                        8
                                 Entertainment/public assembly    2
                                 Healthcare                       1
                                 Lodging/residential              1
                                 Manufacturing/industrial         1
                                 Office                           3
                                 Public services                  2
                                 Retail                           1
                                 Warehouse/storage                1
```

## 시간 분할 규칙

- Source: 각 건물의 실제 최장 연속 구간을 사용한다. 마지막 28일 정답을 Source 검증으로 고정하고, 그 이전 완전한 24→24시간 창만 Source 학습에 쓴다.
- 개발: 각 건물 최장 연속 구간의 마지막 57일을 고정한다. 첫 24시간은 이력, 다음 28일은 최대 적응 구간, 마지막 28일은 하루 간격 28개 평가 창이다.
- 따라서 개발 건물마다 달력 날짜는 다를 수 있지만, 각 건물의 모든 적응 조건은 동일한 고정 시험 기간을 사용한다.
- 0·1·3·7·14·28일 적응 샘플 수는 각각 0·1·49·145·313·649개이다.
- 결측 보간과 시간대 변환은 수행하지 않았다.

## 개발 건물별 고정 평가 기간

```
               building_id  site_id   test_target_start     test_target_end  actual_longest_run_hours
        Bear_education_Bob     Bear 2017-02-12 02:00:00 2017-03-12 01:00:00                      8735
       Bear_public_Jocelyn     Bear 2017-02-12 02:00:00 2017-03-12 01:00:00                      8735
Cockatoo_assembly_Griselda Cockatoo 2016-09-27 12:00:00 2016-10-25 11:00:00                      5030
 Cockatoo_education_Doreen Cockatoo 2016-09-27 12:00:00 2016-10-25 11:00:00                      5030
   Fox_education_Henrietta      Fox 2017-12-04 00:00:00 2017-12-31 23:00:00                     17544
         Fox_lodging_Helen      Fox 2017-12-04 00:00:00 2017-12-31 23:00:00                     15559
Panther_education_Scarlett  Panther 2017-01-17 20:00:00 2017-02-14 19:00:00                      9860
 Panther_office_Clementine  Panther 2017-01-17 20:00:00 2017-02-14 19:00:00                      9860
     Wolf_education_Cheryl     Wolf 2017-10-01 09:00:00 2017-10-29 08:00:00                     16017
       Wolf_office_Darleen     Wolf 2017-10-01 09:00:00 2017-10-29 08:00:00                     16017
  Peacock_education_Robbie  Peacock 2017-02-12 02:00:00 2017-03-12 01:00:00                      8734
       Peacock_lodging_Lou  Peacock 2017-02-12 02:00:00 2017-03-12 01:00:00                      8734
```

## 고정 파일 해시

- cohort manifest SHA-256: `59f4128ebd0a594e60c410c8ae6af48fdd674422553d9e47e4c771195c0bf23b`
- temporal manifest SHA-256: `6dd57dadf2aaae9e565a870784d2ced3482c1466156e1c691a3149bee3ff8f1b`
- 입력 alternative cohort SHA-256: `6c77b0d5b07d0c1a16279a5100cb7ef092526265afc55734778e42fdc74ff902`
- raw electricity SHA-256: `039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418`

이 선정은 개발용 다건물 예비 실험을 위한 것이며 최종 시험 사이트 또는 논문 최종 코호트를 확정하지 않는다.
