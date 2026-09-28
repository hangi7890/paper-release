# BDG2 Stage 6 C0 LSTM 개발 결과 v3

## 완료 상태

- Source 20개 건물 C0 학습·검증: 완료.
- 개발 12개 건물 Source-only 및 0·1·3·7·14·28일 출력층 적응: 완료.
- 사전 고정 시드: 3/3 완료.
- 실패: 0개. final-test 전력 사용: 아니오.
- 실행 장치: `NVIDIA GeForce RTX 3080`, 실행 시간 503.8초.

## 입력과 데이터 분리

- C0 feature: past_electricity_normalized, target_hour_sin, target_hour_cos, target_weekday_sin, target_weekday_cos, target_weekend.
- 목표 달력 feature는 예측 생성 시 확정되며 미래 관측값이 아니다.
- 정규화 mean=167.065197632, std=354.328669572; Source 학습 X와 y만 사용했다.
- Source 검증만 checkpoint 선택에 사용했다.
- v2 power-only LSTM과 비교해 input size만 1→6으로 바뀌었다.

## Source-only 결과

```
               building_id  site_id  completed_seeds  mae_mean  mae_std  rmse_mean  rmse_std
        Bear_education_Bob     Bear                3 16.716528 1.986553  26.849073  3.200330
       Bear_public_Jocelyn     Bear                3  6.239018 0.298256   8.643489  0.506847
Cockatoo_assembly_Griselda Cockatoo                3  7.277279 0.620592   9.222783  0.449465
 Cockatoo_education_Doreen Cockatoo                3 10.292611 0.523507  13.799354  0.738704
   Fox_education_Henrietta      Fox                3 10.584704 0.251661  15.764096  0.576991
         Fox_lodging_Helen      Fox                3 27.222284 2.039059  36.838353  2.250544
Panther_education_Scarlett  Panther                3  7.224706 1.969290   9.304446  2.220342
 Panther_office_Clementine  Panther                3  8.476786 0.066743  11.382613  0.336788
  Peacock_education_Robbie  Peacock                3  8.517833 0.255611  10.991883  0.208985
       Peacock_lodging_Lou  Peacock                3 14.726429 2.104210  19.426967  2.676715
     Wolf_education_Cheryl     Wolf                3 23.942186 1.323279  33.416809  2.391458
       Wolf_office_Darleen     Wolf                3  8.575191 0.566233  11.469420  0.547410
```

## 출력층 적응 결과

양의 delta는 같은 시드 C0 Source-only보다 악화된 결과이다.

```
               building_id  site_id  adaptation_days  completed_seeds  adaptation_samples  mae_mean  rmse_mean  mae_delta_mean  rmse_delta_mean
        Bear_education_Bob     Bear                0                3                   0 16.716528  26.849073        0.000000         0.000000
        Bear_education_Bob     Bear                1                3                   1 17.604089  28.463431        0.887560         1.614358
        Bear_education_Bob     Bear                3                3                  49 16.697424  26.434251       -0.019104        -0.414822
        Bear_education_Bob     Bear                7                3                 145 14.872687  23.942999       -1.843842        -2.906074
        Bear_education_Bob     Bear               14                3                 313 14.124841  21.783004       -2.591687        -5.066068
        Bear_education_Bob     Bear               28                3                 649 13.288921  20.588329       -3.427607        -6.260744
       Bear_public_Jocelyn     Bear                0                3                   0  6.239018   8.643489        0.000000         0.000000
       Bear_public_Jocelyn     Bear                1                3                   1  5.857864   8.075260       -0.381154        -0.568230
       Bear_public_Jocelyn     Bear                3                3                  49  6.640704   8.417128        0.401686        -0.226361
       Bear_public_Jocelyn     Bear                7                3                 145  4.408745   5.946351       -1.830273        -2.697138
       Bear_public_Jocelyn     Bear               14                3                 313  3.978412   5.526714       -2.260606        -3.116775
       Bear_public_Jocelyn     Bear               28                3                 649  3.723758   5.362683       -2.515260        -3.280806
Cockatoo_assembly_Griselda Cockatoo                0                3                   0  7.277279   9.222783        0.000000         0.000000
Cockatoo_assembly_Griselda Cockatoo                1                3                   1  6.250460   8.002326       -1.026819        -1.220458
Cockatoo_assembly_Griselda Cockatoo                3                3                  49  6.156872   7.858468       -1.120407        -1.364316
Cockatoo_assembly_Griselda Cockatoo                7                3                 145  5.320133   6.767772       -1.957146        -2.455011
Cockatoo_assembly_Griselda Cockatoo               14                3                 313  5.089048   6.511380       -2.188231        -2.711403
Cockatoo_assembly_Griselda Cockatoo               28                3                 649  4.800626   6.219050       -2.476653        -3.003733
 Cockatoo_education_Doreen Cockatoo                0                3                   0 10.292611  13.799354        0.000000         0.000000
 Cockatoo_education_Doreen Cockatoo                1                3                   1 11.262205  15.130192        0.969594         1.330837
 Cockatoo_education_Doreen Cockatoo                3                3                  49  9.306659  12.706377       -0.985951        -1.092978
 Cockatoo_education_Doreen Cockatoo                7                3                 145  9.174541  11.828055       -1.118070        -1.971299
 Cockatoo_education_Doreen Cockatoo               14                3                 313  8.134025  10.420008       -2.158586        -3.379346
 Cockatoo_education_Doreen Cockatoo               28                3                 649  7.512510   9.660351       -2.780101        -4.139003
   Fox_education_Henrietta      Fox                0                3                   0 10.584704  15.764096        0.000000         0.000000
   Fox_education_Henrietta      Fox                1                3                   1 13.542145  17.823299        2.957440         2.059203
   Fox_education_Henrietta      Fox                3                3                  49 14.460991  19.104014        3.876287         3.339918
   Fox_education_Henrietta      Fox                7                3                 145 18.555514  22.182194        7.970810         6.418098
   Fox_education_Henrietta      Fox               14                3                 313 15.887772  20.051365        5.303067         4.287269
   Fox_education_Henrietta      Fox               28                3                 649  8.869132  11.828889       -1.715572        -3.935207
         Fox_lodging_Helen      Fox                0                3                   0 27.222284  36.838353        0.000000         0.000000
         Fox_lodging_Helen      Fox                1                3                   1 28.408134  35.826884        1.185850        -1.011470
         Fox_lodging_Helen      Fox                3                3                  49 24.193939  32.353149       -3.028344        -4.485204
         Fox_lodging_Helen      Fox                7                3                 145 22.600914  30.251480       -4.621370        -6.586873
         Fox_lodging_Helen      Fox               14                3                 313 22.215166  29.830794       -5.007118        -7.007559
         Fox_lodging_Helen      Fox               28                3                 649 22.271045  29.813319       -4.951239        -7.025034
Panther_education_Scarlett  Panther                0                3                   0  7.224706   9.304446        0.000000         0.000000
Panther_education_Scarlett  Panther                1                3                   1  9.221811  11.518462        1.997105         2.214016
Panther_education_Scarlett  Panther                3                3                  49  5.949173   7.983389       -1.275533        -1.321057
Panther_education_Scarlett  Panther                7                3                 145  5.721617   7.433216       -1.503089        -1.871230
Panther_education_Scarlett  Panther               14                3                 313  5.605675   7.253590       -1.619031        -2.050856
Panther_education_Scarlett  Panther               28                3                 649  5.218132   6.837470       -2.006575        -2.466975
 Panther_office_Clementine  Panther                0                3                   0  8.476786  11.382613        0.000000         0.000000
 Panther_office_Clementine  Panther                1                3                   1  9.025284  12.333082        0.548498         0.950469
 Panther_office_Clementine  Panther                3                3                  49  6.960394   9.269842       -1.516393        -2.112771
 Panther_office_Clementine  Panther                7                3                 145  7.298108   8.989507       -1.178678        -2.393106
 Panther_office_Clementine  Panther               14                3                 313  8.190666  10.372848       -0.286120        -1.009765
 Panther_office_Clementine  Panther               28                3                 649  6.099827   7.621137       -2.376960        -3.761476
  Peacock_education_Robbie  Peacock                0                3                   0  8.517833  10.991883        0.000000         0.000000
  Peacock_education_Robbie  Peacock                1                3                   1  8.305446  10.834281       -0.212387        -0.157602
  Peacock_education_Robbie  Peacock                3                3                  49  8.390821  10.556312       -0.127011        -0.435571
  Peacock_education_Robbie  Peacock                7                3                 145  7.564842   9.748951       -0.952991        -1.242931
  Peacock_education_Robbie  Peacock               14                3                 313  7.284155   9.236248       -1.233678        -1.755635
  Peacock_education_Robbie  Peacock               28                3                 649  7.097755   8.890466       -1.420077        -2.101417
       Peacock_lodging_Lou  Peacock                0                3                   0 14.726429  19.426967        0.000000         0.000000
       Peacock_lodging_Lou  Peacock                1                3                   1 18.170563  25.024791        3.444134         5.597823
       Peacock_lodging_Lou  Peacock                3                3                  49 17.369647  21.289721        2.643218         1.862753
       Peacock_lodging_Lou  Peacock                7                3                 145 11.314953  14.646580       -3.411476        -4.780388
       Peacock_lodging_Lou  Peacock               14                3                 313 10.597911  13.702062       -4.128518        -5.724906
       Peacock_lodging_Lou  Peacock               28                3                 649  9.825079  12.775497       -4.901350        -6.651471
     Wolf_education_Cheryl     Wolf                0                3                   0 23.942186  33.416809        0.000000         0.000000
     Wolf_education_Cheryl     Wolf                1                3                   1 30.762024  40.290452        6.819838         6.873644
     Wolf_education_Cheryl     Wolf                3                3                  49 23.531602  32.519784       -0.410584        -0.897025
     Wolf_education_Cheryl     Wolf                7                3                 145 23.249430  32.550784       -0.692756        -0.866025
     Wolf_education_Cheryl     Wolf               14                3                 313 22.201752  29.078874       -1.740434        -4.337934
     Wolf_education_Cheryl     Wolf               28                3                 649 20.971992  27.891451       -2.970195        -5.525358
       Wolf_office_Darleen     Wolf                0                3                   0  8.575191  11.469420        0.000000         0.000000
       Wolf_office_Darleen     Wolf                1                3                   1 12.231101  16.138903        3.655909         4.669483
       Wolf_office_Darleen     Wolf                3                3                  49  7.573123   9.936115       -1.002069        -1.533304
       Wolf_office_Darleen     Wolf                7                3                 145  7.257870   9.601840       -1.317322        -1.867579
       Wolf_office_Darleen     Wolf               14                3                 313  7.221943   9.545320       -1.353248        -1.924100
       Wolf_office_Darleen     Wolf               28                3                 649  6.912123   9.218218       -1.663069        -2.251202
```

이 결과는 개발용 C0 예비 실험이며 C1 또는 최종 시험 결과가 아니다.
