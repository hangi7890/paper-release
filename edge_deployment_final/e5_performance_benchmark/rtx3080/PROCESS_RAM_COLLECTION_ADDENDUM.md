# RTX process RAM collection addendum

## 발견된 문제

최초 E5 실행의 `memory_summary.json`에서 RTX 프로세스의 `rss_current_bytes`와 `rss_peak_bytes`가 `null`로 저장되었다. Windows `GetCurrentProcess` 및 `GetProcessMemoryInfo` 호출에 64비트 `restype`과 `argtypes`를 명시하지 않아 API 호출이 실패한 것이 원인이다.

## 영향 범위

- 영향 있음: RTX process current RSS, process peak RSS
- 영향 없음: 모든 raw latency, 지연시간 통계, CUDA allocated/reserved 및 peak 값, telemetry, pre/post numerical consistency, 모델·체크포인트
- 최초 `memory_summary.json`은 수정하지 않았다.
- 성능 결과를 확인한 뒤 E5 benchmark를 재실행하지 않았다.
- 별도 probe에서 확인한 수정 코드의 RSS 값은 benchmark 중 RSS로 대체하지 않는다.

## 코드 보존과 수정

- 실제 E5 실행 코드: `benchmark_fp32_executed_v1.py`
- 실제 실행 코드 SHA-256: `6d069a5411fcc98c3557c36e6aabe96df13b414af217c9eac55f91f188d0d118`
- 수정 코드: `benchmark_fp32.py`
- 수정 코드 SHA-256: `b3f6b44935e3a5e0ed32111a174fb076e9dda8ec5e13aa835f28a793445a41d`

수정본은 `ctypes.WinDLL`의 반환형과 인수형을 명시한다. 수정 후 별도 비성능 probe에서 API 호출이 성공하고 유한한 current/peak RSS가 반환됨을 확인했다. 이 probe 값은 E5 성능 결과가 아니다.

## 보고 원칙

RTX process RAM은 E5 결과에서 **측정 불가(collector error)** 로 보고한다. 다른 메모리 값이나 사후 probe 값으로 보정하지 않는다.

