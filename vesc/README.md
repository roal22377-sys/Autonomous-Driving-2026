# VESC 설정 백업

`vesc6_upenn_foc.xml`: 11/26 17:49 VESC Tool에서 저장한 **변경 전(정상) 모터 설정**. 하드웨어 VESC 6 (HW 60).

- Motor Current Max 30 A / Brake -30 A, Battery 60 / -20 A, **Absolute Max 80 A**, Slow ABS on
- Speed PID Kp 0.006, Ki 0.006, Kd 0.0002, Kd filter 0.1, Min ERPM 500, Allow Braking on

11/26 18:0x: Absolute Max를 120 A로 올린 뒤 주행 중 **모터에서 연기**. 이 파일로 되돌립니다
(VESC Tool → File → Load Motor Configuration XML → Write Motor Configuration).
보호 설정(전류 한계, Absolute Max)은 바꾸지 않습니다. 수업 매뉴얼: Motor Current Max 변경 금지.
자세한 내용: `src/f1tenth/CAR_RESULTS_1126.md` 0절, 2-5절.
