# model comparison (2026-11-26 13:30, 2 lap(s), reference = ppo_s2r/2130)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter | real course (caps 8/4/2) | real 2-lap time, cap 8 |
|---|---|---|---|---|---|---|---|---|---|
| ppo_s2r/2130 | 99.2% (393/396) | 100.0% (132/132) | 82.6% (109/132) | 94.3% (249/264) | reference | 5.25 m/s | 0.0044 rad | 99.4% (161/162) | 30.9 s |
| ppo_s2r/2420 | 100.0% (396/396) | 100.0% (132/132) | 81.8% (108/132) | 94.3% (249/264) | 0.5% slower on 261 runs | 5.24 m/s | 0.0044 rad | 99.4% (161/162) | 30.9 s |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | ppo_s2r/2130 | ppo_s2r/2420 |
|---|---|---|
| Catalunya | 143.7 | 144.2 |
| Catalunya_rev | 143.4 | 144.0 |
| Catalunya_rev_w105 | 142.9 | 144.5 |
| Catalunya_w105 | 145.7 | 145.8 |
| IMS | 84.3 | 84.8 |
| IMS_rev | 86.7 | 86.4 |
| IMS_rev_w105 | 86.8 | 86.3 |
| IMS_w105 | 84.1 | 85.7 |
| MexicoCity | 127.6 | 128.3 |
| MexicoCity_rev | 127.2 | 127.7 |
| MexicoCity_rev_w105 | 127.0 | 128.6 |
| MexicoCity_w105 | 129.3 | 130.0 |
| Montreal | 100.6 | 101.0 |
| Montreal_rev | 100.2 | 100.7 |
| Montreal_rev_w105 | 101.3 | 101.9 |
| Montreal_w105 | 99.8 | 100.9 |
| MoscowRaceway | 123.1 | 123.1 |
| MoscowRaceway_rev | 123.6 | 124.1 |
| MoscowRaceway_rev_w105 | 122.0 | 122.6 |
| MoscowRaceway_w105 | 123.3 | 124.3 |
| SaoPaulo | 119.7 | 120.2 |
| SaoPaulo_rev | 121.5 | 121.5 |
| SaoPaulo_rev_w105 | 119.6 | 120.0 |
| SaoPaulo_w105 | 118.8 | 120.1 |
| Shanghai | 174.0 | 174.4 |
| Shanghai_rev | 172.3 | 173.4 |
| Shanghai_rev_w105 | 172.8 | 174.8 |
| Shanghai_w105 | 172.9 | 174.0 |
| Silverstone | 153.1 | 153.4 |
| Silverstone_rev | 152.4 | 154.1 |
| Silverstone_rev_w105 | 154.0 | 155.6 |
| Silverstone_w105 | 154.9 | 155.7 |
| Sochi | 157.2 | 158.0 |
| Sochi_rev | 157.4 | 157.5 |
| Sochi_rev_w105 | 157.3 | 158.7 |
| Sochi_w105 | 158.9 | 159.8 |
| YasMarina | 146.4 | 146.9 |
| YasMarina_rev | 147.1 | 148.3 |
| YasMarina_rev_w105 | 148.6 | 149.0 |
| YasMarina_w105 | 146.3 | 147.5 |
| Zandvoort | 133.8 | 133.8 |
| Zandvoort_rev | 133.7 | 134.4 |
| Zandvoort_rev_w105 | 132.8 | 133.8 |
| Zandvoort_w105 | 133.5 | 133.5 |

## failures (82)

- ppo_s2r/2130 worst Catalunya_rev_w105@8/center at 5.9 s
- ppo_s2r/2130 worst Montreal_rev@8/center at 14.1 s
- ppo_s2r/2130 worst MoscowRaceway@8/center at 108.1 s
- ppo_s2r/2130 worst MoscowRaceway_rev_w105@8/center at 51.3 s
- ppo_s2r/2130 worst SaoPaulo_w105@8/center at 53.6 s
- ppo_s2r/2130 worst Silverstone@8/center at 81.5 s
- ppo_s2r/2130 worst Silverstone_w105@8/center at 53.9 s
- ppo_s2r/2130 open Catalunya_rev_w902@8/center at 8.7 s
- ppo_s2r/2130 open MexicoCity_rev_w902@8/center at 66.3 s
- ppo_s2r/2130 open Montreal_rev_w902@8/center at 32.1 s
- ppo_s2r/2130 open Montreal_w902@8/center at 60.5 s
- ppo_s2r/2130 open Montreal_rev_w902@5/center at 40.0 s
- ppo_s2r/2130 open Montreal_w902@5/center at 14.9 s
- ppo_s2r/2130 real Practice1124_rev@8/center at 10.6 s
- ppo_s2r/2130 eval Shanghai_rev_w105@8/center#1 at 16.7 s
- ppo_s2r/2130 eval Shanghai_rev_w105@8/random#1 at 16.6 s
- ppo_s2r/2130 worst Catalunya@8/center#1 at 5.4 s
- ppo_s2r/2130 worst MexicoCity_rev_w105@8/center#1 at 32.1 s
- ppo_s2r/2130 worst Montreal_rev@8/center#1 at 59.5 s
- ppo_s2r/2130 worst MoscowRaceway_rev_w105@8/center#1 at 7.1 s
- ppo_s2r/2130 worst MoscowRaceway_w105@8/center#1 at 121.7 s
- ppo_s2r/2130 worst SaoPaulo@8/center#1 at 49.0 s
- ppo_s2r/2130 worst SaoPaulo_w105@8/center#1 at 51.9 s
- ppo_s2r/2130 worst Sochi_w105@8/center#1 at 1.8 s
- ppo_s2r/2130 worst Zandvoort_w105@8/center#1 at 126.8 s
- ppo_s2r/2130 open MexicoCity_rev_w901@8/center#1 at 78.3 s
- ppo_s2r/2130 open MexicoCity_rev_w902@8/center#1 at 60.6 s
- ppo_s2r/2130 open Montreal_w902@8/center#1 at 75.6 s
- ppo_s2r/2130 open Montreal_rev_w902@5/center#1 at 27.0 s
- ppo_s2r/2130 open Montreal_w902@5/center#1 at 34.9 s
- ppo_s2r/2130 eval Montreal_rev@8/random#2 at 36.1 s
- ppo_s2r/2130 worst MexicoCity_rev_w105@8/center#2 at 1.9 s
- ppo_s2r/2130 worst Montreal_w105@8/center#2 at 105.6 s
- ppo_s2r/2130 worst MoscowRaceway_rev_w105@8/center#2 at 41.2 s
- ppo_s2r/2130 worst SaoPaulo_w105@8/center#2 at 50.0 s
- ppo_s2r/2130 worst Sochi_rev_w105@8/center#2 at 15.7 s
- ppo_s2r/2130 worst YasMarina_rev@8/center#2 at 54.2 s
- ppo_s2r/2130 worst YasMarina_rev_w105@8/center#2 at 1.9 s
- ppo_s2r/2130 open Catalunya_rev_w901@8/center#2 at 67.7 s
- ppo_s2r/2130 open Montreal_w902@8/center#2 at 98.2 s
- ppo_s2r/2130 open Montreal_rev_w902@5/center#2 at 61.0 s
- ppo_s2r/2130 open Montreal_w902@5/center#2 at 62.5 s
- ppo_s2r/2420 worst Montreal_rev@8/center at 14.0 s
- ppo_s2r/2420 worst MoscowRaceway@8/center at 108.8 s
- ppo_s2r/2420 worst MoscowRaceway_rev_w105@8/center at 119.1 s
- ppo_s2r/2420 worst Shanghai_w105@8/center at 107.4 s
- ppo_s2r/2420 worst Silverstone@8/center at 82.2 s
- ppo_s2r/2420 worst Silverstone_rev_w105@8/center at 135.3 s
- ppo_s2r/2420 worst Sochi_rev_w105@8/center at 87.1 s
- ppo_s2r/2420 worst Zandvoort_w105@8/center at 55.3 s
- ppo_s2r/2420 open MexicoCity_rev_w902@8/center at 92.7 s
- ppo_s2r/2420 open Montreal_rev_w902@8/center at 32.2 s
- ppo_s2r/2420 open Montreal_w902@8/center at 61.2 s
- ppo_s2r/2420 open Montreal_rev_w902@5/center at 40.0 s
- ppo_s2r/2420 open Montreal_w902@5/center at 14.9 s
- ppo_s2r/2420 real Practice1124_rev@8/center at 10.7 s
- ppo_s2r/2420 worst MexicoCity_rev_w105@8/center#1 at 101.6 s
- ppo_s2r/2420 worst Montreal_rev@8/center#1 at 86.3 s
- ppo_s2r/2420 worst MoscowRaceway_rev_w105@8/center#1 at 74.5 s
- ppo_s2r/2420 worst SaoPaulo_w105@8/center#1 at 30.4 s
- ppo_s2r/2420 worst Shanghai_rev@8/center#1 at 157.9 s
- ppo_s2r/2420 worst Silverstone_rev@8/center#1 at 25.6 s
- ppo_s2r/2420 worst Sochi_w105@8/center#1 at 1.8 s
- ppo_s2r/2420 worst Zandvoort_w105@8/center#1 at 53.0 s
- ppo_s2r/2420 open Montreal_rev_w902@8/center#1 at 21.7 s
- ppo_s2r/2420 open Montreal_w902@8/center#1 at 26.1 s
- ppo_s2r/2420 open Shanghai_rev_w902@8/center#1 at 124.5 s
- ppo_s2r/2420 open Montreal_rev_w902@5/center#1 at 27.0 s
- ppo_s2r/2420 open Montreal_w902@5/center#1 at 34.8 s
- ppo_s2r/2420 worst MexicoCity_rev_w105@8/center#2 at 1.8 s
- ppo_s2r/2420 worst MexicoCity_w105@8/center#2 at 85.0 s
- ppo_s2r/2420 worst Montreal_w105@8/center#2 at 105.9 s
- ppo_s2r/2420 worst MoscowRaceway_rev_w105@8/center#2 at 41.1 s
- ppo_s2r/2420 worst SaoPaulo@8/center#2 at 64.9 s
- ppo_s2r/2420 worst SaoPaulo_w105@8/center#2 at 107.0 s
- ppo_s2r/2420 worst Shanghai_rev_w105@8/center#2 at 53.2 s
- ppo_s2r/2420 worst YasMarina_rev_w105@8/center#2 at 2.0 s
- ppo_s2r/2420 open Montreal_rev_w902@8/center#2 at 98.6 s
- ppo_s2r/2420 open Montreal_w902@8/center#2 at 48.6 s
- ppo_s2r/2420 open Shanghai_rev_w902@8/center#2 at 46.9 s
- ppo_s2r/2420 open Montreal_rev_w902@5/center#2 at 60.9 s
- ppo_s2r/2420 open Montreal_w902@5/center#2 at 62.5 s
