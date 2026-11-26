# model comparison (2026-11-26 10:56, 2 lap(s), reference = imitation/13)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter | real course (caps 8/4/2) | real 2-lap time, cap 8 |
|---|---|---|---|---|---|---|---|---|---|
| imitation/13 | 100.0% (396/396) | 100.0% (132/132) | 78.0% (103/132) | 94.3% (249/264) | reference | 5.30 m/s | 0.0041 rad | 97.5% (158/162) | 31.0 s |
| ppo_s2r/2130 | 99.2% (393/396) | 100.0% (132/132) | 82.6% (109/132) | 94.3% (249/264) | 0.6% slower on 261 runs | 5.25 m/s | 0.0044 rad | 99.4% (161/162) | 30.9 s |
| ppo_s2r/2260 | 100.0% (396/396) | 100.0% (132/132) | 81.8% (108/132) | 91.3% (241/264) | 0.1% faster on 264 runs | 5.30 m/s | 0.0045 rad | 97.5% (158/162) | 30.8 s |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | imitation/13 | ppo_s2r/2130 | ppo_s2r/2260 |
|---|---|---|---|
| Catalunya | 143.4 | 143.7 | 143.9 |
| Catalunya_rev | 143.0 | 143.4 | 143.7 |
| Catalunya_rev_w105 | 141.9 | 142.9 | 141.6 |
| Catalunya_w105 | 144.4 | 145.7 | 143.8 |
| IMS | 84.1 | 84.3 | 84.3 |
| IMS_rev | 85.8 | 86.7 | 84.1 |
| IMS_rev_w105 | 86.0 | 86.8 | 83.6 |
| IMS_w105 | 84.5 | 84.1 | 83.8 |
| MexicoCity | 126.4 | 127.6 | 128.2 |
| MexicoCity_rev | 126.2 | 127.2 | 126.5 |
| MexicoCity_rev_w105 | 126.7 | 127.0 | 126.0 |
| MexicoCity_w105 | 128.0 | 129.3 | 128.3 |
| Montreal | 99.7 | 100.6 | 100.0 |
| Montreal_rev | 100.7 | 100.2 | 100.3 |
| Montreal_rev_w105 | 100.7 | 101.3 | 100.0 |
| Montreal_w105 | 98.8 | 99.8 | 98.3 |
| MoscowRaceway | 122.5 | 123.1 | 123.3 |
| MoscowRaceway_rev | 122.9 | 123.6 | 124.3 |
| MoscowRaceway_rev_w105 | 120.7 | 122.0 | 121.6 |
| MoscowRaceway_w105 | 122.6 | 123.3 | 122.5 |
| SaoPaulo | 119.4 | 119.7 | 119.9 |
| SaoPaulo_rev | 120.8 | 121.5 | 121.4 |
| SaoPaulo_rev_w105 | 118.8 | 119.6 | 118.3 |
| SaoPaulo_w105 | 118.5 | 118.8 | 118.1 |
| Shanghai | 172.4 | 174.0 | 172.3 |
| Shanghai_rev | 171.3 | 172.3 | 171.8 |
| Shanghai_rev_w105 | 171.7 | 172.8 | 171.7 |
| Shanghai_w105 | 170.6 | 172.9 | 169.6 |
| Silverstone | 152.3 | 153.1 | 153.4 |
| Silverstone_rev | 152.0 | 152.4 | 152.9 |
| Silverstone_rev_w105 | 153.4 | 154.0 | 152.1 |
| Silverstone_w105 | 154.3 | 154.9 | 153.4 |
| Sochi | 156.2 | 157.2 | 156.8 |
| Sochi_rev | 157.0 | 157.4 | 157.3 |
| Sochi_rev_w105 | 156.5 | 157.3 | 154.6 |
| Sochi_w105 | 157.1 | 158.9 | 156.4 |
| YasMarina | 145.2 | 146.4 | 145.6 |
| YasMarina_rev | 146.0 | 147.1 | 147.1 |
| YasMarina_rev_w105 | 146.0 | 148.6 | 146.4 |
| YasMarina_w105 | 146.1 | 146.3 | 144.5 |
| Zandvoort | 133.5 | 133.8 | 134.3 |
| Zandvoort_rev | 133.6 | 133.7 | 133.7 |
| Zandvoort_rev_w105 | 132.4 | 132.8 | 132.1 |
| Zandvoort_w105 | 132.8 | 133.5 | 132.3 |

## failures (141)

- imitation/13 worst Montreal_rev@8/center at 13.9 s
- imitation/13 worst Montreal_rev_w105@8/center at 45.0 s
- imitation/13 worst MoscowRaceway@8/center at 108.2 s
- imitation/13 worst MoscowRaceway_rev_w105@8/center at 51.0 s
- imitation/13 worst MoscowRaceway_w105@8/center at 10.9 s
- imitation/13 worst Shanghai_w105@8/center at 64.6 s
- imitation/13 worst Sochi_w105@8/center at 165.4 s
- imitation/13 worst YasMarina_w105@8/center at 16.8 s
- imitation/13 worst Zandvoort_w105@8/center at 54.8 s
- imitation/13 open MexicoCity_rev_w902@8/center at 66.0 s
- imitation/13 open Montreal_w902@8/center at 60.1 s
- imitation/13 open Montreal_rev_w902@5/center at 39.4 s
- imitation/13 open Montreal_w902@5/center at 14.6 s
- imitation/13 open Shanghai_rev_w901@5/center at 37.4 s
- imitation/13 real Practice1124_rev@8/center at 10.3 s
- imitation/13 real Rand305_s85_w305@2/center at 37.9 s
- imitation/13 worst Catalunya@8/center#1 at 5.4 s
- imitation/13 worst Montreal_rev@8/center#1 at 10.5 s
- imitation/13 worst Montreal_rev_w105@8/center#1 at 106.0 s
- imitation/13 worst MoscowRaceway@8/center#1 at 127.5 s
- imitation/13 worst MoscowRaceway_rev_w105@8/center#1 at 7.0 s
- imitation/13 worst SaoPaulo_w105@8/center#1 at 51.9 s
- imitation/13 worst Shanghai@8/center#1 at 128.1 s
- imitation/13 worst Silverstone_rev_w105@8/center#1 at 94.3 s
- imitation/13 worst Sochi_w105@8/center#1 at 1.9 s
- imitation/13 worst YasMarina_w105@8/center#1 at 149.5 s
- imitation/13 worst Zandvoort_w105@8/center#1 at 28.9 s
- imitation/13 open MexicoCity_rev_w902@8/center#1 at 34.3 s
- imitation/13 open Montreal_w902@8/center#1 at 75.4 s
- imitation/13 open Shanghai_w902@8/center#1 at 26.7 s
- imitation/13 open Zandvoort_rev_w902@8/center#1 at 134.0 s
- imitation/13 open Montreal_rev_w902@5/center#1 at 26.6 s
- imitation/13 open Montreal_w902@5/center#1 at 34.1 s
- imitation/13 open Shanghai_w902@5/center#1 at 147.4 s
- imitation/13 worst Montreal_rev_w105@8/center#2 at 10.4 s
- imitation/13 worst MoscowRaceway@8/center#2 at 128.2 s
- imitation/13 worst MoscowRaceway_rev_w105@8/center#2 at 41.2 s
- imitation/13 worst SaoPaulo@8/center#2 at 105.1 s
- imitation/13 worst SaoPaulo_w105@8/center#2 at 49.9 s
- imitation/13 worst Shanghai@8/center#2 at 8.9 s
- imitation/13 worst Silverstone_w105@8/center#2 at 78.2 s
- imitation/13 worst Sochi@8/center#2 at 62.9 s
- imitation/13 worst Sochi_w105@8/center#2 at 70.7 s
- imitation/13 open MexicoCity_rev_w902@8/center#2 at 68.8 s
- imitation/13 open Montreal_rev_w902@5/center#2 at 59.9 s
- imitation/13 open Montreal_w902@5/center#2 at 63.0 s
- imitation/13 real Practice1124_rev@8/center#2 at 18.6 s
- imitation/13 real Rand305_s85_w305@2/center#2 at 64.6 s
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
- ppo_s2r/2260 worst Catalunya@8/center at 72.6 s
- ppo_s2r/2260 worst Catalunya_w105@8/center at 47.7 s
- ppo_s2r/2260 worst MexicoCity_rev_w105@8/center at 81.5 s
- ppo_s2r/2260 worst MoscowRaceway_w105@8/center at 121.4 s
- ppo_s2r/2260 worst SaoPaulo@8/center at 50.5 s
- ppo_s2r/2260 worst Silverstone_w105@8/center at 63.7 s
- ppo_s2r/2260 worst Zandvoort_w105@8/center at 54.8 s
- ppo_s2r/2260 open MexicoCity_rev_w901@8/center at 64.5 s
- ppo_s2r/2260 open MexicoCity_rev_w902@8/center at 92.5 s
- ppo_s2r/2260 open Shanghai_rev_w901@8/center at 115.6 s
- ppo_s2r/2260 open Shanghai_w902@8/center at 156.8 s
- ppo_s2r/2260 open Sochi_rev_w901@8/center at 118.9 s
- ppo_s2r/2260 open YasMarina_rev_w901@8/center at 84.0 s
- ppo_s2r/2260 open Montreal_rev_w902@5/center at 39.6 s
- ppo_s2r/2260 open Shanghai_rev_w901@5/center at 37.8 s
- ppo_s2r/2260 real Practice1124_rev@8/center at 10.8 s
- ppo_s2r/2260 real Rand305_s85_w305@2/center at 37.9 s
- ppo_s2r/2260 worst Catalunya@8/center#1 at 5.5 s
- ppo_s2r/2260 worst Catalunya_w105@8/center#1 at 88.4 s
- ppo_s2r/2260 worst MoscowRaceway_rev_w105@8/center#1 at 7.4 s
- ppo_s2r/2260 worst Silverstone@8/center#1 at 163.9 s
- ppo_s2r/2260 worst Sochi_w105@8/center#1 at 1.8 s
- ppo_s2r/2260 worst Zandvoort_w105@8/center#1 at 53.1 s
- ppo_s2r/2260 open MexicoCity_rev_w902@8/center#1 at 34.7 s
- ppo_s2r/2260 open Montreal_w902@8/center#1 at 25.4 s
- ppo_s2r/2260 open Shanghai_rev_w901@8/center#1 at 73.2 s
- ppo_s2r/2260 open Shanghai_w902@8/center#1 at 114.3 s
- ppo_s2r/2260 open Silverstone_w901@8/center#1 at 94.1 s
- ppo_s2r/2260 open Montreal_rev_w902@5/center#1 at 26.7 s
- ppo_s2r/2260 open Montreal_w902@5/center#1 at 34.0 s
- ppo_s2r/2260 open Shanghai_rev_w901@5/center#1 at 92.5 s
- ppo_s2r/2260 real Rand305_s85_w305@2/center#1 at 45.7 s
- ppo_s2r/2260 worst Catalunya@8/center#2 at 102.8 s
- ppo_s2r/2260 worst Catalunya_w105@8/center#2 at 71.4 s
- ppo_s2r/2260 worst MexicoCity_rev_w105@8/center#2 at 44.7 s
- ppo_s2r/2260 worst MexicoCity_w105@8/center#2 at 84.7 s
- ppo_s2r/2260 worst Montreal_rev_w105@8/center#2 at 83.9 s
- ppo_s2r/2260 worst Montreal_w105@8/center#2 at 51.4 s
- ppo_s2r/2260 worst MoscowRaceway_rev_w105@8/center#2 at 41.4 s
- ppo_s2r/2260 worst SaoPaulo_w105@8/center#2 at 94.7 s
- ppo_s2r/2260 worst Shanghai_rev_w105@8/center#2 at 118.3 s
- ppo_s2r/2260 worst Silverstone_w105@8/center#2 at 78.8 s
- ppo_s2r/2260 worst Sochi_w105@8/center#2 at 1.8 s
- ppo_s2r/2260 open MexicoCity_rev_w902@8/center#2 at 6.0 s
- ppo_s2r/2260 open Montreal_w902@8/center#2 at 49.0 s
- ppo_s2r/2260 open Shanghai_rev_w901@8/center#2 at 165.4 s
- ppo_s2r/2260 open Shanghai_w902@8/center#2 at 46.5 s
- ppo_s2r/2260 open Montreal_rev_w902@5/center#2 at 60.3 s
- ppo_s2r/2260 open Montreal_w902@5/center#2 at 62.9 s
- ppo_s2r/2260 open Shanghai_rev_w901@5/center#2 at 105.4 s
- ppo_s2r/2260 real Rand305_s85_w305@2/center#2 at 22.2 s
