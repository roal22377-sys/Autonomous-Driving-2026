# model comparison (2026-11-26 05:15, 2 lap(s), reference = imitation/13)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter | real course (caps 8/4/2) | real 2-lap time, cap 8 |
|---|---|---|---|---|---|---|---|---|---|
| imitation/13 | 100.0% (396/396) | 100.0% (132/132) | 78.0% (103/132) | 94.3% (249/264) | reference | 5.30 m/s | 0.0041 rad | 97.5% (158/162) | 31.0 s |
| imitation/1319 | 100.0% (396/396) | 100.0% (132/132) | 78.8% (104/132) | 91.7% (242/264) | 0.2% faster on 264 runs | 5.31 m/s | 0.0042 rad | 96.9% (157/162) | 30.8 s |
| ppo_s2r/1920 | 99.2% (393/396) | 99.2% (131/132) | 75.8% (100/132) | 88.3% (233/264) | 0.2% faster on 261 runs | 5.31 m/s | 0.0042 rad | 95.7% (155/162) | 31.0 s |
| ppo_s2r/2020 | 99.5% (394/396) | 100.0% (132/132) | 77.3% (102/132) | 87.5% (231/264) | 0.3% slower on 262 runs | 5.29 m/s | 0.0043 rad | 96.3% (156/162) | 31.0 s |
| ppo_s2r/1730 | 95.5% (378/396) | 97.7% (129/132) | 85.6% (113/132) | 87.9% (232/264) | 3.5% slower on 249 runs | 5.09 m/s | 0.0064 rad | 98.1% (159/162) | 31.4 s |
| ppo_s2r/1640 | 97.7% (387/396) | 99.2% (131/132) | 91.7% (121/132) | 91.3% (241/264) | 8.1% slower on 255 runs | 4.96 m/s | 0.0060 rad | 93.8% (152/162) | 32.7 s |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | imitation/13 | imitation/1319 | ppo_s2r/1920 | ppo_s2r/2020 | ppo_s2r/1730 | ppo_s2r/1640 |
|---|---|---|---|---|---|---|
| Catalunya | 143.4 | 142.9 | 142.8 | 143.8 | 146.9 | 154.8 |
| Catalunya_rev | 143.0 | 142.7 | 142.3 | 143.1 | 148.9 | 154.6 |
| Catalunya_rev_w105 | 141.9 | 141.7 | 141.2 | 142.5 | 148.0 | 152.6 |
| Catalunya_w105 | 144.4 | 144.2 | 144.4 | 144.9 | 147.7 | 152.9 |
| IMS | 84.1 | 84.2 | 84.0 | 84.5 | 86.0 | 91.5 |
| IMS_rev | 85.8 | 85.7 | 86.3 | 87.9 | 86.6 | 94.3 |
| IMS_rev_w105 | 86.0 | 85.7 | 87.0 | 87.9 | 86.2 | 91.5 |
| IMS_w105 | 84.5 | 84.4 | 84.3 | 84.8 | 88.6 | 91.0 |
| MexicoCity | 126.4 | 126.2 | 126.1 | 127.5 | 133.4 | 137.7 |
| MexicoCity_rev | 126.2 | 125.9 | 125.6 | 126.3 | 131.5 | 138.6 |
| MexicoCity_rev_w105 | 126.7 | 126.4 | 125.6 | 126.3 | 136.0 | 136.9 |
| MexicoCity_w105 | 128.0 | 128.0 | 127.7 | 127.6 | 134.9 | 138.3 |
| Montreal | 99.7 | 99.6 | 99.8 | 100.5 | 103.8 | 109.6 |
| Montreal_rev | 100.7 | 99.8 | 99.7 | 100.5 | 103.5 | 108.6 |
| Montreal_rev_w105 | 100.7 | 100.3 | 99.9 | 100.8 | 103.4 | 108.1 |
| Montreal_w105 | 98.8 | 98.7 | 98.9 | 98.7 | 102.3 | 104.9 |
| MoscowRaceway | 122.5 | 122.3 | 122.0 | 122.3 | 127.0 | 133.7 |
| MoscowRaceway_rev | 122.9 | 122.6 | 122.6 | 123.5 | 126.8 | 134.6 |
| MoscowRaceway_rev_w105 | 120.7 | 120.5 | 120.6 | 121.1 | 125.1 | 129.6 |
| MoscowRaceway_w105 | 122.6 | 122.8 | 122.7 | 123.1 | 128.0 | 134.9 |
| SaoPaulo | 119.4 | 119.5 | 118.8 | 119.3 | 124.3 | 130.7 |
| SaoPaulo_rev | 120.8 | 120.9 | 120.7 | 121.9 | 123.4 | 128.6 |
| SaoPaulo_rev_w105 | 118.8 | 118.7 | 119.0 | 119.3 | 121.8 | 126.5 |
| SaoPaulo_w105 | 118.5 | 118.4 | 117.5 | 118.2 | 123.6 | 127.6 |
| Shanghai | 172.4 | 172.0 | 171.2 | 172.5 | 181.2 | - |
| Shanghai_rev | 171.3 | 171.2 | 169.8 | 171.4 | 176.3 | 186.0 |
| Shanghai_rev_w105 | 171.7 | 171.4 | 171.6 | 171.7 | 178.1 | 184.3 |
| Shanghai_w105 | 170.6 | 170.4 | 170.9 | 171.1 | 177.3 | - |
| Silverstone | 152.3 | 152.2 | - | 153.6 | 156.0 | 164.3 |
| Silverstone_rev | 152.0 | 151.8 | 151.3 | 152.5 | 156.5 | 166.1 |
| Silverstone_rev_w105 | 153.4 | 152.9 | 152.4 | 153.1 | 157.9 | 167.1 |
| Silverstone_w105 | 154.3 | 153.7 | 153.9 | 154.4 | 158.0 | 163.7 |
| Sochi | 156.2 | 156.1 | 156.8 | 157.7 | 159.8 | 169.2 |
| Sochi_rev | 157.0 | 157.0 | 156.3 | 156.4 | 160.9 | 172.2 |
| Sochi_rev_w105 | 156.5 | 156.0 | 155.8 | 155.7 | 160.8 | 169.3 |
| Sochi_w105 | 157.1 | 156.8 | 157.3 | 157.4 | 161.0 | 170.2 |
| YasMarina | 145.2 | 144.7 | 144.3 | 145.0 | 152.6 | 158.1 |
| YasMarina_rev | 146.0 | 145.5 | 145.4 | 146.9 | 154.7 | 160.1 |
| YasMarina_rev_w105 | 146.0 | 145.5 | 145.7 | 146.1 | 154.2 | 157.1 |
| YasMarina_w105 | 146.1 | 145.9 | 144.6 | 146.0 | - | 156.5 |
| Zandvoort | 133.5 | 132.9 | 133.8 | 135.1 | 137.9 | 144.1 |
| Zandvoort_rev | 133.6 | 133.4 | 133.1 | 133.7 | 138.9 | 145.2 |
| Zandvoort_rev_w105 | 132.4 | 132.2 | 131.5 | 132.0 | 138.9 | 142.6 |
| Zandvoort_w105 | 132.8 | 132.2 | 133.1 | 133.6 | 137.1 | 141.9 |

## failures (377)

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
- imitation/1319 worst Montreal_rev_w105@8/center at 44.7 s
- imitation/1319 worst Montreal_w105@8/center at 41.9 s
- imitation/1319 worst MoscowRaceway@8/center at 42.3 s
- imitation/1319 worst MoscowRaceway_rev_w105@8/center at 51.1 s
- imitation/1319 worst MoscowRaceway_w105@8/center at 11.1 s
- imitation/1319 worst SaoPaulo_w105@8/center at 32.0 s
- imitation/1319 worst Shanghai_rev@8/center at 65.7 s
- imitation/1319 worst Shanghai_rev_w105@8/center at 86.1 s
- imitation/1319 worst Sochi_w105@8/center at 80.6 s
- imitation/1319 worst YasMarina_w105@8/center at 17.4 s
- imitation/1319 worst Zandvoort_w105@8/center at 54.7 s
- imitation/1319 open Catalunya_rev_w902@8/center at 85.2 s
- imitation/1319 open MexicoCity_rev_w902@8/center at 65.8 s
- imitation/1319 open Montreal_w902@8/center at 60.3 s
- imitation/1319 open YasMarina_rev_w902@8/center at 134.2 s
- imitation/1319 open Montreal_rev_w902@5/center at 39.4 s
- imitation/1319 open Montreal_w902@5/center at 14.6 s
- imitation/1319 real Practice1124_rev@8/center at 10.4 s
- imitation/1319 real Rand305_s85_w305@2/center at 38.0 s
- imitation/1319 worst Catalunya@8/center#1 at 5.4 s
- imitation/1319 worst MexicoCity_rev_w105@8/center#1 at 99.5 s
- imitation/1319 worst Montreal_rev_w105@8/center#1 at 108.0 s
- imitation/1319 worst MoscowRaceway_rev_w105@8/center#1 at 7.0 s
- imitation/1319 worst SaoPaulo_w105@8/center#1 at 51.8 s
- imitation/1319 worst Sochi_rev_w105@8/center#1 at 35.9 s
- imitation/1319 worst Sochi_w105@8/center#1 at 2.0 s
- imitation/1319 worst Zandvoort_w105@8/center#1 at 28.7 s
- imitation/1319 open Catalunya_rev_w902@8/center#1 at 133.3 s
- imitation/1319 open MexicoCity_rev_w902@8/center#1 at 33.6 s
- imitation/1319 open Shanghai_rev_w902@8/center#1 at 36.8 s
- imitation/1319 open Silverstone_rev_w901@8/center#1 at 132.0 s
- imitation/1319 open Silverstone_w901@8/center#1 at 94.8 s
- imitation/1319 open Zandvoort_w902@8/center#1 at 122.3 s
- imitation/1319 open Montreal_rev_w902@5/center#1 at 26.5 s
- imitation/1319 open Montreal_w902@5/center#1 at 34.0 s
- imitation/1319 open Shanghai_rev_w901@5/center#1 at 90.6 s
- imitation/1319 real Rand305_s85_w305@2/center#1 at 3.2 s
- imitation/1319 worst Montreal_rev_w105@8/center#2 at 10.3 s
- imitation/1319 worst MoscowRaceway@8/center#2 at 128.2 s
- imitation/1319 worst MoscowRaceway_rev_w105@8/center#2 at 41.3 s
- imitation/1319 worst SaoPaulo_w105@8/center#2 at 49.8 s
- imitation/1319 worst Shanghai_rev_w105@8/center#2 at 77.3 s
- imitation/1319 worst Silverstone_w105@8/center#2 at 161.5 s
- imitation/1319 worst Sochi_w105@8/center#2 at 70.5 s
- imitation/1319 worst YasMarina_rev@8/center#2 at 70.6 s
- imitation/1319 worst Zandvoort_w105@8/center#2 at 16.5 s
- imitation/1319 open Catalunya_rev_w902@8/center#2 at 42.7 s
- imitation/1319 open MexicoCity_rev_w902@8/center#2 at 43.0 s
- imitation/1319 open Montreal_w902@8/center#2 at 48.8 s
- imitation/1319 open Silverstone_rev_w901@8/center#2 at 61.4 s
- imitation/1319 open Sochi_rev_w902@8/center#2 at 6.5 s
- imitation/1319 open Montreal_rev_w902@5/center#2 at 59.8 s
- imitation/1319 open Montreal_w902@5/center#2 at 63.0 s
- imitation/1319 real Rand303_rev_s60_w303@8/center#2 at 20.7 s
- imitation/1319 real Rand305_s85_w305@2/center#2 at 22.1 s
- ppo_s2r/1920 eval Silverstone@8/center at 72.8 s
- ppo_s2r/1920 eval Silverstone@8/random at 72.9 s
- ppo_s2r/1920 worst Montreal_rev_w105@8/center at 44.6 s
- ppo_s2r/1920 worst Montreal_w105@8/center at 15.5 s
- ppo_s2r/1920 worst MoscowRaceway@8/center at 108.1 s
- ppo_s2r/1920 worst MoscowRaceway_rev_w105@8/center at 34.2 s
- ppo_s2r/1920 worst Shanghai@8/center at 60.1 s
- ppo_s2r/1920 worst Silverstone@8/center at 81.0 s
- ppo_s2r/1920 worst Sochi_rev_w105@8/center at 44.6 s
- ppo_s2r/1920 worst YasMarina_w105@8/center at 17.1 s
- ppo_s2r/1920 worst Zandvoort_w105@8/center at 54.9 s
- ppo_s2r/1920 low Montreal_w105@2/center at 8.3 s
- ppo_s2r/1920 open Catalunya_rev_w902@8/center at 8.7 s
- ppo_s2r/1920 open MexicoCity_rev_w902@8/center at 66.4 s
- ppo_s2r/1920 open Montreal_rev_w902@8/center at 31.5 s
- ppo_s2r/1920 open Montreal_w902@8/center at 60.5 s
- ppo_s2r/1920 open Silverstone_w901@8/center at 7.0 s
- ppo_s2r/1920 open Sochi_rev_w902@8/center at 39.9 s
- ppo_s2r/1920 open MexicoCity_w902@5/center at 80.1 s
- ppo_s2r/1920 open Montreal_rev_w902@5/center at 39.3 s
- ppo_s2r/1920 open Montreal_w902@5/center at 14.7 s
- ppo_s2r/1920 open Shanghai_rev_w901@5/center at 37.3 s
- ppo_s2r/1920 open Shanghai_w902@5/center at 87.4 s
- ppo_s2r/1920 real Practice1124_rev@8/center at 10.2 s
- ppo_s2r/1920 real Rand303_rev_s60_w303@8/center at 9.5 s
- ppo_s2r/1920 real Rand305_s85_w305@2/center at 37.8 s
- ppo_s2r/1920 eval Shanghai_rev_w105@8/random#1 at 16.8 s
- ppo_s2r/1920 worst Catalunya@8/center#1 at 5.4 s
- ppo_s2r/1920 worst MoscowRaceway_rev_w105@8/center#1 at 7.3 s
- ppo_s2r/1920 worst SaoPaulo_w105@8/center#1 at 116.4 s
- ppo_s2r/1920 worst Shanghai@8/center#1 at 35.3 s
- ppo_s2r/1920 worst Silverstone_rev@8/center#1 at 25.1 s
- ppo_s2r/1920 worst Sochi_w105@8/center#1 at 1.9 s
- ppo_s2r/1920 worst Zandvoort_w105@8/center#1 at 53.0 s
- ppo_s2r/1920 open MexicoCity_rev_w902@8/center#1 at 34.9 s
- ppo_s2r/1920 open Montreal_w902@8/center#1 at 26.0 s
- ppo_s2r/1920 open Shanghai_rev_w902@8/center#1 at 122.6 s
- ppo_s2r/1920 open Silverstone_w901@8/center#1 at 19.9 s
- ppo_s2r/1920 open Sochi_rev_w902@8/center#1 at 68.3 s
- ppo_s2r/1920 open Zandvoort_rev_w902@8/center#1 at 133.3 s
- ppo_s2r/1920 open Catalunya_rev_w902@5/center#1 at 77.7 s
- ppo_s2r/1920 open MexicoCity_w902@5/center#1 at 130.6 s
- ppo_s2r/1920 open Montreal_rev_w902@5/center#1 at 26.6 s
- ppo_s2r/1920 open Montreal_w902@5/center#1 at 34.7 s
- ppo_s2r/1920 open Shanghai_w902@5/center#1 at 35.3 s
- ppo_s2r/1920 real Rand305_s85_w305@2/center#1 at 3.1 s
- ppo_s2r/1920 worst Montreal_rev@8/center#2 at 105.7 s
- ppo_s2r/1920 worst MoscowRaceway_rev_w105@8/center#2 at 40.9 s
- ppo_s2r/1920 worst MoscowRaceway_w105@8/center#2 at 61.8 s
- ppo_s2r/1920 worst SaoPaulo@8/center#2 at 1.9 s
- ppo_s2r/1920 worst SaoPaulo_w105@8/center#2 at 49.8 s
- ppo_s2r/1920 worst Shanghai@8/center#2 at 8.6 s
- ppo_s2r/1920 worst Shanghai_rev_w105@8/center#2 at 123.8 s
- ppo_s2r/1920 worst Shanghai_w105@8/center#2 at 135.7 s
- ppo_s2r/1920 worst Silverstone@8/center#2 at 94.2 s
- ppo_s2r/1920 worst Silverstone_rev@8/center#2 at 31.9 s
- ppo_s2r/1920 worst Sochi_rev_w105@8/center#2 at 44.2 s
- ppo_s2r/1920 worst YasMarina_rev@8/center#2 at 132.0 s
- ppo_s2r/1920 worst YasMarina_w105@8/center#2 at 99.0 s
- ppo_s2r/1920 worst Zandvoort@8/center#2 at 59.5 s
- ppo_s2r/1920 worst Zandvoort_rev_w105@8/center#2 at 115.5 s
- ppo_s2r/1920 worst Zandvoort_w105@8/center#2 at 16.4 s
- ppo_s2r/1920 open MexicoCity_rev_w902@8/center#2 at 42.9 s
- ppo_s2r/1920 open Montreal_w902@8/center#2 at 49.1 s
- ppo_s2r/1920 open SaoPaulo_w902@8/center#2 at 65.0 s
- ppo_s2r/1920 open Sochi_rev_w902@8/center#2 at 3.6 s
- ppo_s2r/1920 open Catalunya_rev_w902@5/center#2 at 147.1 s
- ppo_s2r/1920 open MexicoCity_w902@5/center#2 at 40.4 s
- ppo_s2r/1920 open Montreal_rev_w902@5/center#2 at 59.9 s
- ppo_s2r/1920 open Montreal_w902@5/center#2 at 62.2 s
- ppo_s2r/1920 open Shanghai_w902@5/center#2 at 56.6 s
- ppo_s2r/1920 real Practice1124_rev@8/center#2 at 18.5 s
- ppo_s2r/1920 real Rand303_rev_s60_w303@8/center#2 at 8.2 s
- ppo_s2r/1920 real Rand305_s85_w305@2/center#2 at 21.9 s
- ppo_s2r/2020 worst MoscowRaceway_rev_w105@8/center at 34.3 s
- ppo_s2r/2020 worst MoscowRaceway_w105@8/center at 11.1 s
- ppo_s2r/2020 worst SaoPaulo_w105@8/center at 31.8 s
- ppo_s2r/2020 worst Shanghai@8/center at 154.3 s
- ppo_s2r/2020 worst Shanghai_w105@8/center at 56.0 s
- ppo_s2r/2020 worst Silverstone@8/center at 36.7 s
- ppo_s2r/2020 worst Silverstone_w105@8/center at 84.0 s
- ppo_s2r/2020 worst Sochi_rev_w105@8/center at 164.6 s
- ppo_s2r/2020 worst YasMarina@8/center at 18.1 s
- ppo_s2r/2020 worst YasMarina_w105@8/center at 18.5 s
- ppo_s2r/2020 worst Zandvoort_w105@8/center at 55.0 s
- ppo_s2r/2020 open Catalunya_rev_w902@8/center at 13.2 s
- ppo_s2r/2020 open MexicoCity_rev_w902@8/center at 66.8 s
- ppo_s2r/2020 open Montreal_rev_w902@8/center at 9.2 s
- ppo_s2r/2020 open Montreal_w902@8/center at 61.1 s
- ppo_s2r/2020 open Silverstone_rev_w902@8/center at 31.9 s
- ppo_s2r/2020 open Silverstone_w901@8/center at 7.0 s
- ppo_s2r/2020 open Sochi_rev_w902@8/center at 40.2 s
- ppo_s2r/2020 open Zandvoort_rev_w902@8/center at 42.6 s
- ppo_s2r/2020 open Catalunya_rev_w902@5/center at 107.7 s
- ppo_s2r/2020 open MexicoCity_w902@5/center at 80.1 s
- ppo_s2r/2020 open Montreal_rev_w902@5/center at 39.6 s
- ppo_s2r/2020 open Montreal_w902@5/center at 14.6 s
- ppo_s2r/2020 open Sochi_rev_w902@5/center at 52.3 s
- ppo_s2r/2020 real Practice1124_rev@8/center at 10.3 s
- ppo_s2r/2020 real Rand305_s85_w305@2/center at 38.1 s
- ppo_s2r/2020 eval Silverstone_rev_w105@8/center#1 at 31.1 s
- ppo_s2r/2020 eval Silverstone_rev_w105@8/random#1 at 31.1 s
- ppo_s2r/2020 worst Catalunya@8/center#1 at 5.5 s
- ppo_s2r/2020 worst MoscowRaceway_w105@8/center#1 at 121.2 s
- ppo_s2r/2020 worst Shanghai@8/center#1 at 128.8 s
- ppo_s2r/2020 worst Shanghai_w105@8/center#1 at 11.1 s
- ppo_s2r/2020 worst Silverstone_rev@8/center#1 at 107.9 s
- ppo_s2r/2020 worst Sochi_rev_w105@8/center#1 at 36.0 s
- ppo_s2r/2020 worst Sochi_w105@8/center#1 at 1.9 s
- ppo_s2r/2020 worst Zandvoort_w105@8/center#1 at 53.1 s
- ppo_s2r/2020 open Catalunya_rev_w902@8/center#1 at 61.0 s
- ppo_s2r/2020 open MexicoCity_rev_w902@8/center#1 at 34.3 s
- ppo_s2r/2020 open Montreal_rev_w902@8/center#1 at 100.1 s
- ppo_s2r/2020 open Montreal_w902@8/center#1 at 26.0 s
- ppo_s2r/2020 open Silverstone_rev_w902@8/center#1 at 130.1 s
- ppo_s2r/2020 open Silverstone_w901@8/center#1 at 19.9 s
- ppo_s2r/2020 open Sochi_rev_w902@8/center#1 at 68.5 s
- ppo_s2r/2020 open Zandvoort_w902@8/center#1 at 122.7 s
- ppo_s2r/2020 open Catalunya_rev_w902@5/center#1 at 77.8 s
- ppo_s2r/2020 open Montreal_w902@5/center#1 at 34.6 s
- ppo_s2r/2020 open Shanghai_w902@5/center#1 at 147.8 s
- ppo_s2r/2020 real Rand305_s85_w305@2/center#1 at 3.2 s
- ppo_s2r/2020 worst MexicoCity_w105@8/center#2 at 16.1 s
- ppo_s2r/2020 worst MoscowRaceway_w105@8/center#2 at 62.4 s
- ppo_s2r/2020 worst SaoPaulo@8/center#2 at 1.9 s
- ppo_s2r/2020 worst SaoPaulo_w105@8/center#2 at 27.5 s
- ppo_s2r/2020 worst Shanghai@8/center#2 at 8.8 s
- ppo_s2r/2020 worst Shanghai_rev@8/center#2 at 72.7 s
- ppo_s2r/2020 worst Shanghai_w105@8/center#2 at 43.2 s
- ppo_s2r/2020 worst Silverstone_w105@8/center#2 at 162.1 s
- ppo_s2r/2020 worst Sochi_w105@8/center#2 at 1.8 s
- ppo_s2r/2020 worst YasMarina_rev@8/center#2 at 54.1 s
- ppo_s2r/2020 worst Zandvoort_w105@8/center#2 at 16.5 s
- ppo_s2r/2020 open Catalunya_rev_w902@8/center#2 at 115.1 s
- ppo_s2r/2020 open MexicoCity_rev_w902@8/center#2 at 35.6 s
- ppo_s2r/2020 open Montreal_rev_w902@8/center#2 at 24.8 s
- ppo_s2r/2020 open Shanghai_rev_w901@8/center#2 at 166.8 s
- ppo_s2r/2020 open Silverstone_rev_w902@8/center#2 at 70.3 s
- ppo_s2r/2020 open Sochi_rev_w902@8/center#2 at 3.6 s
- ppo_s2r/2020 open Zandvoort_rev_w902@8/center#2 at 42.6 s
- ppo_s2r/2020 open Montreal_w902@5/center#2 at 62.0 s
- ppo_s2r/2020 open Sochi_rev_w902@5/center#2 at 3.9 s
- ppo_s2r/2020 real Practice1124_rev@8/center#2 at 18.5 s
- ppo_s2r/2020 real Rand303_rev_s60_w303@8/center#2 at 8.4 s
- ppo_s2r/2020 real Rand305_s85_w305@2/center#2 at 22.1 s
- ppo_s2r/1730 eval YasMarina_w105@8/center at 117.7 s
- ppo_s2r/1730 eval Shanghai_w105@5/center at 65.7 s
- ppo_s2r/1730 eval MexicoCity_rev_w105@8/random at 112.1 s
- ppo_s2r/1730 eval YasMarina_w105@8/random at 117.7 s
- ppo_s2r/1730 worst IMS_rev_w105@8/center at 1.5 s
- ppo_s2r/1730 worst MoscowRaceway_rev_w105@8/center at 51.9 s
- ppo_s2r/1730 worst Shanghai_w105@8/center at 11.9 s
- ppo_s2r/1730 worst Silverstone@8/center at 165.0 s
- ppo_s2r/1730 worst Silverstone_rev_w105@8/center at 113.5 s
- ppo_s2r/1730 worst Silverstone_w105@8/center at 20.5 s
- ppo_s2r/1730 worst Zandvoort_w105@8/center at 130.9 s
- ppo_s2r/1730 low Shanghai_w105@3/center at 102.0 s
- ppo_s2r/1730 open Catalunya_rev_w902@8/center at 83.8 s
- ppo_s2r/1730 open MexicoCity_rev_w902@8/center at 28.6 s
- ppo_s2r/1730 open MexicoCity_w902@8/center at 66.0 s
- ppo_s2r/1730 open Montreal_w902@8/center at 11.0 s
- ppo_s2r/1730 open Shanghai_w902@8/center at 71.6 s
- ppo_s2r/1730 open Silverstone_rev_w901@8/center at 93.3 s
- ppo_s2r/1730 open MexicoCity_w902@5/center at 170.0 s
- ppo_s2r/1730 open Montreal_rev_w902@5/center at 41.7 s
- ppo_s2r/1730 open Montreal_w902@5/center at 14.6 s
- ppo_s2r/1730 open Shanghai_w901@5/center at 51.9 s
- ppo_s2r/1730 open Shanghai_w902@5/center at 91.9 s
- ppo_s2r/1730 real Rand302_rev_s87_w302@8/center at 12.9 s
- ppo_s2r/1730 eval MexicoCity_rev_w105@8/center#1 at 63.0 s
- ppo_s2r/1730 eval Shanghai_w105@8/center#1 at 13.8 s
- ppo_s2r/1730 eval YasMarina_w105@8/center#1 at 35.2 s
- ppo_s2r/1730 eval Shanghai_w105@5/center#1 at 21.8 s
- ppo_s2r/1730 eval MexicoCity_rev_w105@8/random#1 at 62.9 s
- ppo_s2r/1730 eval Shanghai_w105@8/random#1 at 13.7 s
- ppo_s2r/1730 eval YasMarina_w105@8/random#1 at 35.4 s
- ppo_s2r/1730 worst Catalunya@8/center#1 at 156.7 s
- ppo_s2r/1730 worst MexicoCity_rev_w105@8/center#1 at 32.6 s
- ppo_s2r/1730 worst MoscowRaceway@8/center#1 at 67.3 s
- ppo_s2r/1730 worst MoscowRaceway_w105@8/center#1 at 122.7 s
- ppo_s2r/1730 worst Shanghai_w105@8/center#1 at 98.5 s
- ppo_s2r/1730 low Sochi_rev_w105@2/center#1 at 492.9 s
- ppo_s2r/1730 open Catalunya_rev_w902@8/center#1 at 58.4 s
- ppo_s2r/1730 open Catalunya_w902@8/center#1 at 120.5 s
- ppo_s2r/1730 open MexicoCity_rev_w902@8/center#1 at 62.0 s
- ppo_s2r/1730 open MexicoCity_w902@8/center#1 at 42.3 s
- ppo_s2r/1730 open Montreal_w902@8/center#1 at 26.9 s
- ppo_s2r/1730 open Silverstone_w902@8/center#1 at 62.8 s
- ppo_s2r/1730 open Sochi_w901@8/center#1 at 125.0 s
- ppo_s2r/1730 open MexicoCity_w902@5/center#1 at 137.9 s
- ppo_s2r/1730 open Montreal_rev_w902@5/center#1 at 28.2 s
- ppo_s2r/1730 open Montreal_w902@5/center#1 at 35.3 s
- ppo_s2r/1730 open Shanghai_w901@5/center#1 at 79.1 s
- ppo_s2r/1730 open Shanghai_w902@5/center#1 at 38.1 s
- ppo_s2r/1730 eval MexicoCity_rev_w105@8/center#2 at 124.7 s
- ppo_s2r/1730 eval YasMarina_w105@8/center#2 at 63.4 s
- ppo_s2r/1730 eval Shanghai_w105@5/center#2 at 118.0 s
- ppo_s2r/1730 eval MexicoCity_rev_w105@8/random#2 at 124.9 s
- ppo_s2r/1730 eval Shanghai_w105@8/random#2 at 178.4 s
- ppo_s2r/1730 eval YasMarina_rev@8/random#2 at 135.8 s
- ppo_s2r/1730 eval YasMarina_w105@8/random#2 at 63.4 s
- ppo_s2r/1730 worst MexicoCity_rev_w105@8/center#2 at 115.0 s
- ppo_s2r/1730 worst Montreal_rev_w105@8/center#2 at 107.5 s
- ppo_s2r/1730 worst MoscowRaceway@8/center#2 at 70.6 s
- ppo_s2r/1730 worst MoscowRaceway_w105@8/center#2 at 107.0 s
- ppo_s2r/1730 worst Shanghai@8/center#2 at 104.7 s
- ppo_s2r/1730 worst Shanghai_rev@8/center#2 at 85.5 s
- ppo_s2r/1730 worst Zandvoort_w105@8/center#2 at 91.1 s
- ppo_s2r/1730 low Shanghai_w105@4/center#2 at 141.3 s
- ppo_s2r/1730 open Catalunya_rev_w902@8/center#2 at 39.7 s
- ppo_s2r/1730 open MexicoCity_rev_w902@8/center#2 at 45.2 s
- ppo_s2r/1730 open Montreal_w902@8/center#2 at 50.4 s
- ppo_s2r/1730 open Shanghai_w902@8/center#2 at 46.3 s
- ppo_s2r/1730 open Sochi_w901@8/center#2 at 159.0 s
- ppo_s2r/1730 open Montreal_rev_w902@5/center#2 at 62.7 s
- ppo_s2r/1730 open Montreal_w902@5/center#2 at 64.1 s
- ppo_s2r/1730 open Shanghai_w901@5/center#2 at 35.5 s
- ppo_s2r/1730 open Shanghai_w902@5/center#2 at 60.3 s
- ppo_s2r/1730 real Rand300_s83_w300@2/center#2 at 30.5 s
- ppo_s2r/1730 real Rand302_rev_s87_w302@2/center#2 at 29.6 s
- ppo_s2r/1640 eval Shanghai@8/center at 132.6 s
- ppo_s2r/1640 eval Shanghai_w105@8/center at 53.1 s
- ppo_s2r/1640 eval Shanghai@8/random at 132.5 s
- ppo_s2r/1640 eval Shanghai_w105@8/random at 53.0 s
- ppo_s2r/1640 worst MexicoCity_w105@8/center at 24.6 s
- ppo_s2r/1640 worst Montreal_rev_w105@8/center at 114.1 s
- ppo_s2r/1640 worst Shanghai@8/center at 64.1 s
- ppo_s2r/1640 open MexicoCity_rev_w902@8/center at 125.3 s
- ppo_s2r/1640 open Montreal_rev_w902@8/center at 90.3 s
- ppo_s2r/1640 open Montreal_w902@8/center at 11.9 s
- ppo_s2r/1640 open Shanghai_w901@8/center at 39.5 s
- ppo_s2r/1640 open Shanghai_w902@8/center at 76.7 s
- ppo_s2r/1640 open Montreal_rev_w902@5/center at 42.2 s
- ppo_s2r/1640 open Montreal_w902@5/center at 83.3 s
- ppo_s2r/1640 open Shanghai_w902@5/center at 90.5 s
- ppo_s2r/1640 real Practice1124_rev@8/center at 10.4 s
- ppo_s2r/1640 real Practice1124_rev@4/center at 9.1 s
- ppo_s2r/1640 real Rand300_s83_w300@2/center at 10.2 s
- ppo_s2r/1640 real Rand305_s85_w305@2/center at 41.1 s
- ppo_s2r/1640 eval Shanghai_w105@8/center#1 at 18.1 s
- ppo_s2r/1640 eval Shanghai_w105@8/random#1 at 18.1 s
- ppo_s2r/1640 worst MexicoCity_w105@8/center#1 at 81.3 s
- ppo_s2r/1640 worst Montreal_rev@8/center#1 at 77.1 s
- ppo_s2r/1640 worst Shanghai@8/center#1 at 37.0 s
- ppo_s2r/1640 worst Shanghai_w105@8/center#1 at 3.5 s
- ppo_s2r/1640 low YasMarina_w105@2/center#1 at 136.0 s
- ppo_s2r/1640 open MexicoCity_rev_w902@8/center#1 at 22.8 s
- ppo_s2r/1640 open Montreal_w902@8/center#1 at 28.2 s
- ppo_s2r/1640 open Shanghai_w901@8/center#1 at 62.7 s
- ppo_s2r/1640 open Shanghai_w902@8/center#1 at 31.6 s
- ppo_s2r/1640 open Montreal_rev_w902@5/center#1 at 28.4 s
- ppo_s2r/1640 open Montreal_w902@5/center#1 at 101.1 s
- ppo_s2r/1640 open Shanghai_w902@5/center#1 at 40.1 s
- ppo_s2r/1640 open YasMarina_rev_w902@5/center#1 at 32.7 s
- ppo_s2r/1640 real Practice1124_rev@4/center#1 at 17.8 s
- ppo_s2r/1640 real Rand300_s83_w300@2/center#1 at 31.3 s
- ppo_s2r/1640 eval Shanghai_w105@8/center#2 at 92.5 s
- ppo_s2r/1640 eval YasMarina_w105@8/center#2 at 142.5 s
- ppo_s2r/1640 eval Shanghai_w105@8/random#2 at 92.3 s
- ppo_s2r/1640 worst MexicoCity_w105@8/center#2 at 88.6 s
- ppo_s2r/1640 worst SaoPaulo@8/center#2 at 1.8 s
- ppo_s2r/1640 worst Shanghai@8/center#2 at 109.2 s
- ppo_s2r/1640 worst Shanghai_w105@8/center#2 at 90.0 s
- ppo_s2r/1640 open MexicoCity_rev_w902@8/center#2 at 32.5 s
- ppo_s2r/1640 open Shanghai_w901@8/center#2 at 28.1 s
- ppo_s2r/1640 open Shanghai_w902@8/center#2 at 50.5 s
- ppo_s2r/1640 open Montreal_rev_w902@5/center#2 at 62.7 s
- ppo_s2r/1640 open Montreal_w902@5/center#2 at 63.1 s
- ppo_s2r/1640 open Shanghai_w902@5/center#2 at 62.0 s
- ppo_s2r/1640 open YasMarina_rev_w902@5/center#2 at 12.4 s
- ppo_s2r/1640 real Practice1124_rev@4/center#2 at 8.0 s
- ppo_s2r/1640 real Rand300_s83_w300@2/center#2 at 17.4 s
- ppo_s2r/1640 real Rand302_s87_w302@2/center#2 at 72.2 s
- ppo_s2r/1640 real Rand305_s85_w305@2/center#2 at 69.1 s
