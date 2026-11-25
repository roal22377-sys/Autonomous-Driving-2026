# model comparison (2026-11-26 08:37, 2 lap(s), reference = imitation/13)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter | real course (caps 8/4/2) | real 2-lap time, cap 8 |
|---|---|---|---|---|---|---|---|---|---|
| imitation/13 | 100.0% (396/396) | 100.0% (132/132) | 78.0% (103/132) | 94.3% (249/264) | reference | 5.30 m/s | 0.0041 rad | 97.5% (158/162) | 31.0 s |
| ppo_s2r/1730 | 95.5% (378/396) | 97.7% (129/132) | 85.6% (113/132) | 87.9% (232/264) | 3.5% slower on 249 runs | 5.09 m/s | 0.0064 rad | 98.1% (159/162) | 31.4 s |
| ppo_s2r/2130 | 99.2% (393/396) | 100.0% (132/132) | 82.6% (109/132) | 94.3% (249/264) | 0.6% slower on 261 runs | 5.25 m/s | 0.0044 rad | 99.4% (161/162) | 30.9 s |
| ppo_s2r/2150 | 99.2% (393/396) | 100.0% (132/132) | 86.4% (114/132) | 93.6% (247/264) | 0.9% slower on 261 runs | 5.24 m/s | 0.0046 rad | 98.8% (160/162) | 31.1 s |
| ppo_s2r/2170 | 97.5% (386/396) | 98.5% (130/132) | 84.1% (111/132) | 90.2% (238/264) | 0.8% slower on 254 runs | 5.24 m/s | 0.0048 rad | 98.1% (159/162) | 30.9 s |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | imitation/13 | ppo_s2r/1730 | ppo_s2r/2130 | ppo_s2r/2150 | ppo_s2r/2170 |
|---|---|---|---|---|---|
| Catalunya | 143.4 | 146.9 | 143.7 | 144.1 | 143.7 |
| Catalunya_rev | 143.0 | 148.9 | 143.4 | 143.4 | 142.6 |
| Catalunya_rev_w105 | 141.9 | 148.0 | 142.9 | 143.0 | 143.3 |
| Catalunya_w105 | 144.4 | 147.7 | 145.7 | 145.8 | 145.5 |
| IMS | 84.1 | 86.0 | 84.3 | 84.4 | 83.5 |
| IMS_rev | 85.8 | 86.6 | 86.7 | 86.3 | 86.2 |
| IMS_rev_w105 | 86.0 | 86.2 | 86.8 | 86.4 | 86.3 |
| IMS_w105 | 84.5 | 88.6 | 84.1 | 83.7 | 83.6 |
| MexicoCity | 126.4 | 133.4 | 127.6 | 128.8 | 129.8 |
| MexicoCity_rev | 126.2 | 131.5 | 127.2 | 127.3 | 126.1 |
| MexicoCity_rev_w105 | 126.7 | 136.0 | 127.0 | 126.7 | 127.3 |
| MexicoCity_w105 | 128.0 | 134.9 | 129.3 | 129.5 | 130.3 |
| Montreal | 99.7 | 103.8 | 100.6 | 100.9 | 100.1 |
| Montreal_rev | 100.7 | 103.5 | 100.2 | 100.6 | 101.4 |
| Montreal_rev_w105 | 100.7 | 103.4 | 101.3 | 101.8 | 102.1 |
| Montreal_w105 | 98.8 | 102.3 | 99.8 | 100.3 | 100.2 |
| MoscowRaceway | 122.5 | 127.0 | 123.1 | 124.4 | 123.4 |
| MoscowRaceway_rev | 122.9 | 126.8 | 123.6 | 124.6 | 125.2 |
| MoscowRaceway_rev_w105 | 120.7 | 125.1 | 122.0 | 122.3 | 122.0 |
| MoscowRaceway_w105 | 122.6 | 128.0 | 123.3 | 124.7 | 125.2 |
| SaoPaulo | 119.4 | 124.3 | 119.7 | 120.2 | 120.3 |
| SaoPaulo_rev | 120.8 | 123.4 | 121.5 | 121.6 | 121.3 |
| SaoPaulo_rev_w105 | 118.8 | 121.8 | 119.6 | 119.9 | 119.8 |
| SaoPaulo_w105 | 118.5 | 123.6 | 118.8 | 119.1 | 119.1 |
| Shanghai | 172.4 | 181.2 | 174.0 | 174.8 | 176.0 |
| Shanghai_rev | 171.3 | 176.3 | 172.3 | 172.6 | 172.0 |
| Shanghai_rev_w105 | 171.7 | 178.1 | 172.8 | 173.6 | 173.6 |
| Shanghai_w105 | 170.6 | 177.3 | 172.9 | 172.8 | - |
| Silverstone | 152.3 | 156.0 | 153.1 | 153.0 | 152.6 |
| Silverstone_rev | 152.0 | 156.5 | 152.4 | 153.2 | 153.7 |
| Silverstone_rev_w105 | 153.4 | 157.9 | 154.0 | 153.7 | 155.2 |
| Silverstone_w105 | 154.3 | 158.0 | 154.9 | 155.6 | 155.2 |
| Sochi | 156.2 | 159.8 | 157.2 | 157.6 | 157.5 |
| Sochi_rev | 157.0 | 160.9 | 157.4 | 157.5 | 156.7 |
| Sochi_rev_w105 | 156.5 | 160.8 | 157.3 | 156.9 | 156.8 |
| Sochi_w105 | 157.1 | 161.0 | 158.9 | 158.6 | 159.1 |
| YasMarina | 145.2 | 152.6 | 146.4 | 147.1 | 147.4 |
| YasMarina_rev | 146.0 | 154.7 | 147.1 | 148.3 | - |
| YasMarina_rev_w105 | 146.0 | 154.2 | 148.6 | 149.6 | 149.8 |
| YasMarina_w105 | 146.1 | - | 146.3 | 147.0 | 147.3 |
| Zandvoort | 133.5 | 137.9 | 133.8 | 134.0 | 133.6 |
| Zandvoort_rev | 133.6 | 138.9 | 133.7 | 134.2 | 134.3 |
| Zandvoort_rev_w105 | 132.4 | 138.9 | 132.8 | 132.7 | 132.8 |
| Zandvoort_w105 | 132.8 | 137.1 | 133.5 | 133.7 | 133.4 |

## failures (267)

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
- ppo_s2r/2150 eval Silverstone@8/random at 73.0 s
- ppo_s2r/2150 worst Catalunya_rev_w105@8/center at 5.9 s
- ppo_s2r/2150 worst Montreal_rev@8/center at 13.9 s
- ppo_s2r/2150 worst Shanghai_rev_w105@8/center at 85.6 s
- ppo_s2r/2150 worst Silverstone@8/center at 81.8 s
- ppo_s2r/2150 worst Silverstone_w105@8/center at 83.9 s
- ppo_s2r/2150 worst Zandvoort_w105@8/center at 118.4 s
- ppo_s2r/2150 open MexicoCity_rev_w902@8/center at 66.3 s
- ppo_s2r/2150 open Montreal_rev_w902@8/center at 32.4 s
- ppo_s2r/2150 open Montreal_rev_w902@5/center at 40.3 s
- ppo_s2r/2150 open Montreal_w902@5/center at 15.0 s
- ppo_s2r/2150 real Practice1124_rev@8/center at 10.7 s
- ppo_s2r/2150 eval Shanghai_rev_w105@8/random#1 at 16.7 s
- ppo_s2r/2150 worst MoscowRaceway_rev_w105@8/center#1 at 74.1 s
- ppo_s2r/2150 worst MoscowRaceway_w105@8/center#1 at 121.8 s
- ppo_s2r/2150 worst Shanghai_w105@8/center#1 at 5.9 s
- ppo_s2r/2150 worst Sochi_w105@8/center#1 at 1.8 s
- ppo_s2r/2150 worst Zandvoort_w105@8/center#1 at 126.9 s
- ppo_s2r/2150 open MexicoCity_rev_w902@8/center#1 at 60.4 s
- ppo_s2r/2150 open Shanghai_rev_w902@8/center#1 at 36.6 s
- ppo_s2r/2150 open Silverstone_rev_w902@8/center#1 at 15.9 s
- ppo_s2r/2150 open Montreal_rev_w902@5/center#1 at 27.2 s
- ppo_s2r/2150 open Montreal_w902@5/center#1 at 35.1 s
- ppo_s2r/2150 real Rand305_s85_w305@2/center#1 at 47.1 s
- ppo_s2r/2150 eval Silverstone_rev_w105@8/center#2 at 78.5 s
- ppo_s2r/2150 worst Catalunya_rev_w105@8/center#2 at 59.5 s
- ppo_s2r/2150 worst MexicoCity_rev_w105@8/center#2 at 1.9 s
- ppo_s2r/2150 worst MoscowRaceway_rev_w105@8/center#2 at 41.3 s
- ppo_s2r/2150 worst Silverstone@8/center#2 at 11.8 s
- ppo_s2r/2150 worst Silverstone_w105@8/center#2 at 39.8 s
- ppo_s2r/2150 worst YasMarina_rev@8/center#2 at 54.1 s
- ppo_s2r/2150 worst YasMarina_rev_w105@8/center#2 at 1.9 s
- ppo_s2r/2150 open Catalunya_rev_w901@8/center#2 at 67.6 s
- ppo_s2r/2150 open MexicoCity_rev_w902@8/center#2 at 69.3 s
- ppo_s2r/2150 open Montreal_rev_w902@8/center#2 at 98.4 s
- ppo_s2r/2150 open Silverstone_rev_w901@8/center#2 at 77.6 s
- ppo_s2r/2150 open Silverstone_rev_w902@8/center#2 at 76.9 s
- ppo_s2r/2150 open Silverstone_w901@8/center#2 at 36.7 s
- ppo_s2r/2150 open Montreal_rev_w902@5/center#2 at 61.5 s
- ppo_s2r/2150 open Montreal_w902@5/center#2 at 62.9 s
- ppo_s2r/2170 eval Shanghai_w105@8/center at 50.1 s
- ppo_s2r/2170 eval YasMarina_rev@8/center at 57.9 s
- ppo_s2r/2170 eval Shanghai_w105@8/random at 49.2 s
- ppo_s2r/2170 eval YasMarina_rev@8/random at 57.9 s
- ppo_s2r/2170 eval YasMarina_w105@8/random at 32.6 s
- ppo_s2r/2170 worst Catalunya_rev_w105@8/center at 129.5 s
- ppo_s2r/2170 worst MexicoCity_w105@8/center at 23.8 s
- ppo_s2r/2170 worst Montreal_rev@8/center at 69.2 s
- ppo_s2r/2170 worst Shanghai_rev_w105@8/center at 119.3 s
- ppo_s2r/2170 worst Silverstone@8/center at 81.3 s
- ppo_s2r/2170 worst Silverstone_rev_w105@8/center at 135.7 s
- ppo_s2r/2170 worst Silverstone_w105@8/center at 147.4 s
- ppo_s2r/2170 worst Zandvoort_w105@8/center at 54.9 s
- ppo_s2r/2170 open MexicoCity_rev_w902@8/center at 92.7 s
- ppo_s2r/2170 open Montreal_rev_w902@8/center at 32.6 s
- ppo_s2r/2170 open Montreal_w902@8/center at 61.2 s
- ppo_s2r/2170 open Shanghai_w901@8/center at 36.2 s
- ppo_s2r/2170 open Shanghai_w902@8/center at 159.7 s
- ppo_s2r/2170 open Silverstone_rev_w902@8/center at 149.2 s
- ppo_s2r/2170 open Silverstone_w901@8/center at 57.9 s
- ppo_s2r/2170 open Sochi_w901@8/center at 88.5 s
- ppo_s2r/2170 open Montreal_rev_w902@5/center at 40.3 s
- ppo_s2r/2170 open Montreal_w902@5/center at 79.3 s
- ppo_s2r/2170 open Shanghai_w902@5/center at 200.4 s
- ppo_s2r/2170 real Rand305_s85_w305@2/center at 38.6 s
- ppo_s2r/2170 eval Shanghai_w105@8/random#1 at 13.6 s
- ppo_s2r/2170 worst IMS_rev_w105@8/center#1 at 85.2 s
- ppo_s2r/2170 worst Montreal_rev@8/center#1 at 52.0 s
- ppo_s2r/2170 worst MoscowRaceway_rev_w105@8/center#1 at 7.0 s
- ppo_s2r/2170 worst MoscowRaceway_w105@8/center#1 at 122.7 s
- ppo_s2r/2170 worst Shanghai_w105@8/center#1 at 5.3 s
- ppo_s2r/2170 worst Silverstone_rev@8/center#1 at 148.0 s
- ppo_s2r/2170 worst Sochi_w105@8/center#1 at 1.8 s
- ppo_s2r/2170 worst Zandvoort_w105@8/center#1 at 28.8 s
- ppo_s2r/2170 low Sochi_rev_w105@2/center#1 at 299.1 s
- ppo_s2r/2170 open Catalunya_rev_w901@8/center#1 at 78.3 s
- ppo_s2r/2170 open Shanghai_w901@8/center#1 at 59.7 s
- ppo_s2r/2170 open Shanghai_w902@8/center#1 at 29.1 s
- ppo_s2r/2170 open Silverstone_rev_w902@8/center#1 at 15.8 s
- ppo_s2r/2170 open Montreal_rev_w902@5/center#1 at 94.0 s
- ppo_s2r/2170 open Montreal_w902@5/center#1 at 34.8 s
- ppo_s2r/2170 real Rand305_s85_w305@2/center#1 at 3.3 s
- ppo_s2r/2170 eval Shanghai_w105@8/center#2 at 85.4 s
- ppo_s2r/2170 eval YasMarina_w105@8/center#2 at 52.8 s
- ppo_s2r/2170 eval Shanghai_w105@8/random#2 at 88.2 s
- ppo_s2r/2170 eval YasMarina_w105@8/random#2 at 52.8 s
- ppo_s2r/2170 worst MexicoCity_rev_w105@8/center#2 at 1.7 s
- ppo_s2r/2170 worst Montreal_w105@8/center#2 at 105.6 s
- ppo_s2r/2170 worst MoscowRaceway_rev_w105@8/center#2 at 41.4 s
- ppo_s2r/2170 worst Shanghai@8/center#2 at 9.3 s
- ppo_s2r/2170 worst YasMarina_rev_w105@8/center#2 at 1.8 s
- ppo_s2r/2170 low Sochi_rev@2/center#2 at 315.7 s
- ppo_s2r/2170 open Catalunya_rev_w901@8/center#2 at 67.6 s
- ppo_s2r/2170 open Catalunya_rev_w902@8/center#2 at 37.9 s
- ppo_s2r/2170 open Montreal_rev_w902@8/center#2 at 98.6 s
- ppo_s2r/2170 open Montreal_w902@8/center#2 at 48.5 s
- ppo_s2r/2170 open Shanghai_w901@8/center#2 at 25.6 s
- ppo_s2r/2170 open Shanghai_w902@8/center#2 at 134.8 s
- ppo_s2r/2170 open Sochi_rev_w901@8/center#2 at 35.2 s
- ppo_s2r/2170 open Montreal_rev_w902@5/center#2 at 128.0 s
- ppo_s2r/2170 open Montreal_w902@5/center#2 at 62.4 s
- ppo_s2r/2170 real Rand305_s85_w305@2/center#2 at 22.6 s
