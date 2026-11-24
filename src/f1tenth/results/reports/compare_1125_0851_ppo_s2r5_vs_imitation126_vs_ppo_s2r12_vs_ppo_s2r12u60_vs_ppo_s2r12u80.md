# model comparison (2026-11-25 08:51, 2 lap(s), reference = ppo_s2r/5)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter |
|---|---|---|---|---|---|---|---|
| ppo_s2r/5 | 93.2% (123/132) | 86.4% (38/44) | 90.9% (40/44) | 65.9% (58/88) | reference | 4.32 m/s | 0.0127 rad |
| imitation/126 | 85.6% (113/132) | 86.4% (38/44) | 77.3% (34/44) | 54.5% (48/88) | 10.2% slower on 72 runs | 3.80 m/s | 0.0202 rad |
| ppo_s2r/12 | 93.2% (123/132) | 88.6% (39/44) | 93.2% (41/44) | 68.2% (60/88) | 0.7% faster on 84 runs | 4.35 m/s | 0.0125 rad |
| ppo_s2r/12u60 | 92.4% (122/132) | 90.9% (40/44) | 72.7% (32/44) | 62.5% (55/88) | 6.6% faster on 80 runs | 4.58 m/s | 0.0118 rad |
| ppo_s2r/12u80 | 88.6% (117/132) | 90.9% (40/44) | 86.4% (38/44) | 67.0% (59/88) | 5.0% faster on 77 runs | 4.52 m/s | 0.0123 rad |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | ppo_s2r/5 | imitation/126 | ppo_s2r/12 | ppo_s2r/12u60 | ppo_s2r/12u80 |
|---|---|---|---|---|---|
| Catalunya | 185.7 | 199.0 | 185.7 | 172.5 | 175.5 |
| Catalunya_rev | 186.0 | 195.7 | 184.5 | 173.4 | 175.8 |
| Catalunya_rev_w105 | 193.3 | - | 190.2 | 177.8 | 180.9 |
| Catalunya_w105 | 188.7 | 209.9 | 188.0 | 173.5 | - |
| IMS | 119.0 | 142.4 | 118.8 | 114.4 | 115.6 |
| IMS_rev | 113.5 | 136.7 | 112.8 | 103.1 | 107.1 |
| IMS_rev_w105 | 116.7 | 147.7 | 116.7 | 106.6 | 110.9 |
| IMS_w105 | 114.4 | 151.4 | 114.7 | 112.7 | 115.4 |
| MexicoCity | 169.0 | 178.9 | 168.0 | 156.2 | 159.2 |
| MexicoCity_rev | 167.8 | 180.7 | 166.5 | 155.4 | 156.6 |
| MexicoCity_rev_w105 | 162.5 | 183.3 | 160.6 | - | - |
| MexicoCity_w105 | 166.4 | 192.4 | 165.0 | 154.2 | 155.1 |
| Montreal | 129.6 | 135.3 | 128.1 | 121.7 | 122.3 |
| Montreal_rev | 131.9 | 134.1 | 131.8 | 125.2 | 126.0 |
| Montreal_rev_w105 | 128.0 | 147.9 | 127.6 | 121.8 | 123.3 |
| Montreal_w105 | 127.0 | - | 126.2 | 119.7 | 121.6 |
| MoscowRaceway | 152.5 | 158.2 | 151.0 | 142.1 | 142.8 |
| MoscowRaceway_rev | 157.3 | 161.2 | 156.1 | 146.3 | 148.8 |
| MoscowRaceway_rev_w105 | 153.2 | 173.6 | 151.6 | 143.3 | 146.4 |
| MoscowRaceway_w105 | 146.5 | - | 146.6 | 140.7 | 142.5 |
| SaoPaulo | 154.4 | 161.8 | 153.0 | 145.2 | 148.5 |
| SaoPaulo_rev | 153.0 | 160.2 | 151.7 | 141.3 | 143.7 |
| SaoPaulo_rev_w105 | 153.6 | 174.1 | 152.6 | 141.3 | 143.7 |
| SaoPaulo_w105 | 152.3 | 176.2 | 150.8 | 144.7 | 147.8 |
| Shanghai | 227.7 | 249.6 | 226.3 | 210.3 | 214.5 |
| Shanghai_rev | 230.8 | - | 229.1 | 214.3 | 216.9 |
| Shanghai_rev_w105 | - | 262.8 | - | - | - |
| Shanghai_w105 | 226.1 | - | 225.1 | - | 215.3 |
| Silverstone | 199.8 | 216.8 | 198.5 | 186.0 | 189.0 |
| Silverstone_rev | 198.8 | 211.8 | 197.8 | 186.3 | 188.4 |
| Silverstone_rev_w105 | 201.9 | 236.2 | 201.1 | 190.9 | 194.3 |
| Silverstone_w105 | 201.8 | 239.5 | 199.0 | 189.8 | 192.6 |
| Sochi | 206.7 | 213.9 | 205.2 | 195.4 | 198.8 |
| Sochi_rev | 206.0 | 215.3 | 204.3 | 191.0 | 194.3 |
| Sochi_rev_w105 | 206.0 | 234.6 | 204.4 | 194.8 | 196.9 |
| Sochi_w105 | 210.7 | 248.3 | 208.5 | 193.5 | 198.9 |
| YasMarina | 196.2 | 200.2 | 194.5 | 179.0 | - |
| YasMarina_rev | 192.9 | 195.7 | 190.5 | 179.9 | 182.0 |
| YasMarina_rev_w105 | 194.9 | - | 193.6 | 178.4 | 184.1 |
| YasMarina_w105 | - | - | - | - | - |
| Zandvoort | 171.6 | 176.2 | 170.4 | 159.2 | 162.2 |
| Zandvoort_rev | 171.6 | 175.8 | 170.3 | 161.5 | 162.6 |
| Zandvoort_rev_w105 | 173.5 | 187.5 | 171.9 | 161.7 | 164.9 |
| Zandvoort_w105 | 172.6 | 188.4 | 171.1 | 158.2 | 161.7 |

## failures (282)

- ppo_s2r/5 eval Shanghai_rev_w105@8/center at 23.2 s
- ppo_s2r/5 eval YasMarina_w105@8/center at 52.0 s
- ppo_s2r/5 eval Catalunya_w105@5/center at 26.3 s
- ppo_s2r/5 eval Montreal_w105@5/center at 6.3 s
- ppo_s2r/5 eval Shanghai_rev_w105@5/center at 23.0 s
- ppo_s2r/5 eval Shanghai_w105@5/center at 289.2 s
- ppo_s2r/5 eval YasMarina_w105@5/center at 216.5 s
- ppo_s2r/5 eval Shanghai_rev_w105@8/random at 23.1 s
- ppo_s2r/5 eval YasMarina_w105@8/random at 52.1 s
- ppo_s2r/5 worst Shanghai_rev_w105@8/center at 82.0 s
- ppo_s2r/5 worst Shanghai_w105@8/center at 14.0 s
- ppo_s2r/5 worst YasMarina_rev_w105@8/center at 507.6 s
- ppo_s2r/5 worst YasMarina_w105@8/center at 91.4 s
- ppo_s2r/5 low Montreal_w105@2/center at 11.0 s
- ppo_s2r/5 low Shanghai_rev_w105@2.5/center at 42.9 s
- ppo_s2r/5 low Silverstone_w105@2.5/center at 214.9 s
- ppo_s2r/5 low Sochi_rev_w105@4/center at 38.3 s
- ppo_s2r/5 low YasMarina_rev_w105@3.5/center at 52.6 s
- ppo_s2r/5 low YasMarina_w105@4/center at 61.0 s
- ppo_s2r/5 open Catalunya_rev_w902@8/center at 531.0 s
- ppo_s2r/5 open MexicoCity_rev_w901@8/center at 455.9 s
- ppo_s2r/5 open MexicoCity_w901@8/center at 44.7 s
- ppo_s2r/5 open Montreal_rev_w901@8/center at 17.4 s
- ppo_s2r/5 open Montreal_rev_w902@8/center at 63.1 s
- ppo_s2r/5 open MoscowRaceway_rev_w902@8/center at 38.5 s
- ppo_s2r/5 open MoscowRaceway_w902@8/center at 27.0 s
- ppo_s2r/5 open Shanghai_rev_w901@8/center at 41.9 s
- ppo_s2r/5 open Shanghai_rev_w902@8/center at 82.2 s
- ppo_s2r/5 open Shanghai_w901@8/center at 49.6 s
- ppo_s2r/5 open Shanghai_w902@8/center at 91.8 s
- ppo_s2r/5 open Silverstone_rev_w902@8/center at 34.0 s
- ppo_s2r/5 open Silverstone_w902@8/center at 64.6 s
- ppo_s2r/5 open Sochi_rev_w902@8/center at 46.8 s
- ppo_s2r/5 open YasMarina_rev_w901@8/center at 507.6 s
- ppo_s2r/5 open YasMarina_w901@8/center at 152.0 s
- ppo_s2r/5 open Catalunya_rev_w902@5/center at 18.8 s
- ppo_s2r/5 open MexicoCity_rev_w901@5/center at 77.3 s
- ppo_s2r/5 open MexicoCity_w901@5/center at 44.4 s
- ppo_s2r/5 open Montreal_rev_w901@5/center at 18.0 s
- ppo_s2r/5 open Montreal_w901@5/center at 39.0 s
- ppo_s2r/5 open Montreal_w902@5/center at 67.5 s
- ppo_s2r/5 open MoscowRaceway_w902@5/center at 27.5 s
- ppo_s2r/5 open Shanghai_rev_w901@5/center at 155.0 s
- ppo_s2r/5 open Shanghai_rev_w902@5/center at 85.0 s
- ppo_s2r/5 open Shanghai_w901@5/center at 50.6 s
- ppo_s2r/5 open Silverstone_rev_w902@5/center at 582.4 s
- ppo_s2r/5 open Silverstone_w902@5/center at 174.4 s
- ppo_s2r/5 open YasMarina_rev_w902@5/center at 126.2 s
- ppo_s2r/5 open Zandvoort_w902@5/center at 58.2 s
- imitation/126 eval Catalunya_rev_w105@8/center at 167.1 s
- imitation/126 eval Montreal_w105@8/center at 4.7 s
- imitation/126 eval MoscowRaceway_w105@8/center at 167.0 s
- imitation/126 eval Shanghai_rev@8/center at 32.9 s
- imitation/126 eval Shanghai_w105@8/center at 64.1 s
- imitation/126 eval YasMarina_rev_w105@8/center at 46.3 s
- imitation/126 eval YasMarina_w105@8/center at 98.4 s
- imitation/126 eval Catalunya_w105@5/center at 94.8 s
- imitation/126 eval Montreal_w105@5/center at 6.1 s
- imitation/126 eval Shanghai_rev_w105@5/center at 32.7 s
- imitation/126 eval Shanghai_w105@5/center at 79.1 s
- imitation/126 eval YasMarina_rev_w105@5/center at 53.8 s
- imitation/126 eval Catalunya_w105@8/random at 327.5 s
- imitation/126 eval MoscowRaceway_rev@8/random at 11.4 s
- imitation/126 eval MoscowRaceway_w105@8/random at 94.1 s
- imitation/126 eval Shanghai_rev_w105@8/random at 28.4 s
- imitation/126 eval Shanghai_w105@8/random at 65.1 s
- imitation/126 eval Silverstone_rev_w105@8/random at 184.3 s
- imitation/126 eval YasMarina_rev_w105@8/random at 46.3 s
- imitation/126 worst Catalunya@8/center at 60.4 s
- imitation/126 worst Catalunya_w105@8/center at 156.0 s
- imitation/126 worst Montreal_rev@8/center at 46.0 s
- imitation/126 worst Shanghai_rev@8/center at 62.8 s
- imitation/126 worst Shanghai_rev_w105@8/center at 93.2 s
- imitation/126 worst Shanghai_w105@8/center at 17.8 s
- imitation/126 worst YasMarina@8/center at 42.3 s
- imitation/126 worst YasMarina_rev@8/center at 38.0 s
- imitation/126 worst YasMarina_rev_w105@8/center at 120.8 s
- imitation/126 worst YasMarina_w105@8/center at 37.5 s
- imitation/126 low Catalunya_w105@3.5/center at 29.5 s
- imitation/126 low Montreal_w105@2/center at 9.5 s
- imitation/126 low SaoPaulo@2/center at 95.4 s
- imitation/126 low Shanghai_rev_w105@2.5/center at 260.7 s
- imitation/126 low Silverstone_rev_w105@2/center at 7.5 s
- imitation/126 low YasMarina_rev_w105@3.5/center at 64.3 s
- imitation/126 open Catalunya_w902@8/center at 20.7 s
- imitation/126 open MexicoCity_rev_w901@8/center at 40.0 s
- imitation/126 open MexicoCity_rev_w902@8/center at 10.8 s
- imitation/126 open MexicoCity_w902@8/center at 104.9 s
- imitation/126 open Montreal_rev_w902@8/center at 24.7 s
- imitation/126 open Montreal_w901@8/center at 54.9 s
- imitation/126 open Montreal_w902@8/center at 14.3 s
- imitation/126 open MoscowRaceway_rev_w902@8/center at 35.6 s
- imitation/126 open MoscowRaceway_w902@8/center at 65.7 s
- imitation/126 open SaoPaulo_rev_w902@8/center at 9.7 s
- imitation/126 open SaoPaulo_w902@8/center at 9.8 s
- imitation/126 open Shanghai_rev_w901@8/center at 178.8 s
- imitation/126 open Shanghai_rev_w902@8/center at 28.3 s
- imitation/126 open Shanghai_w901@8/center at 199.8 s
- imitation/126 open Shanghai_w902@8/center at 33.4 s
- imitation/126 open Silverstone_rev_w901@8/center at 40.8 s
- imitation/126 open Silverstone_rev_w902@8/center at 29.3 s
- imitation/126 open Silverstone_w902@8/center at 10.4 s
- imitation/126 open Sochi_rev_w902@8/center at 45.2 s
- imitation/126 open Sochi_w902@8/center at 30.0 s
- imitation/126 open YasMarina_rev_w901@8/center at 3.6 s
- imitation/126 open YasMarina_rev_w902@8/center at 7.2 s
- imitation/126 open YasMarina_w901@8/center at 219.8 s
- imitation/126 open YasMarina_w902@8/center at 33.3 s
- imitation/126 open Zandvoort_rev_w902@8/center at 56.4 s
- imitation/126 open Catalunya_w902@5/center at 60.9 s
- imitation/126 open MexicoCity_rev_w902@5/center at 44.6 s
- imitation/126 open MexicoCity_w902@5/center at 93.6 s
- imitation/126 open Montreal_rev_w901@5/center at 24.5 s
- imitation/126 open Montreal_rev_w902@5/center at 67.1 s
- imitation/126 open Montreal_w901@5/center at 78.2 s
- imitation/126 open Montreal_w902@5/center at 71.4 s
- imitation/126 open Shanghai_rev_w901@5/center at 50.4 s
- imitation/126 open Shanghai_w901@5/center at 72.0 s
- imitation/126 open Shanghai_w902@5/center at 101.8 s
- imitation/126 open Silverstone_rev_w902@5/center at 33.6 s
- imitation/126 open Silverstone_w902@5/center at 73.1 s
- imitation/126 open YasMarina_rev_w901@5/center at 3.7 s
- imitation/126 open YasMarina_rev_w902@5/center at 176.9 s
- imitation/126 open YasMarina_w901@5/center at 51.3 s
- ppo_s2r/12 eval Shanghai_rev_w105@8/center at 23.1 s
- ppo_s2r/12 eval YasMarina_w105@8/center at 52.1 s
- ppo_s2r/12 eval Catalunya_w105@5/center at 76.3 s
- ppo_s2r/12 eval Montreal_w105@5/center at 6.2 s
- ppo_s2r/12 eval Shanghai_rev_w105@5/center at 22.9 s
- ppo_s2r/12 eval Shanghai_w105@5/center at 176.6 s
- ppo_s2r/12 eval YasMarina_w105@5/center at 54.8 s
- ppo_s2r/12 eval Zandvoort_w105@5/center at 59.8 s
- ppo_s2r/12 eval Shanghai_rev_w105@8/random at 22.9 s
- ppo_s2r/12 worst Shanghai_rev_w105@8/center at 81.5 s
- ppo_s2r/12 worst Shanghai_w105@8/center at 16.0 s
- ppo_s2r/12 worst YasMarina_rev_w105@8/center at 203.4 s
- ppo_s2r/12 low Catalunya_w105@3.5/center at 30.2 s
- ppo_s2r/12 low Shanghai_rev_w105@2.5/center at 42.8 s
- ppo_s2r/12 low Silverstone_w105@2.5/center at 32.3 s
- ppo_s2r/12 low Sochi_rev_w105@4/center at 38.3 s
- ppo_s2r/12 low YasMarina_w105@4/center at 136.4 s
- ppo_s2r/12 open Catalunya_rev_w902@8/center at 117.6 s
- ppo_s2r/12 open MexicoCity_rev_w901@8/center at 162.0 s
- ppo_s2r/12 open MexicoCity_w901@8/center at 44.4 s
- ppo_s2r/12 open Montreal_rev_w901@8/center at 18.1 s
- ppo_s2r/12 open Montreal_rev_w902@8/center at 60.3 s
- ppo_s2r/12 open MoscowRaceway_rev_w902@8/center at 38.2 s
- ppo_s2r/12 open MoscowRaceway_w902@8/center at 27.5 s
- ppo_s2r/12 open SaoPaulo_w902@8/center at 86.6 s
- ppo_s2r/12 open Shanghai_rev_w901@8/center at 632.1 s
- ppo_s2r/12 open Shanghai_rev_w902@8/center at 197.3 s
- ppo_s2r/12 open Shanghai_w902@8/center at 91.0 s
- ppo_s2r/12 open Silverstone_rev_w902@8/center at 33.8 s
- ppo_s2r/12 open Silverstone_w902@8/center at 64.1 s
- ppo_s2r/12 open Sochi_rev_w902@8/center at 46.6 s
- ppo_s2r/12 open YasMarina_rev_w901@8/center at 63.6 s
- ppo_s2r/12 open YasMarina_rev_w902@8/center at 64.5 s
- ppo_s2r/12 open YasMarina_w901@8/center at 148.1 s
- ppo_s2r/12 open Catalunya_rev_w902@5/center at 531.0 s
- ppo_s2r/12 open MexicoCity_rev_w901@5/center at 80.4 s
- ppo_s2r/12 open MexicoCity_w901@5/center at 43.7 s
- ppo_s2r/12 open Montreal_rev_w901@5/center at 18.0 s
- ppo_s2r/12 open Montreal_rev_w902@5/center at 30.5 s
- ppo_s2r/12 open Montreal_w901@5/center at 109.2 s
- ppo_s2r/12 open MoscowRaceway_w902@5/center at 27.3 s
- ppo_s2r/12 open Shanghai_rev_w902@5/center at 205.6 s
- ppo_s2r/12 open Shanghai_w901@5/center at 49.5 s
- ppo_s2r/12 open Silverstone_rev_w902@5/center at 32.8 s
- ppo_s2r/12 open YasMarina_rev_w902@5/center at 27.4 s
- ppo_s2r/12u60 eval MexicoCity_rev_w105@8/center at 14.3 s
- ppo_s2r/12u60 eval Shanghai_rev_w105@8/center at 21.4 s
- ppo_s2r/12u60 eval Shanghai_w105@8/center at 57.4 s
- ppo_s2r/12u60 eval YasMarina_w105@8/center at 112.1 s
- ppo_s2r/12u60 eval Shanghai_rev_w105@5/center at 22.6 s
- ppo_s2r/12u60 eval YasMarina_w105@5/center at 507.6 s
- ppo_s2r/12u60 eval MexicoCity_rev_w105@8/random at 14.3 s
- ppo_s2r/12u60 eval Shanghai_rev_w105@8/random at 21.4 s
- ppo_s2r/12u60 eval Shanghai_w105@8/random at 57.6 s
- ppo_s2r/12u60 eval YasMarina_w105@8/random at 112.0 s
- ppo_s2r/12u60 worst MexicoCity_rev_w105@8/center at 54.9 s
- ppo_s2r/12u60 worst MexicoCity_w105@8/center at 109.1 s
- ppo_s2r/12u60 worst Montreal_w105@8/center at 104.2 s
- ppo_s2r/12u60 worst MoscowRaceway@8/center at 25.2 s
- ppo_s2r/12u60 worst Shanghai_rev_w105@8/center at 77.1 s
- ppo_s2r/12u60 worst Shanghai_w105@8/center at 127.1 s
- ppo_s2r/12u60 worst Silverstone_rev@8/center at 96.1 s
- ppo_s2r/12u60 worst Silverstone_rev_w105@8/center at 48.9 s
- ppo_s2r/12u60 worst Sochi_w105@8/center at 44.4 s
- ppo_s2r/12u60 worst YasMarina_w105@8/center at 5.6 s
- ppo_s2r/12u60 worst Zandvoort_rev@8/center at 75.0 s
- ppo_s2r/12u60 worst Zandvoort_rev_w105@8/center at 55.1 s
- ppo_s2r/12u60 low Catalunya_w105@3.5/center at 29.2 s
- ppo_s2r/12u60 low Montreal_w105@2/center at 154.2 s
- ppo_s2r/12u60 low Shanghai_rev_w105@2.5/center at 43.4 s
- ppo_s2r/12u60 low Sochi_rev_w105@4/center at 37.9 s
- ppo_s2r/12u60 open Catalunya_rev_w902@8/center at 7.6 s
- ppo_s2r/12u60 open Catalunya_w901@8/center at 95.7 s
- ppo_s2r/12u60 open MexicoCity_rev_w901@8/center at 455.9 s
- ppo_s2r/12u60 open MexicoCity_rev_w902@8/center at 76.6 s
- ppo_s2r/12u60 open MexicoCity_w901@8/center at 42.7 s
- ppo_s2r/12u60 open Montreal_rev_w901@8/center at 366.3 s
- ppo_s2r/12u60 open Montreal_rev_w902@8/center at 63.8 s
- ppo_s2r/12u60 open MoscowRaceway_w901@8/center at 42.0 s
- ppo_s2r/12u60 open MoscowRaceway_w902@8/center at 26.2 s
- ppo_s2r/12u60 open Shanghai_rev_w901@8/center at 38.7 s
- ppo_s2r/12u60 open Shanghai_rev_w902@8/center at 187.5 s
- ppo_s2r/12u60 open Shanghai_w901@8/center at 159.2 s
- ppo_s2r/12u60 open Shanghai_w902@8/center at 10.4 s
- ppo_s2r/12u60 open Silverstone_rev_w902@8/center at 582.4 s
- ppo_s2r/12u60 open Silverstone_w902@8/center at 62.4 s
- ppo_s2r/12u60 open Sochi_rev_w902@8/center at 3.2 s
- ppo_s2r/12u60 open YasMarina_rev_w902@8/center at 7.3 s
- ppo_s2r/12u60 open YasMarina_w901@8/center at 44.7 s
- ppo_s2r/12u60 open YasMarina_w902@8/center at 10.3 s
- ppo_s2r/12u60 open Zandvoort_rev_w902@8/center at 33.9 s
- ppo_s2r/12u60 open Catalunya_rev_w902@5/center at 6.5 s
- ppo_s2r/12u60 open MexicoCity_w901@5/center at 43.2 s
- ppo_s2r/12u60 open Montreal_rev_w901@5/center at 17.4 s
- ppo_s2r/12u60 open Montreal_rev_w902@5/center at 96.9 s
- ppo_s2r/12u60 open Montreal_w902@5/center at 63.3 s
- ppo_s2r/12u60 open MoscowRaceway_w901@5/center at 117.7 s
- ppo_s2r/12u60 open Shanghai_rev_w901@5/center at 40.9 s
- ppo_s2r/12u60 open Shanghai_rev_w902@5/center at 102.4 s
- ppo_s2r/12u60 open Silverstone_rev_w902@5/center at 140.4 s
- ppo_s2r/12u60 open Sochi_rev_w902@5/center at 3.1 s
- ppo_s2r/12u60 open YasMarina_rev_w902@5/center at 172.0 s
- ppo_s2r/12u60 open YasMarina_w901@5/center at 85.0 s
- ppo_s2r/12u60 open YasMarina_w902@5/center at 11.1 s
- ppo_s2r/12u80 eval Catalunya_w105@8/center at 531.0 s
- ppo_s2r/12u80 eval MexicoCity_rev_w105@8/center at 14.9 s
- ppo_s2r/12u80 eval Shanghai_rev_w105@8/center at 21.7 s
- ppo_s2r/12u80 eval YasMarina@8/center at 62.4 s
- ppo_s2r/12u80 eval YasMarina_w105@8/center at 52.1 s
- ppo_s2r/12u80 eval Catalunya_w105@5/center at 531.0 s
- ppo_s2r/12u80 eval Shanghai_rev_w105@5/center at 22.7 s
- ppo_s2r/12u80 eval Shanghai_w105@5/center at 62.3 s
- ppo_s2r/12u80 eval YasMarina_w105@5/center at 148.4 s
- ppo_s2r/12u80 eval Catalunya_w105@8/random at 111.4 s
- ppo_s2r/12u80 eval MexicoCity_rev_w105@8/random at 15.0 s
- ppo_s2r/12u80 eval Shanghai_rev_w105@8/random at 21.7 s
- ppo_s2r/12u80 eval YasMarina@8/random at 62.6 s
- ppo_s2r/12u80 eval YasMarina_rev_w105@8/random at 127.8 s
- ppo_s2r/12u80 eval YasMarina_w105@8/random at 66.9 s
- ppo_s2r/12u80 worst Catalunya_w105@8/center at 123.4 s
- ppo_s2r/12u80 worst Shanghai_rev_w105@8/center at 77.9 s
- ppo_s2r/12u80 worst Shanghai_w105@8/center at 127.9 s
- ppo_s2r/12u80 worst Sochi_w105@8/center at 147.1 s
- ppo_s2r/12u80 worst YasMarina_rev_w105@8/center at 108.4 s
- ppo_s2r/12u80 worst YasMarina_w105@8/center at 5.7 s
- ppo_s2r/12u80 low Catalunya_w105@3.5/center at 28.8 s
- ppo_s2r/12u80 low Montreal_w105@2/center at 9.5 s
- ppo_s2r/12u80 low Shanghai_rev_w105@2.5/center at 43.4 s
- ppo_s2r/12u80 low YasMarina_rev_w105@3.5/center at 53.9 s
- ppo_s2r/12u80 open Catalunya_rev_w902@8/center at 6.9 s
- ppo_s2r/12u80 open MexicoCity_rev_w901@8/center at 71.2 s
- ppo_s2r/12u80 open MexicoCity_w901@8/center at 42.2 s
- ppo_s2r/12u80 open Montreal_rev_w901@8/center at 16.6 s
- ppo_s2r/12u80 open Montreal_rev_w902@8/center at 56.8 s
- ppo_s2r/12u80 open MoscowRaceway_w902@8/center at 26.5 s
- ppo_s2r/12u80 open Shanghai_rev_w901@8/center at 38.2 s
- ppo_s2r/12u80 open Shanghai_rev_w902@8/center at 54.4 s
- ppo_s2r/12u80 open Shanghai_w901@8/center at 48.2 s
- ppo_s2r/12u80 open Shanghai_w902@8/center at 84.3 s
- ppo_s2r/12u80 open Silverstone_rev_w902@8/center at 29.9 s
- ppo_s2r/12u80 open Silverstone_w901@8/center at 582.4 s
- ppo_s2r/12u80 open Silverstone_w902@8/center at 60.8 s
- ppo_s2r/12u80 open Sochi_rev_w902@8/center at 45.0 s
- ppo_s2r/12u80 open YasMarina_w901@8/center at 45.4 s
- ppo_s2r/12u80 open YasMarina_w902@8/center at 85.9 s
- ppo_s2r/12u80 open Zandvoort_rev_w902@8/center at 34.3 s
- ppo_s2r/12u80 open MexicoCity_rev_w901@5/center at 73.9 s
- ppo_s2r/12u80 open MexicoCity_rev_w902@5/center at 123.9 s
- ppo_s2r/12u80 open MexicoCity_w901@5/center at 42.9 s
- ppo_s2r/12u80 open Montreal_rev_w901@5/center at 17.6 s
- ppo_s2r/12u80 open Montreal_rev_w902@5/center at 30.9 s
- ppo_s2r/12u80 open Shanghai_rev_w901@5/center at 39.1 s
- ppo_s2r/12u80 open Shanghai_rev_w902@5/center at 83.0 s
- ppo_s2r/12u80 open Shanghai_w901@5/center at 163.0 s
- ppo_s2r/12u80 open Shanghai_w902@5/center at 89.7 s
- ppo_s2r/12u80 open Silverstone_rev_w902@5/center at 63.8 s
- ppo_s2r/12u80 open Sochi_rev_w902@5/center at 105.4 s
- ppo_s2r/12u80 open YasMarina_w901@5/center at 43.2 s
