# model comparison (2026-11-25 12:45, 2 lap(s), reference = ppo_s2r/5)

tracks: eval 44 x [(8.0, 'center'), (5.0, 'center'), (8.0, 'random')], worst-case 44 x cap 8 centerline, open-space (big) 44 x [(8.0, 'center'), (5.0, 'center')]

| model | eval success | low caps 2-4 m/s | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter |
|---|---|---|---|---|---|---|---|
| ppo_s2r/5 | 91.7% (242/264) | 85.2% (75/88) | 90.9% (80/88) | 67.0% (118/176) | reference | 4.33 m/s | 0.0124 rad |
| ppo_s2r/12150 | 89.0% (235/264) | 90.9% (80/88) | 90.9% (80/88) | 69.9% (123/176) | 3.4% faster on 156 runs | 4.43 m/s | 0.0123 rad |
| ppo_s2r/12170 | 91.3% (241/264) | 89.8% (79/88) | 89.8% (79/88) | 72.2% (127/176) | 4.6% faster on 160 runs | 4.46 m/s | 0.0123 rad |
| ppo_s2r/12190 | 90.9% (240/264) | 89.8% (79/88) | 90.9% (80/88) | 73.9% (130/176) | 4.3% faster on 159 runs | 4.47 m/s | 0.0121 rad |
| ppo_s2r/12200 | 91.3% (241/264) | 87.5% (77/88) | 89.8% (79/88) | 69.3% (122/176) | 6.7% faster on 162 runs | 4.56 m/s | 0.0118 rad |

## uncapped 2-lap time per track (centerline start; - = did not finish)

| track | ppo_s2r/5 | ppo_s2r/12150 | ppo_s2r/12170 | ppo_s2r/12190 | ppo_s2r/12200 |
|---|---|---|---|---|---|
| Catalunya | 185.7 | 180.5 | 178.8 | 179.4 | 175.6 |
| Catalunya_rev | 186.0 | 181.6 | 179.6 | 181.3 | 175.6 |
| Catalunya_rev_w105 | 193.3 | 182.4 | 180.2 | 179.4 | 178.0 |
| Catalunya_w105 | 188.7 | 179.5 | 174.7 | 175.6 | 172.7 |
| IMS | 119.0 | 114.8 | 114.8 | 118.3 | 113.7 |
| IMS_rev | 113.5 | 110.2 | 106.6 | 106.5 | 100.6 |
| IMS_rev_w105 | 116.7 | 109.3 | 107.5 | 105.8 | 102.9 |
| IMS_w105 | 114.4 | 111.2 | 111.2 | 113.2 | 109.4 |
| MexicoCity | 169.0 | 161.8 | 159.8 | 159.5 | 156.3 |
| MexicoCity_rev | 167.8 | 164.5 | 159.7 | 159.2 | 154.4 |
| MexicoCity_rev_w105 | 162.5 | 156.3 | 154.1 | 153.5 | 149.0 |
| MexicoCity_w105 | 166.4 | 154.8 | 154.2 | 155.2 | 153.4 |
| Montreal | 129.6 | 125.5 | 123.2 | 123.5 | 121.4 |
| Montreal_rev | 131.9 | 130.7 | 128.4 | 127.9 | 123.9 |
| Montreal_rev_w105 | 128.0 | 125.7 | 124.9 | 123.1 | 119.4 |
| Montreal_w105 | 127.0 | 123.7 | 122.3 | 121.7 | 117.7 |
| MoscowRaceway | 152.5 | 151.8 | 148.9 | 148.4 | 144.6 |
| MoscowRaceway_rev | 157.3 | 151.2 | 149.5 | 150.3 | 147.1 |
| MoscowRaceway_rev_w105 | 153.2 | 146.5 | 145.8 | 146.3 | 145.2 |
| MoscowRaceway_w105 | 146.5 | 144.5 | 142.2 | 142.8 | 140.2 |
| SaoPaulo | 154.4 | 151.6 | 151.4 | 153.0 | 148.7 |
| SaoPaulo_rev | 153.0 | 149.0 | 147.1 | 146.9 | 143.9 |
| SaoPaulo_rev_w105 | 153.6 | 145.7 | 144.4 | 143.2 | 139.9 |
| SaoPaulo_w105 | 152.3 | 147.0 | 145.6 | 148.6 | 143.9 |
| Shanghai | 227.7 | 219.9 | 217.0 | 218.2 | 213.4 |
| Shanghai_rev | 230.8 | 222.6 | 219.6 | 221.0 | 213.9 |
| Shanghai_rev_w105 | - | - | - | - | - |
| Shanghai_w105 | 226.1 | - | - | - | 207.5 |
| Silverstone | 199.8 | 194.2 | 191.5 | 193.6 | 189.3 |
| Silverstone_rev | 198.8 | 194.3 | 191.5 | 193.4 | 187.4 |
| Silverstone_rev_w105 | 201.9 | 193.1 | 191.0 | 192.2 | 187.3 |
| Silverstone_w105 | 201.8 | 191.9 | 190.6 | 191.6 | 187.5 |
| Sochi | 206.7 | 201.5 | 200.2 | 201.5 | 196.7 |
| Sochi_rev | 206.0 | 203.6 | 199.1 | 198.9 | 193.5 |
| Sochi_rev_w105 | 206.0 | 197.6 | 194.8 | 194.6 | 188.8 |
| Sochi_w105 | 210.7 | 197.1 | 196.0 | 198.0 | 193.8 |
| YasMarina | 196.2 | 191.5 | 189.2 | 186.3 | 180.9 |
| YasMarina_rev | 192.9 | 186.4 | 183.5 | 184.0 | 179.4 |
| YasMarina_rev_w105 | 194.9 | 184.0 | 181.6 | - | - |
| YasMarina_w105 | - | - | - | - | - |
| Zandvoort | 171.6 | 167.3 | 165.8 | 166.1 | 163.0 |
| Zandvoort_rev | 171.6 | 168.5 | 167.3 | 168.9 | 164.2 |
| Zandvoort_rev_w105 | 173.5 | 164.9 | 162.8 | 164.5 | 159.4 |
| Zandvoort_w105 | 172.6 | 163.1 | 161.7 | 162.0 | 158.8 |

## failures (473)

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
- ppo_s2r/5 eval Shanghai_rev_w105@8/center#1 at 111.9 s
- ppo_s2r/5 eval Sochi_rev_w105@8/center#1 at 26.6 s
- ppo_s2r/5 eval YasMarina_rev_w105@8/center#1 at 507.6 s
- ppo_s2r/5 eval YasMarina_w105@8/center#1 at 138.8 s
- ppo_s2r/5 eval Catalunya_w105@5/center#1 at 116.4 s
- ppo_s2r/5 eval Montreal_w105@5/center#1 at 114.0 s
- ppo_s2r/5 eval Shanghai_rev_w105@5/center#1 at 110.6 s
- ppo_s2r/5 eval Sochi_rev_w105@5/center#1 at 27.2 s
- ppo_s2r/5 eval YasMarina_w105@5/center#1 at 208.6 s
- ppo_s2r/5 eval Shanghai_rev_w105@8/random#1 at 112.0 s
- ppo_s2r/5 eval Sochi_rev_w105@8/random#1 at 26.6 s
- ppo_s2r/5 eval YasMarina_rev_w105@8/random#1 at 507.6 s
- ppo_s2r/5 eval YasMarina_w105@8/random#1 at 140.2 s
- ppo_s2r/5 worst Shanghai_rev_w105@8/center#1 at 103.9 s
- ppo_s2r/5 worst Shanghai_w105@8/center#1 at 126.1 s
- ppo_s2r/5 worst YasMarina_rev_w105@8/center#1 at 507.6 s
- ppo_s2r/5 worst YasMarina_w105@8/center#1 at 113.5 s
- ppo_s2r/5 low Catalunya_w105@4/center#1 at 133.0 s
- ppo_s2r/5 low Montreal_w105@2.5/center#1 at 87.6 s
- ppo_s2r/5 low Shanghai_rev_w105@3/center#1 at 165.3 s
- ppo_s2r/5 low Shanghai_w105@3.5/center#1 at 25.1 s
- ppo_s2r/5 low Silverstone_w105@3/center#1 at 257.7 s
- ppo_s2r/5 low YasMarina_rev_w105@4/center#1 at 11.3 s
- ppo_s2r/5 low YasMarina_w105@2/center#1 at 98.5 s
- ppo_s2r/5 open Catalunya_rev_w902@8/center#1 at 180.1 s
- ppo_s2r/5 open MexicoCity_rev_w901@8/center#1 at 455.9 s
- ppo_s2r/5 open MexicoCity_w901@8/center#1 at 14.0 s
- ppo_s2r/5 open Montreal_rev_w901@8/center#1 at 17.7 s
- ppo_s2r/5 open Montreal_rev_w902@8/center#1 at 48.3 s
- ppo_s2r/5 open Montreal_w901@8/center#1 at 115.5 s
- ppo_s2r/5 open MoscowRaceway_rev_w902@8/center#1 at 161.8 s
- ppo_s2r/5 open MoscowRaceway_w902@8/center#1 at 70.0 s
- ppo_s2r/5 open SaoPaulo_w902@8/center#1 at 102.7 s
- ppo_s2r/5 open Shanghai_rev_w901@8/center#1 at 214.1 s
- ppo_s2r/5 open Shanghai_w902@8/center#1 at 34.7 s
- ppo_s2r/5 open Silverstone_rev_w902@8/center#1 at 171.3 s
- ppo_s2r/5 open Silverstone_w902@8/center#1 at 170.9 s
- ppo_s2r/5 open Sochi_rev_w902@8/center#1 at 204.4 s
- ppo_s2r/5 open YasMarina_rev_w902@8/center#1 at 29.5 s
- ppo_s2r/5 open YasMarina_w901@8/center#1 at 125.4 s
- ppo_s2r/5 open Catalunya_rev_w902@5/center#1 at 531.0 s
- ppo_s2r/5 open MexicoCity_rev_w901@5/center#1 at 100.1 s
- ppo_s2r/5 open MexicoCity_w901@5/center#1 at 13.4 s
- ppo_s2r/5 open Montreal_rev_w901@5/center#1 at 17.7 s
- ppo_s2r/5 open MoscowRaceway_w902@5/center#1 at 100.9 s
- ppo_s2r/5 open Shanghai_rev_w901@5/center#1 at 95.9 s
- ppo_s2r/5 open Shanghai_rev_w902@5/center#1 at 147.8 s
- ppo_s2r/5 open Shanghai_w901@5/center#1 at 80.6 s
- ppo_s2r/5 open Silverstone_rev_w902@5/center#1 at 582.4 s
- ppo_s2r/5 open Silverstone_w901@5/center#1 at 192.5 s
- ppo_s2r/5 open Silverstone_w902@5/center#1 at 69.4 s
- ppo_s2r/5 open YasMarina_w901@5/center#1 at 235.7 s
- ppo_s2r/12150 eval Shanghai_rev_w105@8/center at 21.2 s
- ppo_s2r/12150 eval Shanghai_w105@8/center at 59.1 s
- ppo_s2r/12150 eval YasMarina_w105@8/center at 49.9 s
- ppo_s2r/12150 eval Montreal_w105@5/center at 70.1 s
- ppo_s2r/12150 eval MoscowRaceway_w105@5/center at 1.4 s
- ppo_s2r/12150 eval Shanghai_rev_w105@5/center at 22.7 s
- ppo_s2r/12150 eval Shanghai_w105@5/center at 61.0 s
- ppo_s2r/12150 eval YasMarina_rev_w105@5/center at 40.7 s
- ppo_s2r/12150 eval YasMarina_w105@5/center at 507.6 s
- ppo_s2r/12150 eval Catalunya_w105@8/random at 111.9 s
- ppo_s2r/12150 eval Shanghai_rev_w105@8/random at 20.9 s
- ppo_s2r/12150 eval Shanghai_w105@8/random at 59.1 s
- ppo_s2r/12150 eval Sochi_rev_w105@8/random at 1.1 s
- ppo_s2r/12150 eval YasMarina_w105@8/random at 50.0 s
- ppo_s2r/12150 worst Catalunya_w105@8/center at 125.7 s
- ppo_s2r/12150 worst Shanghai_rev_w105@8/center at 78.1 s
- ppo_s2r/12150 worst Shanghai_w105@8/center at 127.2 s
- ppo_s2r/12150 worst YasMarina_w105@8/center at 6.1 s
- ppo_s2r/12150 low Montreal_w105@2/center at 9.4 s
- ppo_s2r/12150 low Shanghai_rev_w105@2.5/center at 42.4 s
- ppo_s2r/12150 low YasMarina_rev_w105@3.5/center at 52.4 s
- ppo_s2r/12150 low YasMarina_w105@4/center at 507.6 s
- ppo_s2r/12150 open Catalunya_rev_w902@8/center at 18.0 s
- ppo_s2r/12150 open MexicoCity_rev_w901@8/center at 72.3 s
- ppo_s2r/12150 open MexicoCity_rev_w902@8/center at 38.7 s
- ppo_s2r/12150 open MexicoCity_w901@8/center at 42.2 s
- ppo_s2r/12150 open Montreal_rev_w901@8/center at 16.5 s
- ppo_s2r/12150 open Montreal_rev_w902@8/center at 60.0 s
- ppo_s2r/12150 open Montreal_w901@8/center at 37.8 s
- ppo_s2r/12150 open Montreal_w902@8/center at 45.8 s
- ppo_s2r/12150 open MoscowRaceway_w902@8/center at 55.1 s
- ppo_s2r/12150 open Shanghai_rev_w901@8/center at 38.5 s
- ppo_s2r/12150 open Shanghai_rev_w902@8/center at 79.7 s
- ppo_s2r/12150 open Shanghai_w902@8/center at 2.3 s
- ppo_s2r/12150 open Silverstone_rev_w902@8/center at 31.4 s
- ppo_s2r/12150 open Silverstone_w902@8/center at 63.1 s
- ppo_s2r/12150 open Sochi_rev_w902@8/center at 47.7 s
- ppo_s2r/12150 open YasMarina_rev_w901@8/center at 8.7 s
- ppo_s2r/12150 open YasMarina_w901@8/center at 140.6 s
- ppo_s2r/12150 open YasMarina_w902@8/center at 40.0 s
- ppo_s2r/12150 open Catalunya_rev_w902@5/center at 17.6 s
- ppo_s2r/12150 open MexicoCity_w901@5/center at 42.5 s
- ppo_s2r/12150 open Montreal_rev_w901@5/center at 17.5 s
- ppo_s2r/12150 open Shanghai_rev_w902@5/center at 83.9 s
- ppo_s2r/12150 open Shanghai_w901@5/center at 49.6 s
- ppo_s2r/12150 open Shanghai_w902@5/center at 91.6 s
- ppo_s2r/12150 open Silverstone_rev_w902@5/center at 33.9 s
- ppo_s2r/12150 open Silverstone_w901@5/center at 65.0 s
- ppo_s2r/12150 open Silverstone_w902@5/center at 66.5 s
- ppo_s2r/12150 open YasMarina_w901@5/center at 43.7 s
- ppo_s2r/12150 eval Catalunya_w105@8/center#1 at 17.2 s
- ppo_s2r/12150 eval Shanghai_rev_w105@8/center#1 at 105.6 s
- ppo_s2r/12150 eval Shanghai_w105@8/center#1 at 18.1 s
- ppo_s2r/12150 eval YasMarina_rev_w105@8/center#1 at 507.6 s
- ppo_s2r/12150 eval YasMarina_w105@8/center#1 at 44.3 s
- ppo_s2r/12150 eval Montreal_w105@5/center#1 at 112.5 s
- ppo_s2r/12150 eval Shanghai_rev_w105@5/center#1 at 108.4 s
- ppo_s2r/12150 eval Shanghai_w105@5/center#1 at 130.2 s
- ppo_s2r/12150 eval YasMarina_rev_w105@5/center#1 at 9.6 s
- ppo_s2r/12150 eval YasMarina_w105@5/center#1 at 46.1 s
- ppo_s2r/12150 eval Catalunya_w105@8/random#1 at 531.0 s
- ppo_s2r/12150 eval Shanghai_rev_w105@8/random#1 at 104.9 s
- ppo_s2r/12150 eval Shanghai_w105@8/random#1 at 18.3 s
- ppo_s2r/12150 eval YasMarina_rev_w105@8/random#1 at 507.6 s
- ppo_s2r/12150 eval YasMarina_w105@8/random#1 at 43.4 s
- ppo_s2r/12150 worst Shanghai_rev_w105@8/center#1 at 99.1 s
- ppo_s2r/12150 worst Shanghai_w105@8/center#1 at 6.4 s
- ppo_s2r/12150 worst YasMarina_rev_w105@8/center#1 at 134.4 s
- ppo_s2r/12150 worst YasMarina_w105@8/center#1 at 14.6 s
- ppo_s2r/12150 low Montreal_w105@2.5/center#1 at 87.3 s
- ppo_s2r/12150 low Shanghai_rev_w105@3/center#1 at 163.1 s
- ppo_s2r/12150 low Silverstone_w105@3/center#1 at 256.0 s
- ppo_s2r/12150 low YasMarina_w105@2/center#1 at 507.6 s
- ppo_s2r/12150 open Catalunya_rev_w902@8/center#1 at 79.8 s
- ppo_s2r/12150 open MexicoCity_rev_w901@8/center#1 at 10.1 s
- ppo_s2r/12150 open MexicoCity_w901@8/center#1 at 14.3 s
- ppo_s2r/12150 open Montreal_rev_w901@8/center#1 at 15.5 s
- ppo_s2r/12150 open Montreal_w901@8/center#1 at 87.4 s
- ppo_s2r/12150 open Montreal_w902@8/center#1 at 33.2 s
- ppo_s2r/12150 open MoscowRaceway_w902@8/center#1 at 19.2 s
- ppo_s2r/12150 open Shanghai_rev_w901@8/center#1 at 89.6 s
- ppo_s2r/12150 open Shanghai_rev_w902@8/center#1 at 25.6 s
- ppo_s2r/12150 open Shanghai_w902@8/center#1 at 33.0 s
- ppo_s2r/12150 open Silverstone_rev_w902@8/center#1 at 59.6 s
- ppo_s2r/12150 open Silverstone_w902@8/center#1 at 66.2 s
- ppo_s2r/12150 open Sochi_rev_w902@8/center#1 at 83.6 s
- ppo_s2r/12150 open YasMarina_rev_w901@8/center#1 at 507.6 s
- ppo_s2r/12150 open YasMarina_w901@8/center#1 at 21.2 s
- ppo_s2r/12150 open YasMarina_w902@8/center#1 at 56.4 s
- ppo_s2r/12150 open Catalunya_rev_w902@5/center#1 at 81.6 s
- ppo_s2r/12150 open MexicoCity_w901@5/center#1 at 12.1 s
- ppo_s2r/12150 open Montreal_rev_w901@5/center#1 at 17.3 s
- ppo_s2r/12150 open Shanghai_rev_w902@5/center#1 at 28.2 s
- ppo_s2r/12150 open Shanghai_w901@5/center#1 at 195.6 s
- ppo_s2r/12150 open Shanghai_w902@5/center#1 at 40.9 s
- ppo_s2r/12150 open Silverstone_rev_w902@5/center#1 at 582.4 s
- ppo_s2r/12150 open Silverstone_w902@5/center#1 at 69.2 s
- ppo_s2r/12150 open YasMarina_w901@5/center#1 at 121.1 s
- ppo_s2r/12170 eval Shanghai_rev_w105@8/center at 20.9 s
- ppo_s2r/12170 eval Shanghai_w105@8/center at 58.7 s
- ppo_s2r/12170 eval YasMarina_w105@8/center at 49.5 s
- ppo_s2r/12170 eval Montreal_w105@5/center at 6.0 s
- ppo_s2r/12170 eval MoscowRaceway_w105@5/center at 1.4 s
- ppo_s2r/12170 eval Shanghai_rev_w105@5/center at 22.8 s
- ppo_s2r/12170 eval Shanghai_w105@5/center at 60.7 s
- ppo_s2r/12170 eval YasMarina_rev_w105@5/center at 40.1 s
- ppo_s2r/12170 eval YasMarina_w105@5/center at 507.6 s
- ppo_s2r/12170 eval Shanghai_rev_w105@8/random at 20.8 s
- ppo_s2r/12170 eval Sochi_rev_w105@8/random at 1.1 s
- ppo_s2r/12170 eval YasMarina_w105@8/random at 139.7 s
- ppo_s2r/12170 worst Shanghai_rev_w105@8/center at 77.0 s
- ppo_s2r/12170 worst Shanghai_w105@8/center at 14.8 s
- ppo_s2r/12170 worst YasMarina_rev_w105@8/center at 507.6 s
- ppo_s2r/12170 worst YasMarina_w105@8/center at 6.0 s
- ppo_s2r/12170 low Montreal_w105@2/center at 8.8 s
- ppo_s2r/12170 low MoscowRaceway_w105@4/center at 1.4 s
- ppo_s2r/12170 low Shanghai_rev_w105@2.5/center at 41.8 s
- ppo_s2r/12170 low Silverstone_w105@2.5/center at 31.0 s
- ppo_s2r/12170 low Sochi_rev_w105@4/center at 38.0 s
- ppo_s2r/12170 low YasMarina_w105@4/center at 129.8 s
- ppo_s2r/12170 open Catalunya_rev_w902@8/center at 16.5 s
- ppo_s2r/12170 open MexicoCity_rev_w901@8/center at 71.7 s
- ppo_s2r/12170 open MexicoCity_w901@8/center at 41.6 s
- ppo_s2r/12170 open Montreal_rev_w901@8/center at 16.2 s
- ppo_s2r/12170 open Montreal_rev_w902@8/center at 60.9 s
- ppo_s2r/12170 open Shanghai_rev_w901@8/center at 37.3 s
- ppo_s2r/12170 open Shanghai_rev_w902@8/center at 79.5 s
- ppo_s2r/12170 open Shanghai_w902@8/center at 87.2 s
- ppo_s2r/12170 open Silverstone_rev_w902@8/center at 30.4 s
- ppo_s2r/12170 open Silverstone_w902@8/center at 62.2 s
- ppo_s2r/12170 open Sochi_rev_w902@8/center at 144.8 s
- ppo_s2r/12170 open YasMarina_rev_w901@8/center at 7.1 s
- ppo_s2r/12170 open YasMarina_rev_w902@8/center at 159.4 s
- ppo_s2r/12170 open YasMarina_w901@8/center at 45.1 s
- ppo_s2r/12170 open YasMarina_w902@8/center at 164.3 s
- ppo_s2r/12170 open Catalunya_rev_w902@5/center at 531.0 s
- ppo_s2r/12170 open MexicoCity_rev_w902@5/center at 39.1 s
- ppo_s2r/12170 open MexicoCity_w901@5/center at 42.0 s
- ppo_s2r/12170 open Montreal_rev_w901@5/center at 17.4 s
- ppo_s2r/12170 open MoscowRaceway_w902@5/center at 55.1 s
- ppo_s2r/12170 open Shanghai_rev_w902@5/center at 202.2 s
- ppo_s2r/12170 open Shanghai_w901@5/center at 48.3 s
- ppo_s2r/12170 open Silverstone_rev_w902@5/center at 32.7 s
- ppo_s2r/12170 open Silverstone_w902@5/center at 67.3 s
- ppo_s2r/12170 open YasMarina_w901@5/center at 248.6 s
- ppo_s2r/12170 eval Shanghai_rev_w105@8/center#1 at 104.0 s
- ppo_s2r/12170 eval Shanghai_w105@8/center#1 at 17.8 s
- ppo_s2r/12170 eval YasMarina_rev_w105@8/center#1 at 9.6 s
- ppo_s2r/12170 eval YasMarina_w105@8/center#1 at 132.5 s
- ppo_s2r/12170 eval Montreal_w105@5/center#1 at 112.3 s
- ppo_s2r/12170 eval Shanghai_rev_w105@5/center#1 at 108.7 s
- ppo_s2r/12170 eval Shanghai_w105@5/center#1 at 18.5 s
- ppo_s2r/12170 eval YasMarina_rev_w105@5/center#1 at 9.1 s
- ppo_s2r/12170 eval YasMarina_w105@5/center#1 at 202.2 s
- ppo_s2r/12170 eval Shanghai_rev_w105@8/random#1 at 104.0 s
- ppo_s2r/12170 eval Shanghai_w105@8/random#1 at 17.9 s
- ppo_s2r/12170 worst Shanghai_rev_w105@8/center#1 at 97.6 s
- ppo_s2r/12170 worst Shanghai_w105@8/center#1 at 6.1 s
- ppo_s2r/12170 worst Sochi_rev_w105@8/center#1 at 101.9 s
- ppo_s2r/12170 worst YasMarina_rev_w105@8/center#1 at 38.8 s
- ppo_s2r/12170 worst YasMarina_w105@8/center#1 at 14.2 s
- ppo_s2r/12170 low Montreal_w105@2.5/center#1 at 88.9 s
- ppo_s2r/12170 low Shanghai_rev_w105@3/center#1 at 163.9 s
- ppo_s2r/12170 low YasMarina_w105@2/center#1 at 428.6 s
- ppo_s2r/12170 open Catalunya_rev_w902@8/center#1 at 77.6 s
- ppo_s2r/12170 open MexicoCity_rev_w901@8/center#1 at 11.4 s
- ppo_s2r/12170 open MexicoCity_w901@8/center#1 at 12.1 s
- ppo_s2r/12170 open Montreal_rev_w901@8/center#1 at 15.5 s
- ppo_s2r/12170 open Montreal_rev_w902@8/center#1 at 47.5 s
- ppo_s2r/12170 open Montreal_w902@8/center#1 at 98.7 s
- ppo_s2r/12170 open MoscowRaceway_w902@8/center#1 at 19.0 s
- ppo_s2r/12170 open Shanghai_rev_w902@8/center#1 at 26.2 s
- ppo_s2r/12170 open Shanghai_w902@8/center#1 at 32.8 s
- ppo_s2r/12170 open Silverstone_rev_w902@8/center#1 at 58.6 s
- ppo_s2r/12170 open Silverstone_w902@8/center#1 at 66.3 s
- ppo_s2r/12170 open Sochi_rev_w902@8/center#1 at 187.2 s
- ppo_s2r/12170 open YasMarina_w901@8/center#1 at 21.4 s
- ppo_s2r/12170 open Catalunya_rev_w902@5/center#1 at 531.0 s
- ppo_s2r/12170 open MexicoCity_w901@5/center#1 at 455.9 s
- ppo_s2r/12170 open Montreal_rev_w901@5/center#1 at 18.7 s
- ppo_s2r/12170 open Shanghai_rev_w902@5/center#1 at 27.2 s
- ppo_s2r/12170 open Shanghai_w901@5/center#1 at 421.4 s
- ppo_s2r/12170 open Shanghai_w902@5/center#1 at 35.8 s
- ppo_s2r/12170 open Silverstone_rev_w902@5/center#1 at 60.7 s
- ppo_s2r/12170 open Silverstone_w902@5/center#1 at 69.5 s
- ppo_s2r/12170 open Sochi_rev_w902@5/center#1 at 93.9 s
- ppo_s2r/12170 open YasMarina_rev_w902@5/center#1 at 175.9 s
- ppo_s2r/12170 open YasMarina_w901@5/center#1 at 18.8 s
- ppo_s2r/12190 eval Shanghai_rev_w105@8/center at 21.0 s
- ppo_s2r/12190 eval Shanghai_w105@8/center at 163.7 s
- ppo_s2r/12190 eval YasMarina_rev_w105@8/center at 38.1 s
- ppo_s2r/12190 eval YasMarina_w105@8/center at 49.7 s
- ppo_s2r/12190 eval MoscowRaceway_w105@5/center at 1.4 s
- ppo_s2r/12190 eval Shanghai_rev_w105@5/center at 22.7 s
- ppo_s2r/12190 eval Sochi_rev_w105@5/center at 132.7 s
- ppo_s2r/12190 eval YasMarina_rev_w105@5/center at 40.2 s
- ppo_s2r/12190 eval YasMarina_w105@5/center at 52.1 s
- ppo_s2r/12190 eval Shanghai_rev_w105@8/random at 20.8 s
- ppo_s2r/12190 eval Shanghai_w105@8/random at 58.1 s
- ppo_s2r/12190 eval YasMarina_w105@8/random at 49.6 s
- ppo_s2r/12190 worst Catalunya_rev_w105@8/center at 1.7 s
- ppo_s2r/12190 worst Shanghai_rev_w105@8/center at 77.7 s
- ppo_s2r/12190 worst Shanghai_w105@8/center at 15.6 s
- ppo_s2r/12190 worst YasMarina_w105@8/center at 6.2 s
- ppo_s2r/12190 low Montreal_w105@2/center at 10.9 s
- ppo_s2r/12190 low Shanghai_rev_w105@2.5/center at 40.4 s
- ppo_s2r/12190 low Silverstone_w105@2.5/center at 31.1 s
- ppo_s2r/12190 low Sochi_rev_w105@4/center at 156.8 s
- ppo_s2r/12190 low YasMarina_w105@4/center at 59.7 s
- ppo_s2r/12190 open Catalunya_rev_w902@8/center at 16.3 s
- ppo_s2r/12190 open MexicoCity_rev_w901@8/center at 70.3 s
- ppo_s2r/12190 open MexicoCity_rev_w902@8/center at 120.4 s
- ppo_s2r/12190 open MexicoCity_w901@8/center at 41.8 s
- ppo_s2r/12190 open Montreal_rev_w901@8/center at 16.3 s
- ppo_s2r/12190 open Montreal_rev_w902@8/center at 60.7 s
- ppo_s2r/12190 open MoscowRaceway_w901@8/center at 68.9 s
- ppo_s2r/12190 open Shanghai_w902@8/center at 2.0 s
- ppo_s2r/12190 open Silverstone_rev_w902@8/center at 31.0 s
- ppo_s2r/12190 open Silverstone_w902@8/center at 63.8 s
- ppo_s2r/12190 open Sochi_rev_w902@8/center at 145.5 s
- ppo_s2r/12190 open YasMarina_rev_w901@8/center at 6.3 s
- ppo_s2r/12190 open YasMarina_rev_w902@8/center at 65.7 s
- ppo_s2r/12190 open YasMarina_w901@8/center at 137.3 s
- ppo_s2r/12190 open Catalunya_rev_w902@5/center at 531.0 s
- ppo_s2r/12190 open MexicoCity_rev_w902@5/center at 38.9 s
- ppo_s2r/12190 open MexicoCity_w901@5/center at 43.0 s
- ppo_s2r/12190 open Montreal_rev_w901@5/center at 83.5 s
- ppo_s2r/12190 open Shanghai_rev_w902@5/center at 82.8 s
- ppo_s2r/12190 open Shanghai_w901@5/center at 48.5 s
- ppo_s2r/12190 open Shanghai_w902@5/center at 2.3 s
- ppo_s2r/12190 open Silverstone_rev_w902@5/center at 33.5 s
- ppo_s2r/12190 open Silverstone_w902@5/center at 66.6 s
- ppo_s2r/12190 open YasMarina_w901@5/center at 146.2 s
- ppo_s2r/12190 eval Shanghai_rev_w105@8/center#1 at 104.0 s
- ppo_s2r/12190 eval Shanghai_w105@8/center#1 at 17.4 s
- ppo_s2r/12190 eval YasMarina_rev_w105@8/center#1 at 9.5 s
- ppo_s2r/12190 eval YasMarina_w105@8/center#1 at 43.2 s
- ppo_s2r/12190 eval Catalunya_w105@5/center#1 at 109.2 s
- ppo_s2r/12190 eval Shanghai_rev_w105@5/center#1 at 109.2 s
- ppo_s2r/12190 eval Shanghai_w105@5/center#1 at 18.6 s
- ppo_s2r/12190 eval Sochi_rev_w105@5/center#1 at 27.1 s
- ppo_s2r/12190 eval YasMarina_w105@5/center#1 at 44.9 s
- ppo_s2r/12190 eval Shanghai_rev_w105@8/random#1 at 104.0 s
- ppo_s2r/12190 eval Shanghai_w105@8/random#1 at 17.2 s
- ppo_s2r/12190 eval YasMarina_rev_w105@8/random#1 at 9.4 s
- ppo_s2r/12190 worst Catalunya_rev_w105@8/center#1 at 155.3 s
- ppo_s2r/12190 worst Shanghai_rev_w105@8/center#1 at 98.2 s
- ppo_s2r/12190 worst Shanghai_w105@8/center#1 at 5.8 s
- ppo_s2r/12190 worst YasMarina_w105@8/center#1 at 14.2 s
- ppo_s2r/12190 low Montreal_w105@2.5/center#1 at 86.9 s
- ppo_s2r/12190 low Shanghai_rev_w105@3/center#1 at 163.6 s
- ppo_s2r/12190 low Shanghai_w105@3.5/center#1 at 168.9 s
- ppo_s2r/12190 low YasMarina_w105@2/center#1 at 226.5 s
- ppo_s2r/12190 open Catalunya_rev_w902@8/center#1 at 78.8 s
- ppo_s2r/12190 open MexicoCity_rev_w901@8/center#1 at 9.5 s
- ppo_s2r/12190 open MexicoCity_w901@8/center#1 at 13.3 s
- ppo_s2r/12190 open Montreal_rev_w901@8/center#1 at 15.8 s
- ppo_s2r/12190 open Shanghai_rev_w901@8/center#1 at 632.1 s
- ppo_s2r/12190 open Shanghai_rev_w902@8/center#1 at 140.9 s
- ppo_s2r/12190 open Shanghai_w901@8/center#1 at 73.3 s
- ppo_s2r/12190 open Shanghai_w902@8/center#1 at 33.2 s
- ppo_s2r/12190 open Silverstone_rev_w902@8/center#1 at 59.4 s
- ppo_s2r/12190 open Silverstone_w902@8/center#1 at 65.7 s
- ppo_s2r/12190 open YasMarina_rev_w902@8/center#1 at 32.1 s
- ppo_s2r/12190 open YasMarina_w901@8/center#1 at 507.6 s
- ppo_s2r/12190 open Catalunya_rev_w902@5/center#1 at 531.0 s
- ppo_s2r/12190 open MexicoCity_rev_w902@5/center#1 at 168.4 s
- ppo_s2r/12190 open MexicoCity_w901@5/center#1 at 96.2 s
- ppo_s2r/12190 open Montreal_rev_w901@5/center#1 at 83.7 s
- ppo_s2r/12190 open Shanghai_rev_w902@5/center#1 at 28.2 s
- ppo_s2r/12190 open Shanghai_w901@5/center#1 at 77.2 s
- ppo_s2r/12190 open Shanghai_w902@5/center#1 at 35.4 s
- ppo_s2r/12190 open Silverstone_rev_w902@5/center#1 at 62.0 s
- ppo_s2r/12190 open Silverstone_w902@5/center#1 at 69.6 s
- ppo_s2r/12190 open YasMarina_w901@5/center#1 at 125.3 s
- ppo_s2r/12200 eval Shanghai_rev_w105@8/center at 20.9 s
- ppo_s2r/12200 eval YasMarina_rev_w105@8/center at 37.0 s
- ppo_s2r/12200 eval YasMarina_w105@8/center at 49.0 s
- ppo_s2r/12200 eval MoscowRaceway_w105@5/center at 1.4 s
- ppo_s2r/12200 eval Shanghai_rev_w105@5/center at 23.1 s
- ppo_s2r/12200 eval Shanghai_w105@5/center at 60.7 s
- ppo_s2r/12200 eval Sochi_rev_w105@5/center at 32.0 s
- ppo_s2r/12200 eval YasMarina_rev_w105@5/center at 40.0 s
- ppo_s2r/12200 eval YasMarina_w105@5/center at 52.0 s
- ppo_s2r/12200 eval Shanghai_rev_w105@8/random at 20.8 s
- ppo_s2r/12200 eval YasMarina_rev_w105@8/random at 122.9 s
- ppo_s2r/12200 eval YasMarina_w105@8/random at 507.6 s
- ppo_s2r/12200 worst Catalunya_rev_w105@8/center at 1.8 s
- ppo_s2r/12200 worst Shanghai_rev_w105@8/center at 75.9 s
- ppo_s2r/12200 worst Shanghai_w105@8/center at 15.1 s
- ppo_s2r/12200 worst YasMarina_w105@8/center at 6.9 s
- ppo_s2r/12200 low Montreal_w105@2/center at 9.3 s
- ppo_s2r/12200 low MoscowRaceway_rev_w105@3.5/center at 347.5 s
- ppo_s2r/12200 low Shanghai_rev_w105@2.5/center at 42.8 s
- ppo_s2r/12200 low Silverstone_w105@2.5/center at 30.2 s
- ppo_s2r/12200 low Sochi_rev_w105@4/center at 156.7 s
- ppo_s2r/12200 low YasMarina_rev_w105@3.5/center at 53.4 s
- ppo_s2r/12200 low YasMarina_w105@4/center at 164.9 s
- ppo_s2r/12200 open Catalunya_rev_w902@8/center at 15.8 s
- ppo_s2r/12200 open MexicoCity_rev_w901@8/center at 70.0 s
- ppo_s2r/12200 open MexicoCity_w901@8/center at 41.3 s
- ppo_s2r/12200 open Montreal_rev_w901@8/center at 16.4 s
- ppo_s2r/12200 open Montreal_rev_w902@8/center at 58.8 s
- ppo_s2r/12200 open Shanghai_rev_w901@8/center at 141.2 s
- ppo_s2r/12200 open Shanghai_w902@8/center at 2.1 s
- ppo_s2r/12200 open Silverstone_rev_w902@8/center at 30.8 s
- ppo_s2r/12200 open Silverstone_w902@8/center at 115.0 s
- ppo_s2r/12200 open Sochi_rev_w902@8/center at 46.1 s
- ppo_s2r/12200 open YasMarina_rev_w901@8/center at 7.8 s
- ppo_s2r/12200 open YasMarina_rev_w902@8/center at 67.6 s
- ppo_s2r/12200 open YasMarina_w901@8/center at 44.4 s
- ppo_s2r/12200 open YasMarina_w902@8/center at 36.3 s
- ppo_s2r/12200 open Catalunya_rev_w902@5/center at 531.0 s
- ppo_s2r/12200 open Catalunya_w902@5/center at 123.0 s
- ppo_s2r/12200 open MexicoCity_w901@5/center at 43.0 s
- ppo_s2r/12200 open Montreal_rev_w902@5/center at 129.7 s
- ppo_s2r/12200 open Shanghai_rev_w902@5/center at 82.5 s
- ppo_s2r/12200 open Shanghai_w901@5/center at 49.6 s
- ppo_s2r/12200 open Shanghai_w902@5/center at 2.3 s
- ppo_s2r/12200 open Silverstone_rev_w902@5/center at 32.9 s
- ppo_s2r/12200 open Silverstone_w902@5/center at 66.6 s
- ppo_s2r/12200 open Sochi_rev_w902@5/center at 156.9 s
- ppo_s2r/12200 open YasMarina_w901@5/center at 146.5 s
- ppo_s2r/12200 open Zandvoort_w902@5/center at 144.3 s
- ppo_s2r/12200 eval Shanghai_rev_w105@8/center#1 at 101.8 s
- ppo_s2r/12200 eval Sochi_rev_w105@8/center#1 at 25.6 s
- ppo_s2r/12200 eval YasMarina_rev_w105@8/center#1 at 9.5 s
- ppo_s2r/12200 eval YasMarina_w105@8/center#1 at 41.6 s
- ppo_s2r/12200 eval Shanghai_rev_w105@5/center#1 at 108.5 s
- ppo_s2r/12200 eval Shanghai_w105@5/center#1 at 20.1 s
- ppo_s2r/12200 eval Sochi_rev_w105@5/center#1 at 27.0 s
- ppo_s2r/12200 eval Shanghai_rev_w105@8/random#1 at 101.8 s
- ppo_s2r/12200 eval Sochi_rev_w105@8/random#1 at 25.8 s
- ppo_s2r/12200 eval YasMarina_rev_w105@8/random#1 at 9.5 s
- ppo_s2r/12200 eval YasMarina_w105@8/random#1 at 41.9 s
- ppo_s2r/12200 worst Catalunya_rev_w105@8/center#1 at 152.2 s
- ppo_s2r/12200 worst Shanghai_rev_w105@8/center#1 at 96.2 s
- ppo_s2r/12200 worst Shanghai_w105@8/center#1 at 6.0 s
- ppo_s2r/12200 worst YasMarina_rev_w105@8/center#1 at 37.3 s
- ppo_s2r/12200 worst YasMarina_w105@8/center#1 at 14.0 s
- ppo_s2r/12200 low Montreal_w105@2.5/center#1 at 88.2 s
- ppo_s2r/12200 low Shanghai_rev_w105@3/center#1 at 162.1 s
- ppo_s2r/12200 low Shanghai_w105@3.5/center#1 at 24.7 s
- ppo_s2r/12200 low YasMarina_w105@2/center#1 at 227.3 s
- ppo_s2r/12200 open Catalunya_rev_w902@8/center#1 at 77.1 s
- ppo_s2r/12200 open Catalunya_w902@8/center#1 at 147.8 s
- ppo_s2r/12200 open MexicoCity_rev_w901@8/center#1 at 87.2 s
- ppo_s2r/12200 open MexicoCity_w901@8/center#1 at 91.3 s
- ppo_s2r/12200 open Montreal_rev_w901@8/center#1 at 15.6 s
- ppo_s2r/12200 open Montreal_rev_w902@8/center#1 at 46.3 s
- ppo_s2r/12200 open Montreal_w902@8/center#1 at 64.2 s
- ppo_s2r/12200 open Shanghai_rev_w901@8/center#1 at 205.2 s
- ppo_s2r/12200 open Shanghai_rev_w902@8/center#1 at 26.6 s
- ppo_s2r/12200 open Shanghai_w902@8/center#1 at 140.0 s
- ppo_s2r/12200 open Silverstone_rev_w902@8/center#1 at 58.6 s
- ppo_s2r/12200 open Silverstone_w902@8/center#1 at 63.9 s
- ppo_s2r/12200 open Sochi_rev_w902@8/center#1 at 81.0 s
- ppo_s2r/12200 open YasMarina_rev_w902@8/center#1 at 124.9 s
- ppo_s2r/12200 open YasMarina_w901@8/center#1 at 21.2 s
- ppo_s2r/12200 open Catalunya_rev_w902@5/center#1 at 70.1 s
- ppo_s2r/12200 open Catalunya_w902@5/center#1 at 157.1 s
- ppo_s2r/12200 open MexicoCity_w901@5/center#1 at 12.9 s
- ppo_s2r/12200 open Montreal_rev_w902@5/center#1 at 47.6 s
- ppo_s2r/12200 open Shanghai_rev_w901@5/center#1 at 93.7 s
- ppo_s2r/12200 open Shanghai_rev_w902@5/center#1 at 28.5 s
- ppo_s2r/12200 open Shanghai_w901@5/center#1 at 77.2 s
- ppo_s2r/12200 open Shanghai_w902@5/center#1 at 38.1 s
- ppo_s2r/12200 open Silverstone_rev_w902@5/center#1 at 61.9 s
- ppo_s2r/12200 open Silverstone_w902@5/center#1 at 68.8 s
- ppo_s2r/12200 open Sochi_rev_w902@5/center#1 at 92.2 s
- ppo_s2r/12200 open YasMarina_w901@5/center#1 at 124.2 s
- ppo_s2r/12200 open Zandvoort_w901@5/center#1 at 161.1 s
