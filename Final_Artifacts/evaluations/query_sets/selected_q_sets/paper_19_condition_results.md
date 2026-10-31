# Paper-authoritative 19-condition results

**Status:** canonical summary for `Reports/paper1_ieee_format.tex`.

Each value below is calculated over exactly 19 conditions: `original` plus
the following 18 distinct attacks: `crop20`, `crop5`,
`mix_crop20_watermark`, `mix_rotate10_bright`, `rotate30`, `rotate5`,
`rotate_m5`, `bright_light`, `resize`, `resize_compress`, `rotate_m10`,
`blur`, `contrast`, `watermark_text`, `watermark_asset`, `rotate15`,
`crop10`, and `flip_h`.

| Dataset | Q | Threshold | Balanced accuracy | Known recall | Unknown rejection | Bands (90/80/70) |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | 100 | 2.91e-4 | 87.47 | 88.53 | 86.42 | 8/7/4 |
| CIFAR-10 | 200 | 2.91e-4 | 88.58 | 88.95 | 88.21 | 8/5/6 |
| CIFAR-10 | 500 | 2.91e-4 | 88.71 | 89.17 | 88.25 | 8/5/6 |
| CIFAR-10 | 1000 | 2.91e-4 | 89.19 | 89.96 | 88.42 | 8/6/5 |
| Flickr | 100 | 2.04e-6 | 87.74 | 89.05 | 86.42 | 8/6/5 |
| Flickr | 200 | 2.04e-6 | 88.91 | 89.61 | 88.21 | 8/6/5 |
| Flickr | 500 | 2.04e-6 | 89.30 | 90.40 | 88.20 | 8/6/5 |
| Flickr | 1000 | 2.04e-6 | 89.18 | 91.47 | 86.89 | 9/5/5 |
| UCID | 100 | 3.56e-5 | 87.39 | 87.89 | 86.89 | 8/5/6 |
| UCID | 200 | 3.56e-5 | 88.84 | 89.55 | 88.13 | 9/4/6 |
| UCID | 400 | 3.56e-5 | 88.69 | 89.84 | 87.54 | 9/4/6 |
| UCID | 500 | 3.56e-5 | 88.59 | 89.88 | 87.31 | 9/4/6 |
| Amazon | 100 | 3.78e-8 | 89.34 | 89.74 | 88.95 | 9/7/3 |
| Amazon | 200 | 3.78e-8 | 89.70 | 90.05 | 89.34 | 9/7/3 |
| Amazon | 500 | 3.78e-8 | 88.41 | 88.19 | 88.62 | 6/11/2 |
| Amazon | 1000 | 3.78e-8 | 87.22 | 85.48 | 88.95 | 6/10/3 |

At the largest reported workload, the balanced accuracies are 89.19% (CIFAR-10), 89.18% (Flickr), 88.59% (UCID), and 87.22% (Amazon), for an average of 88.55%.
