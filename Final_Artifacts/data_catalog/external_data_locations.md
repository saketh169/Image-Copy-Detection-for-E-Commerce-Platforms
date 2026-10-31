# External Dataset Locations

Large source and generated image collections are intentionally kept outside
this artifact bundle to avoid duplicating multi-gigabyte benchmark data.

| Dataset | Location | Notes |
|---|---|---|
| Amazon 50k source benchmark | `E:\BTP\benchmark_datasets\amazon` | 35,000 known and 15,000 unknown originals |
| Amazon compact curated output | `E:\BTP\Curated_Dataset` | Complete: 35,000 known folders, 15,000 unknown folders, 200,000 image files; each folder contains one original and three random light edits. Variants are progressive JPEG quality 92. |

The code paths are configurable through `.env` variables documented in
`Codebase_Final\.env.example`.
