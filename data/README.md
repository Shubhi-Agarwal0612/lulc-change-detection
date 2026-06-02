# Data

This directory should contain the satellite imagery and ground truth used for training.

**The data files are not included in this repository** due to their large size and licensing restrictions (ISRO/PlanetScope imagery).

## Required Files

| File | Description | Size (approx.) |
|------|-------------|-----------------|
| `Aligned_2016_FINAL.tif` | PlanetScope 3m RGB image — pre-change (2016) | ~500 MB |
| `Aligned_2025_MATCHED.tif` | PlanetScope 3m RGB image — post-change (2025) | ~500 MB |
| `change_mask.tif` | Binary ground truth mask (0 = no change, 1 = change) | ~50 MB |
| `training_data.csv` | Spectral signature samples with class labels | ~1 MB |

## Data Source

The satellite imagery was obtained from **ISRO's PlanetScope** archive at **3-meter spatial resolution**. Images are co-registered and aligned in **UTM Zone 43N (EPSG:32643)** projection.

Study areas cover **Chandigarh** and **Varanasi**, India — two cities with contrasting urbanization patterns.

## How to Obtain

If you wish to reproduce this work:

1. Request PlanetScope imagery through [ISRO's Bhuvan portal](https://bhuvan.nrsc.gov.in/) or [Planet Explorer](https://www.planet.com/explorer/).
2. Ensure temporal alignment between T1 and T2 images.
3. Generate a binary change mask using supervised annotation or reference data.
4. Place all files in this `data/` directory.
