# Data sources for SIMPLEX

Every link below was verified with a live HTTP request on 2026-08-06 08:47:10. No registration, login, API key or data-access application is required for any of them.

| # | Source | Role | HTTP | Size (bytes) | License | Link |
|---|--------|------|------|--------------|---------|------|
| 1 | hydrogel_df180 | internal | 206 OK | 19,766 | MIT | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_180.csv> |
| 2 | hydrogel_df341 | external | 200 OK | 38,627 | MIT | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_341.csv> |
| 3 | hydrogel_df316 | annotation | 200 OK | 35,982 | MIT | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_316.csv> |

## Details

### hydrogel_df180
- **Role**: internal
- **Direct download**: <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_180.csv>
- **Landing page**: <https://github.com/sheng-hu/hydrogels>
- **Local absolute path**: `C:/Users/TS/WorkBuddy/HydroGelNet\data\raw\df_180.csv`
- **License**: MIT
- **Citation**: Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.
- **SHA-256**: `15f5ec9f3380e1d64a3de55cd20a5d4a42ded0b1c28af65e50718f7765683673`
- **Note**: Round-1 baseline: 180 formulations, 6 monomer molar fractions -> Glass (kPa)_max adhesion. Train region (low-performance).

### hydrogel_df341
- **Role**: external
- **Direct download**: <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_341.csv>
- **Landing page**: <https://github.com/sheng-hu/hydrogels>
- **Local absolute path**: `C:/Users/TS/WorkBuddy/HydroGelNet\data\raw\df_341.csv`
- **License**: MIT
- **Citation**: Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.
- **SHA-256**: `afbc0a6b3aed2a44a1fbf609b033eeb8cfb09c5626fcb8ff08f39cc60ae8ec5b`
- **Note**: Full dataset 341 formulas. External set = rows not in df_180 (161 SMBO-guided high-performance formulas).

### hydrogel_df316
- **Role**: annotation
- **Direct download**: <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_316.csv>
- **Landing page**: <https://github.com/sheng-hu/hydrogels>
- **Local absolute path**: `C:/Users/TS/WorkBuddy/HydroGelNet\data\raw\df_316.csv`
- **License**: MIT
- **Citation**: Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.
- **SHA-256**: `b87d8f5f73acb3769fd23b36e7fef5397c262f537e89ea9c9b13b91adcda60c6`
- **Note**: Intermediate round-3 dataset (316 formulas); used for data-size ablation.
