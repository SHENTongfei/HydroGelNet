**Table 7.** Ablation study. A positive contribution means removing the component degrades performance.

| Target             | Variant                      |   Variant R2 |   Full R2 |   Contribution |   Contribution (%) |   p (Holm) | Sig.   | Verdict         |
|:-------------------|:-----------------------------|-------------:|----------:|---------------:|-------------------:|-----------:|:-------|:----------------|
| glass_adhesion_kpa | fusion = cross               |       0.7087 |    0.7131 |         0.0044 |             0.6213 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | fusion = film                |       0.6826 |    0.7131 |         0.0305 |             4.2767 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | fusion = gated               |       0.6513 |    0.7131 |         0.0619 |             8.6786 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o EMA                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o FiLM conditioning        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o MC-Dropout               |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o MFM pre-training         |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o Mixup                    |       0.646  |    0.7131 |         0.0671 |             9.4157 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o R-Drop                   |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o SAM                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o SWA                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o attention sparsity reg.  |       0.7133 |    0.7131 |        -0.0002 |            -0.0268 |          1 | ns     | neutral/harmful |
| glass_adhesion_kpa | w/o contrastive pre-training |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o domain constraint        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o feature noise            |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o modality gate            |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o multimodal fusion        |       0.6198 |    0.7131 |         0.0934 |            13.0913 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o pretrained transfer      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o residual blocks          |       0.6515 |    0.7131 |         0.0617 |             8.6511 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o sparse attention         |       0.7207 |    0.7131 |        -0.0075 |            -1.0566 |          1 | ns     | neutral/harmful |
| glass_adhesion_kpa | w/o task-specific gating     |       0.6916 |    0.7131 |         0.0215 |             3.0171 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o transformer block        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o uncertainty weighting    |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
