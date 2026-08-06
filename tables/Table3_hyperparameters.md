**Table 3.** Search space and finally selected hyper-parameter values.

| Hyper-parameter           | Search range   | Selected value   |
|:--------------------------|:---------------|:-----------------|
| d_model                   | -              | 64               |
| n_blocks                  | -              | 2                |
| n_heads                   | -              | 4                |
| dropout                   | -              | 0.15             |
| use_transformer           | -              | False            |
| use_attention             | -              | True             |
| use_film                  | -              | False            |
| use_modality_gate         | -              | False            |
| use_contrastive           | -              | False            |
| use_pretrain_recon        | -              | False            |
| use_mfm                   | -              | False            |
| use_sam                   | -              | False            |
| use_ema                   | -              | False            |
| use_mixup                 | -              | True             |
| mixup_alpha               | -              | 0.4              |
| use_swa                   | -              | True             |
| use_uncertainty_weighting | -              | False            |
| use_domain_constraint     | -              | True             |
| constraint_w              | -              | 0.1              |
| y_transform               | -              | standard         |
| max_epochs                | -              | 200              |
| patience                  | -              | 30               |
| batch_size                | -              | 32               |
| lr                        | -              | 0.003            |
| weight_decay              | -              | 0.001            |
| scaler                    | -              | standard         |
