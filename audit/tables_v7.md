=== Table 1: Internal 5-fold grouped CV (5 seeds), mean (SD) ===
Model                     R2          RMSE           MAE
RandomForest   0.809 (0.057)31.766 (5.108)22.214 (3.071)
SVR-RBF        0.799 (0.053)32.730 (4.955)23.578 (3.548)
KNN            0.789 (0.084)33.056 (5.721)23.197 (2.536)
SIMPLEX        0.778 (0.069)34.165 (5.195)25.004 (3.350) *
Ridge          0.769 (0.055)35.051 (4.539)26.715 (2.978)
ElasticNet     0.769 (0.055)35.066 (4.573)26.731 (3.024)
HistGB         0.763 (0.070)35.362 (5.385)24.384 (3.062)
MLP            0.699 (0.096)39.796 (6.587)30.578 (4.763)
Mean          -0.005 (0.007)73.703 (5.637)58.597 (2.982)

=== Table 2: Prospective 25-formulation screening metrics ===
SIMPLEX external R2=0.647  Spearman=0.796
SIMPLEX external R2 95% CI [0.445, 0.843]

=== Table 3: Internal -> External generalisation gap ===
ElasticNet     internal 0.769 -> external 0.636  gap +0.133
HistGB         internal 0.763 -> external 0.513  gap +0.250
KNN            internal 0.789 -> external 0.438  gap +0.352
MLP            internal 0.699 -> external 0.542  gap +0.157
Mean           internal -0.005 -> external -1.095  gap +1.090
RandomForest   internal 0.809 -> external 0.564  gap +0.245
Ridge          internal 0.769 -> external 0.637  gap +0.132
SIMPLEX        internal 0.778 -> external 0.647  gap +0.131
SVR-RBF        internal 0.799 -> external 0.633  gap +0.166

=== Table 4: Ablation (ablation_results.csv) ===
                     variant      kind  seed  fold             target       R2      RMSE       MAE    NRMSE  PearsonR  SpearmanRho      CCC  TopK20   TopK30
                  full model component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
                  full model component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
                  full model component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
       w/o multimodal fusion component    42     0 glass_adhesion_kpa 0.767962 36.103891 25.898233 0.481703  0.879720     0.866625 0.872647    0.85 0.866667
       w/o multimodal fusion component    42     1 glass_adhesion_kpa 0.749538 36.482106 29.770264 0.500461  0.868122     0.820184 0.871633    0.85 0.800000
       w/o multimodal fusion component    42     2 glass_adhesion_kpa 0.701511 40.253829 31.783869 0.546342  0.839783     0.855434 0.826571    0.65 0.800000
        w/o sparse attention component    42     0 glass_adhesion_kpa 0.764682 36.358151 24.369545 0.485096  0.887483     0.892055 0.895201    0.85 0.866667
        w/o sparse attention component    42     1 glass_adhesion_kpa 0.743763 36.900302 28.054697 0.506198  0.869893     0.882038 0.876889    0.80 0.766667
        w/o sparse attention component    42     2 glass_adhesion_kpa 0.786725 34.026176 24.283047 0.461817  0.891244     0.898939 0.899574    0.70 0.800000
 w/o attention sparsity reg. component    42     0 glass_adhesion_kpa 0.792629 34.130998 23.602381 0.455381  0.890890     0.892423 0.895426    0.85 0.866667
 w/o attention sparsity reg. component    42     1 glass_adhesion_kpa 0.796590 32.877245 23.233574 0.451010  0.892586     0.898271 0.894000    0.75 0.733333
 w/o attention sparsity reg. component    42     2 glass_adhesion_kpa 0.813665 31.804594 23.059593 0.431665  0.905847     0.913080 0.897483    0.75 0.800000
       w/o FiLM conditioning component    42     0 glass_adhesion_kpa 0.791087 34.257594 22.741304 0.457070  0.890370     0.897156 0.893612    0.85 0.866667
       w/o FiLM conditioning component    42     1 glass_adhesion_kpa 0.782799 33.973495 24.254639 0.466048  0.885081     0.895535 0.884562    0.75 0.733333
       w/o FiLM conditioning component    42     2 glass_adhesion_kpa 0.808067 32.278864 23.098412 0.438102  0.900078     0.904885 0.898226    0.75 0.766667
    w/o task-specific gating component    42     0 glass_adhesion_kpa 0.790182 34.331782 23.178339 0.458059  0.889858     0.892589 0.895923    0.85 0.900000
    w/o task-specific gating component    42     1 glass_adhesion_kpa 0.797171 32.830297 23.589542 0.450366  0.893140     0.896483 0.896954    0.75 0.700000
    w/o task-specific gating component    42     2 glass_adhesion_kpa 0.799395 33.000024 24.224323 0.447890  0.897236     0.909384 0.891923    0.75 0.800000
         w/o residual blocks component    42     0 glass_adhesion_kpa 0.780600 35.106943 24.635576 0.468402  0.884279     0.882088 0.884305    0.85 0.833333
         w/o residual blocks component    42     1 glass_adhesion_kpa 0.760500 35.674859 26.614799 0.489388  0.873599     0.873584 0.866182    0.75 0.766667
         w/o residual blocks component    42     2 glass_adhesion_kpa 0.813183 31.845730 23.199240 0.432223  0.908368     0.919425 0.895359    0.75 0.833333
w/o contrastive pre-training component    42     0 glass_adhesion_kpa 0.783090 34.907163 24.327339 0.465736  0.885087     0.887027 0.888735    0.85 0.833333
w/o contrastive pre-training component    42     1 glass_adhesion_kpa 0.776566 34.457515 25.627914 0.472688  0.881943     0.886739 0.878276    0.75 0.733333
w/o contrastive pre-training component    42     2 glass_adhesion_kpa 0.814957 31.694178 23.636038 0.430167  0.904952     0.912427 0.901660    0.75 0.833333
                   w/o Mixup component    42     0 glass_adhesion_kpa 0.792842 34.113465 21.905146 0.455147  0.894135     0.904128 0.901475    0.85 0.866667
                   w/o Mixup component    42     1 glass_adhesion_kpa 0.790633 33.355190 25.163771 0.457566  0.890148     0.904609 0.895455    0.75 0.766667
                   w/o Mixup component    42     2 glass_adhesion_kpa 0.803152 32.689537 23.694504 0.443676  0.897163     0.902879 0.896880    0.75 0.733333
                     w/o SWA component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
                     w/o SWA component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
                     w/o SWA component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
   w/o uncertainty weighting component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
   w/o uncertainty weighting component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
   w/o uncertainty weighting component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
       w/o domain constraint component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
       w/o domain constraint component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
       w/o domain constraint component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
           w/o modality gate component    42     0 glass_adhesion_kpa 0.777189 35.378804 24.850170 0.472029  0.882999     0.875235 0.888912    0.85 0.833333
           w/o modality gate component    42     1 glass_adhesion_kpa 0.776021 34.499494 25.062889 0.473264  0.882840     0.891933 0.876865    0.75 0.733333
           w/o modality gate component    42     2 glass_adhesion_kpa 0.805079 32.529141 24.135151 0.441499  0.902916     0.912479 0.890465    0.70 0.833333
       w/o transformer block component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
       w/o transformer block component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
       w/o transformer block component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
                     w/o SAM component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
                     w/o SAM component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
                     w/o SAM component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
                     w/o EMA component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
                     w/o EMA component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
                     w/o EMA component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
                  w/o R-Drop component    42     0 glass_adhesion_kpa 0.784896 34.761515 24.216221 0.463793  0.889595     0.885177 0.897683    0.85 0.833333
                  w/o R-Drop component    42     1 glass_adhesion_kpa 0.778582 34.301744 24.729601 0.470551  0.892267     0.900815 0.900025    0.75 0.733333
                  w/o R-Drop component    42     2 glass_adhesion_kpa 0.822656 31.027834 21.805441 0.421123  0.909083     0.908648 0.906540    0.75 0.700000
           w/o feature noise component    42     0 glass_adhesion_kpa 0.800112 33.509476 22.326902 0.447088  0.896138     0.899556 0.901119    0.85 0.866667
           w/o feature noise component    42     1 glass_adhesion_kpa 0.760340 35.686741 26.206240 0.489551  0.872745     0.885545 0.871695    0.75 0.766667
           w/o feature noise component    42     2 glass_adhesion_kpa 0.813418 31.825658 23.137402 0.431951  0.904792     0.911188 0.898436    0.75 0.800000
        w/o MFM pre-training component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
        w/o MFM pre-training component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
        w/o MFM pre-training component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
              w/o MC-Dropout component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
              w/o MC-Dropout component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
              w/o MC-Dropout component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
     w/o pretrained transfer component    42     0 glass_adhesion_kpa 0.792630 34.130875 23.602434 0.455379  0.890890     0.892423 0.895426    0.85 0.866667
     w/o pretrained transfer component    42     1 glass_adhesion_kpa 0.796591 32.877189 23.233513 0.451009  0.892586     0.898271 0.894001    0.75 0.733333
     w/o pretrained transfer component    42     2 glass_adhesion_kpa 0.813666 31.804522 23.059256 0.431664  0.905847     0.913080 0.897484    0.75 0.800000
             fusion = concat    fusion    42     0 glass_adhesion_kpa 0.776907 35.401163 25.223364 0.472327  0.881974     0.887872 0.886265    0.85 0.866667
             fusion = concat    fusion    42     1 glass_adhesion_kpa 0.773520 34.691601 26.190975 0.475899  0.882371     0.882240 0.875205    0.75 0.766667
             fusion = concat    fusion    42     2 glass_adhesion_kpa 0.789143 33.832740 23.747158 0.459192  0.889666     0.893118 0.890952    0.70 0.766667
               fusion = film    fusion    42     0 glass_adhesion_kpa 0.798630 33.633447 22.099581 0.448742  0.894811     0.900271 0.899589    0.85 0.866667
               fusion = film    fusion    42     1 glass_adhesion_kpa 0.769824 34.973537 26.694027 0.479767  0.877959     0.856051 0.875482    0.75 0.800000
               fusion = film    fusion    42     2 glass_adhesion_kpa 0.798393 33.082326 23.884310 0.449007  0.896402     0.906554 0.888465    0.70 0.733333
              fusion = cross    fusion    42     0 glass_adhesion_kpa 0.781735 35.015996 24.375187 0.467188  0.884631     0.875121 0.888741    0.85 0.833333
              fusion = cross    fusion    42     1 glass_adhesion_kpa 0.770067 34.955021 25.668783 0.479513  0.879078     0.868611 0.872732    0.80 0.800000
              fusion = cross    fusion    42     2 glass_adhesion_kpa 0.812602 31.895168 22.966148 0.432894  0.904728     0.907891 0.898804    0.75 0.800000
