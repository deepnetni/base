# MCSE Experiments

# 1.Train

MCSE model:
- `train_mcse_chime.py` training mcse model based on CHiMe3 dataset;
- `train_mcse_spdns.py` training mcse model based on SPDNS dataset;
- `train_mcse_l3das.py` training mcse model based on L3DAS dataset;

DeFT-AN model:
- `train_deftan_chime.py` training mcse model based on CHiMe3 dataset;
- `train_deftan_spdns.py` training mcse model based on SPDNS dataset;
- `train_deftan_l3das.py` training mcse model based on L3DAS dataset;

MCNet model:
- `train_mcnet_chime.py` training mcse model based on CHiMe3 dataset;
- `train_mcnet_spdns.py` training mcse model based on SPDNS dataset;
- `train_mcnet_l3das.py` training mcse model based on L3DAS dataset;

# 2.Inference

## The CHiMe3 dataset

*The mcse model* 

```python 
python train_mcse_chime.py --pred --vtest --ckpt trained_mcse_chime3/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcse_vtest_50 
```

```pyth
python train_mcse_chime.py --pred --valid --ckpt trained_mcse_chime3/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcse_valid_50 
```

*The deftan model*

```python
python train_deftan_chime.py --pred --vtest --ckpt trained_mcse_chime3/deftan/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_deftan_vtest_50
```

```py
python train_deftan_chime.py --pred --valid --ckpt trained_mcse_chime3/deftan/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_deftan_valid_50
```

*The mcnet model*

```python
python train_mcnet_chime.py --pred --vtest --ckpt trained_mcse_chime3/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcnet_vtest_50
```

```py
python train_mcnet_chime.py --pred --valid --ckpt trained_mcse_chime3/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcnet_valid_50
```

## The Spatial DNS dataset

*The mcse model*

```python
python train_mcse_spdns.py --pred --vtest --ckpt trained_mcse_spdns/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcse_vtest_50
```

```py
python train_mcse_spdns.py --pred --valid --ckpt trained_mcse_spdns/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcse_valid_50
```

*The deftan model*

```python
python train_deftan_spdns.py --pred --vtest --ckpt trained_mcse_spdns/deftan/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_deftan_vtest_50
```
```python
python train_deftan_spdns.py --pred --valid --ckpt trained_mcse_spdns/deftan/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_deftan_valid_50
```

*The mcnet model*

```python
python train_mcnet_spdns.py --pred --vtest --ckpt trained_mcse_spdns/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcnet_vtest_50
```

```py
python train_mcnet_spdns.py --pred --valid --ckpt trained_mcse_spdns/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcnet_valid_50
```

## The L3DAS dataset

==*The L3DAS dataset only provides a validate dataset, without blind test set.*== 

The mcse model
```python
python train_mcse_l3das.py --pred --vtest --ckpt trained_mcse_l3das/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_mcse_vtest_50
```

The deftan model
```python
python train_deftan_l3das.py --pred --vtest --ckpt trained_mcse_l3das/deftan/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_deftan_vtest_50
```

The mcnet model
```python
python train_mcnet_l3das.py --pred --vtest --ckpt trained_mcse_l3das/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_mcnet_vtest_50
```

# 3.Metrics

## The mcse model chime3 vtest

```python 
python train_mcse_chime.py --pred --vtest --ckpt trained_mcse_chime3/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcse_vtest_50 
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_mcse_vtest_50  --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi
```

> **total-50**
> {'pesqw': [2.8653, 0.4422], 'pesqn': [3.3426, 0.3254], 'stoi': [0.9747, 0.0194], '-pesq': [2.8653, 0.4422], 'csig': [3.939, 0.4565], 'cbak': [2.8338, 0.2628], 'covl': [3.4079, 0.4509], '-snr': [0.7174, 2.9355], '-seg-snr': [-0.3202, 1.7392]}
> **et05_str_simu**
> {'pesqw': [2.7092, 0.4372], 'pesqn': [3.2681, 0.3293], 'stoi': [0.9729, 0.0183], '-pesq': [2.7092, 0.4372], 'csig': [3.7922, 0.4603], 'cbak': [2.7608, 0.248], 'covl': [3.2534, 0.4514], '-snr': [1.1787, 2.9213], '-seg-snr': [-0.1791, 1.6636]}
> **et05_caf_simu**
> {'pesqw': [2.9165, 0.3458], 'pesqn': [3.3371, 0.2283], 'stoi': [0.9774, 0.0103], '-pesq': [2.9165, 0.3458], 'csig': [3.9726, 0.3695], 'cbak': [2.8694, 0.1951], 'covl': [3.4538, 0.3565], '-snr': [0.6002, 2.98], '-seg-snr': [-0.2885, 1.7677]}
> **et05_bus_simu**
> {'pesqw': [3.0323, 0.3124], 'pesqn': [3.5035, 0.253], 'stoi': [0.9795, 0.0098], '-pesq': [3.0323, 0.3124], 'csig': [4.1295, 0.3138], 'cbak': [2.9083, 0.2147], 'covl': [3.5864, 0.3125], '-snr': [0.7284, 2.8944], '-seg-snr': [-0.3745, 1.8074]}
> **et05_ped_simu**
> {'pesqw': [2.8031, 0.5613], 'pesqn': [3.2619, 0.4012], 'stoi': [0.9689, 0.0301], '-pesq': [2.8031, 0.5613], 'csig': [3.8615, 0.5674], 'cbak': [2.7968, 0.342], 'covl': [3.3381, 0.5677], '-snr': [0.3622, 2.8849], '-seg-snr': [-0.4388, 1.7038]}
>
> **total-100**
>
> {'pesqw': [3.0613, 0.4221], 'pesqn': [3.4502, 0.313], 'stoi': [0.9775, 0.0169], '-pesq': [3.0613, 0.4221], 'csig': [4.1751, 0.4124], 'cbak': [2.959, 0.2609], 'covl': [3.61259, 0.4194], '-snr': [1.3286, 3.1255], '-seg-snr': [0.1141, 1.7792]}
> **et05_str_simu**
> {'pesqw': [3.0008, 0.4386], 'pesqn': [3.4278, 0.3366], 'stoi': [0.9761, 0.0165], '-pesq': [3.0008, 0.4386], 'csig': [4.1245, 0.4262], 'cbak': [2.9368, 0.2541], 'covl': [3.5674, 0.4371], '-snr': [1.8593, 3.0597], '-seg-snr': [0.3414, 1.6622]}
> **et05_caf_simu**
> {'pesqw': [3.1026, 0.3212], 'pesqn': [3.4398, 0.2265], 'stoi': [0.9797, 0.0095], '-pesq': [3.1026, 0.3212], 'csig': [4.1997, 0.3311], 'cbak': [2.9929, 0.1878], 'covl': [3.6627, 0.3252], '-snr': [1.2113, 3.1374], '-seg-snr': [0.1807, 1.757]}
> **et05_bus_simu**
> {'pesqw': [3.1594, 0.3064], 'pesqn': [3.5684, 0.2426], 'stoi': [0.9817, 0.0087], '-pesq': [3.1594, 0.3064], 'csig': [4.2839, 0.2952], 'cbak': [2.9915, 0.2191], 'covl': [3.7282, 0.3001], '-snr': [1.3665, 3.1389], '-seg-snr': [-0.0551, 1.8861]}
> **et05_ped_simu**
> {'pesqw': [2.9825, 0.5495], 'pesqn': [3.3648, 0.3828], 'stoi': [0.9726, 0.0257], '-pesq': [2.9825, 0.5495], 'csig': [4.0926, 0.5293], 'cbak': [2.9148, 0.346], 'covl': [3.5455, 0.543], '-snr': [0.8774, 3.085], '-seg-snr': [-0.0108, 1.7761]}

## The deftan model chime3 vtest

```python
python train_deftan_chime.py --pred --vtest --ckpt trained_mcse_chime3/deftan/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_deftan_vtest_50
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_deftan_vtest_50 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi
```

> **total-50**
> {'pesqw': [2.7865, 0.4957], 'pesqn': [3.3268, 0.3979], 'stoi': [0.9689, 0.0336], '_pesq': [2.7865, 0.4957], 'csig': [3.8803, 0.5236], 'cbak': [2.7451, 0.2801], 'covl': [3.335, 0.5125], '_snr': [-0.626, 2.5196], '_seg_snr': [-0.9605, 1.5707]}
> **et05_str_simu**
> {'pesqw': [2.5216, 0.4048], 'pesqn': [3.1873, 0.3365], 'stoi': [0.9679, 0.0254], '_pesq': [2.5216, 0.4048], 'csig': [3.6106, 0.4578], 'cbak': [2.6076, 0.2287], 'covl': [3.0652, 0.4324], '_snr': [-0.3537, 2.6567], '_seg_snr': [-1.0491, 1.5991]}
> **et05_caf_simu**
> {'pesqw': [2.8602, 0.3705], 'pesqn': [3.3598, 0.2377], 'stoi': [0.9738, 0.0115], '_pesq': [2.8602, 0.3704], 'csig': [3.9569, 0.4208], 'cbak': [2.7941, 0.1929], 'covl': [3.4137, 0.3948], '_snr': [-0.7902, 2.5667], '_seg_snr': [-0.8898, 1.6079]}
> **et05_bus_simu**
> {'pesqw': [3.0222, 0.3711], 'pesqn': [3.5346, 0.2657], 'stoi': [0.9786, 0.009], '_pesq': [3.0222, 0.3711], 'csig': [4.1144, 0.4007], 'cbak': [2.8539, 0.2348], 'covl': [3.5697, 0.3876], '_snr': [-0.6266, 2.345], '_seg_snr': [-0.9834, 1.5649]}
> **et05_ped_simu**
> {'pesqw': [2.7418, 0.6411], 'pesqn': [3.2253, 0.5646], 'stoi': [0.9551, 0.0579], '_pesq': [2.7418, 0.6411], 'csig': [3.8391, 0.6436], 'cbak': [2.7248, 0.3682], 'covl': [3.2915, 0.6493], '_snr': [-0.7334, 2.4772], '_seg_snr': [-0.9196, 1.5042]}
>
> **total-100**
> {'pesqw': [2.8734, 0.4655], 'pesqn': [3.3868, 0.3732], 'stoi': [0.9716, 0.0292], '_pesq': [2.8734, 0.4655], 'csig': [3.9515, 0.478], 'cbak': [2.8201, 0.2679], 'covl': [3.4165, 0.4748], '_snr': [0.1334, 2.7274], '_seg_snr': [-0.5229, 1.6011]}
> **et05_str_simu**
> {'pesqw': [2.7004, 0.4027], 'pesqn': [3.2911, 0.323], 'stoi': [0.9709, 0.0226], '_pesq': [2.7004, 0.4027], 'csig': [3.7901, 0.4311], 'cbak': [2.7341, 0.2264], 'covl': [3.247, 0.4205], '_snr': [0.4762, 2.7916], '_seg_snr': [-0.4947, 1.5916]}
> **et05_caf_simu**
> {'pesqw': [2.9215, 0.3686], 'pesqn': [3.404, 0.2401], 'stoi': [0.9758, 0.0106], '_pesq': [2.9215, 0.3687], 'csig': [3.9971, 0.405], 'cbak': [2.8581, 0.1982], 'covl': [3.4668, 0.3863], '_snr': [0.0132, 2.7864], '_seg_snr': [-0.4317, 1.6419]}
> **et05_bus_simu**
> {'pesqw': [3.0369, 0.354], 'pesqn': [3.5562, 0.2605], 'stoi': [0.9795, 0.0088], '_pesq': [3.0369, 0.354], 'csig': [4.0998, 0.3861], 'cbak': [2.8863, 0.2282], 'covl': [3.5712, 0.371], '_snr': [0.1322, 2.6214], '_seg_snr': [-0.6396, 1.637]}
> **et05_ped_simu**
> {'pesqw': [2.8347, 0.6187], 'pesqn': [3.2957, 0.5299], 'stoi': [0.9602, 0.05], '_pesq': [2.8347, 0.6187], 'csig': [3.9188, 0.6031], 'cbak': [2.8018, 0.3618], 'covl': [3.3808, 0.6176], '_snr': [-0.0879, 2.6728], '_seg_snr': [-0.5257, 1.5242]}

## The mcnet model chime3 vtest

```python
python train_mcnet_chime.py --pred --vtest --ckpt trained_mcse_chime3/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcnet_vtest_50
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_mcnet_vtest_50 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/test --metrics asr pesqn pesqw stoi
```

> **total-50**
> {'pesqw': [2.5915, 0.4479], 'pesqn': [3.1341, 0.3883], 'stoi': [0.9579, 0.0334], '_pesq': [2.5915, 0.4479], 'csig': [3.7077, 0.4877], 'cbak': [2.4552, 0.2506], 'covl': [3.1431, 0.4711], '_snr': [-3.36, 1.4241], '_seg_snr': [-3.7477, 1.0631]}
> **et05_str_simu**
> {'pesqw': [2.4828, 0.4647], 'pesqn': [3.0379, 0.4091], 'stoi': [0.9535, 0.0345], '_pesq': [2.4828, 0.4647], 'csig': [3.5922, 0.5172], 'cbak': [2.3825, 0.274], 'covl': [3.0274, 0.4963], '_snr': [-3.4633, 1.3661], '_seg_snr': [-3.9254, 1.0858]}
> **et05_caf_simu**
> {'pesqw': [2.6632, 0.3676], 'pesqn': [3.1729, 0.277], 'stoi': [0.9644, 0.0163], '_pesq': [2.6632, 0.3676], 'csig': [3.7861, 0.3928], 'cbak': [2.5202, 0.2097], 'covl': [3.2227, 0.3787], '_snr': [-3.203, 1.4893], '_seg_snr': [-3.4531, 1.0028]}
> **et05_bus_simu**
> {'pesqw': [2.6612, 0.3693], 'pesqn': [3.2549, 0.3063], 'stoi': [0.9649, 0.0181], '_pesq': [2.6612, 0.3693], 'csig': [3.7877, 0.4027], 'cbak': [2.462, 0.2154], 'covl': [3.2156, 0.3879], '_snr': [-3.4456, 1.4318], '_seg_snr': [-4.0574, 1.0848]}
> **et05_ped_simu**
> {'pesqw': [2.5587, 0.5406], 'pesqn': [3.0707, 0.4857], 'stoi': [0.9488, 0.0497], '_pesq': [2.5587, 0.5406], 'csig': [3.6646, 0.5829], 'cbak': [2.4561, 0.2761], 'covl': [3.1067, 0.5666], '_snr': [-3.3282, 1.3909], '_seg_snr': [-3.5551, 0.9528]}
>
> 
>
> **total-100**
> {'pesqw': [2.6319, 0.4445], 'pesqn': [3.1481, 0.3742], 'stoi': [0.9638, 0.0318], '_pesq': [2.6319, 0.4445], 'csig': [3.6984, 0.5093], 'cbak': [2.455, 0.2437], 'covl': [3.163, 0.4788], '_snr': [-3.8136, 1.3829], '_seg_snr': [-4.238, 1.0447]}
> **et05_str_simu**
> {'pesqw': [2.5185, 0.4527], 'pesqn': [3.0499, 0.3823], 'stoi': [0.9604, 0.0307], '_pesq': [2.5185, 0.4527], 'csig': [3.6049, 0.4999], 'cbak': [2.3853, 0.2611], 'covl': [3.0568, 0.482], '_snr': [-3.9526, 1.2338], '_seg_snr': [-4.3774, 0.9956]}
> **et05_caf_simu**
> {'pesqw': [2.6899, 0.3578], 'pesqn': [3.1834, 0.2485], 'stoi': [0.9706, 0.013], '_pesq': [2.6899, 0.3578], 'csig': [3.708, 0.4123], 'cbak': [2.5055, 0.2066], 'covl': [3.2008, 0.3832], '_snr': [-3.6915, 1.4352], '_seg_snr': [-4.041, 1.0034]}
> **et05_bus_simu**
> {'pesqw': [2.7383, 0.3308], 'pesqn': [3.2764, 0.2904], 'stoi': [0.9697, 0.0163], '_pesq': [2.7383, 0.3308], 'csig': [3.8831, 0.3548], 'cbak': [2.4826, 0.1925], 'covl': [3.3064, 0.3437], '_snr': [-3.8807, 1.4547], '_seg_snr': [-4.5097, 1.088]}
> **et05_ped_simu**
> {'pesqw': [2.581, 0.5636], 'pesqn': [3.0827, 0.4861], 'stoi': [0.9546, 0.0499], '_pesq': [2.581, 0.5636], 'csig': [3.5977, 0.6623], 'cbak': [2.4465, 0.2851], 'covl': [3.088, 0.617], '_snr': [-3.7295, 1.3806], '_seg_snr': [-4.0241, 1.0031]}


## The mcse model chime3 valid

```python
python train_mcse_chime.py --pred --valid --ckpt trained_mcse_chime3/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcse_valid_50 
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_mcse_valid_50  --src ~/datasets/CHiME3/data/audio/16kHz/isolated/dev --metrics asr pesqn pesqw stoi
```



## The MVDR model chime3 vtest

> ![image-20240818202553266](C:\Users\niye\AppData\Roaming\Typora\typora-user-images\image-20240818202553266.png)

## The deftan model chime3 valid

```python
python train_deftan_chime.py --pred --valid --ckpt trained_mcse_chime3/deftan/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_deftan_valid_50
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_deftan_valid_50 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/dev --metrics asr pesqn pesqw stoi
```

> **total-50**
> {'pesqw': [2.6229, 0.4028], 'pesqn': [3.2712, 0.3168], 'stoi': [0.9724, 0.0169], '_pesq': [2.6229, 0.4028], 'csig': [3.7854, 0.3877], 'cbak': [2.6913, 0.2321], 'covl': [3.21, 0.3963], '_snr': [0.9689, 2.3338], '_seg_snr': [-0.7857, 1.6069]}
> **dt05_bus_simu**
> {'pesqw': [2.7183, 0.4654], 'pesqn': [3.3335, 0.3762], 'stoi': [0.9759, 0.0156], '_pesq': [2.7183, 0.4654], 'csig': [3.8839, 0.4275], 'cbak': [2.7144, 0.2484], 'covl': [3.3039, 0.4485], '_snr': [1.0457, 2.4324], '_seg_snr': [-0.9987, 1.5141]}
> **dt05_ped_simu**
> {'pesqw': [2.7726, 0.3705], 'pesqn': [3.3821, 0.2625], 'stoi': [0.9769, 0.0117], '_pesq': [2.7726, 0.3705], 'csig': [3.9203, 0.3826], 'cbak': [2.7759, 0.2292], 'covl': [3.3576, 0.3756], '_snr': [0.6975, 2.3976], '_seg_snr': [-0.793, 1.7539]}
> **dt05_caf_simu**
> {'pesqw': [2.5875, 0.3211], 'pesqn': [3.261, 0.2555], 'stoi': [0.9698, 0.0146], '_pesq': [2.5875, 0.3211], 'csig': [3.7527, 0.3113], 'cbak': [2.703, 0.1712], 'covl': [3.1774, 0.3152], '_snr': [1.1322, 2.1495], '_seg_snr': [-0.3947, 1.4581]}
> **dt05_str_simu**
> {'pesqw': [2.4131, 0.3392], 'pesqn': [3.1083, 0.2881], 'stoi': [0.9668, 0.022], '_pesq': [2.4131, 0.3392], 'csig': [3.5847, 0.3255], 'cbak': [2.5717, 0.2232], 'covl': [3.0012, 0.3336], '_snr': [1.0003, 2.3226], '_seg_snr': [-0.9563, 1.6146]}
>
> **total-100**
> {'pesqw': [2.7256, 0.4013], 'pesqn': [3.328, 0.3063], 'stoi': [0.9757, 0.0152], '_pesq': [2.7256, 0.4013], 'csig': [3.8482, 0.3898], 'cbak': [2.7648, 0.2307], 'covl': [3.2956, 0.3971], '_snr': [1.4698, 2.373], '_seg_snr': [-0.5068, 1.5957]}
> **dt05_bus_simu**
> {'pesqw': [2.7587, 0.4632], 'pesqn': [3.3589, 0.3595], 'stoi': [0.9788, 0.0141], '_pesq': [2.7587, 0.4632], 'csig': [3.8781, 0.4398], 'cbak': [2.7559, 0.2477], 'covl': [3.3234, 0.4529], '_snr': [1.5481, 2.4133], '_seg_snr': [-0.733, 1.5217]}
> **dt05_ped_simu**
> {'pesqw': [2.9258, 0.3467], 'pesqn': [3.4569, 0.2511], 'stoi': [0.98, 0.0107], '_pesq': [2.9259, 0.3467], 'csig': [4.0309, 0.3544], 'cbak': [2.8751, 0.2091], 'covl': [3.4927, 0.3512], '_snr': [1.2302, 2.4788], '_seg_snr': [-0.4964, 1.7179]}
> **dt05_caf_simu**
> {'pesqw': [2.6928, 0.332], 'pesqn': [3.3237, 0.2539], 'stoi': [0.9731, 0.0134], '_pesq': [2.6928, 0.332], 'csig': [3.8198, 0.3276], 'cbak': [2.7814, 0.1727], 'covl': [3.2668, 0.328], '_snr': [1.7125, 2.2471], '_seg_snr': [-0.0804, 1.4303]}
> **dt05_str_simu**
> {'pesqw': [2.5251, 0.3417], 'pesqn': [3.1723, 0.277], 'stoi': [0.9708, 0.0193], '_pesq': [2.5251, 0.3417], 'csig': [3.664, 0.3354], 'cbak': [2.6467, 0.2269], 'covl': [3.0995, 0.3405], '_snr': [1.3883, 2.3189], '_seg_snr': [-0.7176, 1.611]}

## The mcnet model chime3 valid

```python
python train_mcnet_chime.py --pred --valid --ckpt trained_mcse_chime3/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_chime3/pred_mcnet_valid_50
```

```python
python compute_metrics.py --sub --out ../trained_mcse_chime3/pred_mcnet_valid_50 --src ~/datasets/CHiME3/data/audio/16kHz/isolated/dev --metrics asr pesqn pesqw stoi
```

> **total-50**
> {'pesqw': [2.7288, 0.3894], 'pesqn': [3.2372, 0.333], 'stoi': [0.969, 0.0202], '_pesq': [2.7288, 0.3894], 'csig': [3.872, 0.3663], 'cbak': [2.468, 0.233], 'covl': [3.3044, 0.3821], '_snr': [-4.6631, 1.2454], '_seg_snr': [-5.0366, 1.1555]}
> **dt05_bus_simu**
> {'pesqw': [2.773, 0.467], 'pesqn': [3.3022, 0.3994], 'stoi': [0.9718, 0.0203], '_pesq': [2.773, 0.467], 'csig': [3.9513, 0.4353], 'cbak': [2.4773, 0.265], 'covl': [3.3659, 0.4575], '_snr': [-4.6203, 1.1884], '_seg_snr': [-5.2126, 1.1072]}
> **dt05_ped_simu**
> {'pesqw': [2.9256, 0.2639], 'pesqn': [3.3795, 0.2354], 'stoi': [0.9756, 0.0127], '_pesq': [2.9256, 0.2639], 'csig': [4.0404, 0.2537], 'cbak': [2.5904, 0.1688], 'covl': [3.4919, 0.2591], '_snr': [-4.5637, 1.2025], '_seg_snr': [-4.7784, 1.146]}
> **dt05_caf_simu**
> {'pesqw': [2.7319, 0.3103], 'pesqn': [3.206, 0.2748], 'stoi': [0.9671, 0.0156], '_pesq': [2.7319, 0.3103], 'csig': [3.8513, 0.2958], 'cbak': [2.4795, 0.1938], 'covl': [3.2959, 0.303], '_snr': [-4.6935, 1.2361], '_seg_snr': [-4.8883, 1.1656]}
> **dt05_str_simu**
> {'pesqw': [2.4848, 0.3498], 'pesqn': [3.061, 0.3108], 'stoi': [0.9615, 0.0265], '_pesq': [2.4848, 0.3498], 'csig': [3.6448, 0.3298], 'cbak': [2.3247, 0.2124], 'covl': [3.0638, 0.3446], '_snr': [-4.775, 1.3388], '_seg_snr': [-5.267, 1.127]}
>
> **total-100**
> {'pesqw': [2.7224, 0.3877], 'pesqn': [3.2361, 0.3258], 'stoi': [0.9728, 0.0185], '_pesq': [2.7224, 0.3877], 'csig': [3.8253, 0.3806], 'cbak': [2.4542, 0.23], 'covl': [3.2803, 0.3867], '_snr': [-4.8472, 1.1894], '_seg_snr': [-5.3082, 1.1124]}
> **dt05_bus_simu**
> {'pesqw': [2.783, 0.4404], 'pesqn': [3.3022, 0.3748], 'stoi': [0.9755, 0.0179], '_pesq': [2.783, 0.4404], 'csig': [3.947, 0.4082], 'cbak': [2.4712, 0.2479], 'covl': [3.3709, 0.4291], '_snr': [-4.8151, 1.0952], '_seg_snr': [-5.4766, 1.0369]}
> **dt05_ped_simu**
> {'pesqw': [2.9441, 0.2881], 'pesqn': [3.3898, 0.249], 'stoi': [0.9786, 0.0115], '_pesq': [2.9441, 0.2881], 'csig': [4.0198, 0.2913], 'cbak': [2.5904, 0.18], 'covl': [3.4931, 0.2898], '_snr': [-4.7369, 1.1609], '_seg_snr': [-5.0089, 1.108]}
> **dt05_caf_simu**
> {'pesqw': [2.6768, 0.3129], 'pesqn': [3.1903, 0.2677], 'stoi': [0.9712, 0.0142], '_pesq': [2.6768, 0.3129], 'csig': [3.7316, 0.32], 'cbak': [2.4395, 0.1898], 'covl': [3.2111, 0.3158], '_snr': [-4.8949, 1.1955], '_seg_snr': [-5.219, 1.1322]}
> **dt05_str_simu**
> {'pesqw': [2.4856, 0.3395], 'pesqn': [3.062, 0.2998], 'stoi': [0.9658, 0.0249], '_pesq': [2.4856, 0.3395], 'csig': [3.603, 0.3389], 'cbak': [2.3157, 0.2084], 'covl': [3.0459, 0.342], '_snr': [-4.9419, 1.2877], '_seg_snr': [-5.5284, 1.0913]}

## The mcse model spdns vtest

```python
python train_mcse_spdns.py --pred --vtest --ckpt trained_mcse_spdns/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcse_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_spdns/pred_mcse_vtest_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi 
```

## The deftan model spdns vtest

```python
python train_deftan_spdns.py --pred --vtest --ckpt trained_mcse_spdns/deftan/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_deftan_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_spdns/pred_deftan_vtest_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi
```

> {'pesqw': [2.0567, 0.9596], 'pesqn': [2.4776, 1.1078], 'stoi': [0.7313, 0.2432], '_pesq': [2.0567, 0.9596], 'csig': [3.1105, 1.3953], 'cbak': [2.1813, 0.5658], 'covl': [2.5429, 1.2143], '_snr': [-2.6229, 0.8695], '_seg_snr': [-2.3438, 1.0696]}

## The mcnet model spdns vtest

```python
python train_mcnet_spdns.py --pred --vtest --ckpt trained_mcse_spdns/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcnet_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_spdns/pred_mcnet_vtest_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi
```

> 


## The mcse model spdns valid

```py
python train_mcse_spdns.py --pred --valid --ckpt trained_mcse_spdns/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcse_valid_50
```

```pyt
python compute_metrics.py --out ../trained_mcse_spdns/pred_mcse_valid_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi
```

> 

## The deftan model spdns valid

```python
python train_deftan_spdns.py --pred --valid --ckpt trained_mcse_spdns/deftan/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_deftan_valid_50
```

```python
python compute_metrics.py --out ../trained_mcse_spdns/pred_deftan_valid_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi
```

> {'pesqw': [2.9617, 0.4292], 'pesqn': [3.537, 0.2808], 'stoi': [0.9493, 0.0169], '_pesq': [2.9617, 0.4292], 'csig': [4.4454, 0.321], 'cbak': [2.7142, 0.2331], 'covl': [3.7079, 0.3854], '_snr': [-2.8983, 0.8935], '_seg_snr': [-2.7355, 1.0799]}

## The mcnet model spdns valid

```py
python train_mcnet_spdns.py --pred --valid --ckpt trained_mcse_spdns/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_spdns/pred_mcnet_valid_50
```

```pyt
python compute_metrics.py --out ../trained_mcse_spdns/pred_mcnet_valid_50  --src ~/datasets/spatialReverbNoise --metrics asr pesqn pesqw stoi
```

>
>

## The mcse model l3das vtest

```python
python train_mcse_l3das.py --pred --vtest --ckpt trained_mcse_l3das/mcse_3en_wdefu_wfafu_c72_B2/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_mcse_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_l3das/pred_mcse_vtest_50  --src ~/datasets/l3das/L3DAS22_Task1_dev --metrics asr pesqn pesqw stoi
```

> {'pesqw': [3.2538, 0.5232], 'pesqn': [3.6823, 0.3361], 'stoi': [0.9574, 0.031], '_pesq': [3.2538, 0.5232], 'csig': [4.6548, 0.3939], 'cbak': [2.9379, 0.2919], 'covl': [3.983, 0.4816], '_snr': [-1.6638, 2.8578], '_seg_snr': [-1.846, 1.9135]}

## The deftan model l3das vtest

```python
python train_deftan_l3das.py --pred --vtest --ckpt trained_mcse_l3das/deftan/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_deftan_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_l3das/pred_deftan_vtest_50  --src ~/datasets/l3das/L3DAS22_Task1_dev --metrics asr pesqn pesqw stoi
```

> {'pesqw': [3.0906, 0.5116], 'pesqn': [3.559, 0.3225], 'stoi': [0.9443, 0.0306], '_pesq': [3.0907, 0.5116], 'csig': [4.5242, 0.4153], 'cbak': [2.7832, 0.2768], 'covl': [3.8203, 0.4773], '_snr': [-2.9765, 0.8568], '_seg_snr': [-2.8275, 0.8377]}

## The mcnet model l3das vtest

```python
python train_mcnet_l3das.py --pred --vtest --ckpt trained_mcse_l3das/mcnet_raw/checkpoints/epoch_0050.pth --out trained_mcse_l3das/pred_mcnet_vtest_50
```

```python
python compute_metrics.py --out ../trained_mcse_l3das/pred_mcnet_vtest_50  --src ~/datasets/l3das/L3DAS22_Task1_dev --metrics asr pesqn pesqw stoi
```

> {'pesqw': [2.8169, 0.5], 'pesqn': [3.3456, 0.3577], 'stoi': [0.9364, 0.0333], '_pesq': [2.8169, 0.5], 'csig': [4.3186, 0.4458], 'cbak': [2.605, 0.2844], 'covl': [3.5718, 0.4819], '_snr': [-3.2305, 1.485], '_seg_snr': [-3.4721, 1.1218]}
