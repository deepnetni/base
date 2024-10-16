# Augment scripts tutorials

## 生成单通道带噪语音；

``````python
python augment_sig.py --yaml template/synthesizer_config_16.yaml --outdir /home/deepni/datasets/dns --time 150
``````

## 生成多通道带噪数据；

```python
# generate rirs
python augment_rir.py --num 10000 --yaml template/spatialReverb.yaml
# generate wavs with rirs
python augment_spatial_v2.py --yaml template/spatialReverb.yaml --wav --hour 2 --out ~/disk/spatialReverbNoise/test
```



``````python
python augment_spatial_v2.py --yaml template/spatialReverbTest.yaml --rir --num 40000
python augment_spatial_v2.py --yaml template/spatialReverbTest.yaml --wav --hour 2 --out ~/disk/spatialReverbNoise/test
``````

## 生成 VAD 数据

```python
 python augment_vad.py --hour 300 --out /home/deepni/disk/vad_by_libri_300
```

