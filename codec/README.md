# StableSR `codec/` — 自包含图像/视频 codec

**不依赖** `/home/exx/Documents/Tianma/LIC-HPCM` 或 git submodule。  
所有代码在 `StableSR/codec/` 内，与 StableSR 同一仓库管理。

## 目录结构

```
codec/
  __init__.py           # from codec import HPCMCodec
  paths.py
  utils.py
  hpcm/
    codec.py            # StableSR 封装
  src/                  # HPCM 模型实现
    models/             # HPCM_Base, HPCM_Base_Lite, HPCM_Video_PFrame, ...
    layers/
    entropy_models/
    datasets/
  train.py              # I-frame 训练
  train_video.py        # 视频 I/P-frame 训练
  test.py
  test_video_iframe.py
  test_video_pframe.py
  scripts/
    setup_codec.sh      # 编译 C++ 扩展（首次）
    build_hpcm_extensions.sh
    smoke_test_hpcm.sh
    hpcm/               # 数据预处理、训练 shell 等
```

将来其他 codec 可加 `codec/jpeg/`、`codec/dcvc/` 等。

## 首次设置

```bash
cd StableSR
bash codec/scripts/setup_codec.sh
bash codec/scripts/smoke_test_hpcm.sh
```

## 训练 HPCM（在 StableSR 内）

```bash
cd StableSR
LAMBDA=0.00105 bash codec/scripts/hpcm/train_hpcm_canny.sh
```

或直接用 Python：

```bash
cd StableSR
python codec/train.py --model_name HPCM_Base --lambda 0.00105 ...
```

## Python 联合训练

```python
from codec import HPCMCodec

codec = HPCMCodec.from_checkpoint("epoch_best.pth.tar", model_name="HPCM_Base")
x_hat = codec.encode_decode(x, training=True)
```

## Lambda

训练时设 `LAMBDA=...`；推理时用 `CKPT=/path/to/ckpt` 选择对应码率点。

```bash
CKPT=/path/to/epoch_best.pth.tar bash codec/scripts/smoke_test_hpcm.sh
```
