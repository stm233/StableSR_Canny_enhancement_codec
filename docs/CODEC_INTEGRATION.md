# 已迁移 — 见 `codec/README.md`

Codec 完全在 `StableSR/codec/` 内，**不再需要** 外部 `LIC-HPCM` 目录。

```bash
bash codec/scripts/setup_codec.sh
bash codec/scripts/smoke_test_hpcm.sh
```

```python
from codec import HPCMCodec
```
