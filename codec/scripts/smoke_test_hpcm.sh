#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/codec_env.sh"
cd "${STABLESR_ROOT}"

CKPT="${CKPT:-/data/Dataset/LIC-HPCM_outputs/train_canny_lambda0.00105/checkpoints/HPCM_Base_lmbda0.00105/epoch_best.pth.tar}"
MODEL="${MODEL:-HPCM_Base}"
[[ -f "${CKPT}" ]] || { echo "Missing ckpt: ${CKPT}" >&2; exit 1; }

"${PYTHON}" - <<PY
import torch
from codec import HPCMCodec, codec_root

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("codec root:", codec_root())
codec = HPCMCodec.from_checkpoint("${CKPT}", model_name="${MODEL}").to(device)
x = torch.rand(1, 3, 512, 512, device=device)
with torch.no_grad():
    y = codec.encode_decode(x)
print("encode_decode:", tuple(y.shape))
packed = codec.compress(x)
rec = codec.decompress(packed["strings"], packed["shape"], packed["orig_size"])
print("compress/decompress:", tuple(rec.shape))
PY
echo OK
