#!/usr/bash
# StableSR full training data -> /data/Dataset/
# Paper/README: DIV8K train_HR + df2k_ost (DIV2K+Flickr2K+OST) + FFHQ 1024 (10k faces)
set -euo pipefail

ROOT="${1:-/data/Dataset}"
LOG="${ROOT}/download_training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== StableSR training data download ==="
echo "ROOT=$ROOT  LOG=$LOG"

download_zip() {
  local url="$1" dest="$2" zipname="${3:-}"
  mkdir -p "$dest"
  [[ -n "$zipname" ]] || zipname="$(basename "$url")"
  local zpath="$dest/$zipname"
  if [[ -f "$zpath" ]]; then
    echo "[skip dl] $zpath exists"
  else
    echo "[wget] $url"
    wget -c "$url" -O "$zpath"
  fi
  echo "[unzip] $zpath -> $dest"
  unzip -n -q "$zpath" -d "$dest" || unzip -n "$zpath" -d "$dest"
}

# --- DIV2K (train + valid) ---
mkdir -p "$ROOT/DIV2K"
if [[ -d "$ROOT/DIV2K/DIV2K_train_HR" ]] && [[ $(ls "$ROOT/DIV2K/DIV2K_train_HR" | wc -l) -ge 800 ]]; then
  echo "[ok] DIV2K_train_HR"
else
  download_zip "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" "$ROOT/DIV2K"
fi
if [[ -d "$ROOT/DIV2K/DIV2K_valid_HR" ]] && [[ $(ls "$ROOT/DIV2K/DIV2K_valid_HR" | wc -l) -ge 100 ]]; then
  echo "[ok] DIV2K_valid_HR"
else
  download_zip "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" "$ROOT/DIV2K"
fi

# --- Flickr2K ---
mkdir -p "$ROOT/Flickr2K"
if [[ -d "$ROOT/Flickr2K/Flickr2K_HR" ]] || [[ -d "$ROOT/Flickr2K/Flickr2K" ]]; then
  echo "[ok] Flickr2K"
else
  download_zip "https://data.vision.ee.ethz.ch/cvl/DIV2K/Flickr2K_HR.zip" "$ROOT/Flickr2K" || {
    echo "[try tar] Flickr2K from SNU"
    mkdir -p "$ROOT/Flickr2K"
    wget -c "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar" -O "$ROOT/Flickr2K/Flickr2K.tar"
    tar -xf "$ROOT/Flickr2K/Flickr2K.tar" -C "$ROOT/Flickr2K"
  }
fi

# --- OST (Real-ESRGAN / OpenMMLab mirror) ---
if [[ -d "$ROOT/OST/images" ]] || [[ -d "$ROOT/OST_dataset/images" ]]; then
  echo "[ok] OST"
else
  download_zip "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip" "$ROOT" || \
  download_zip "https://github.com/xinntao/OST_dataset/releases/download/v0.1.0/OST_dataset.zip" "$ROOT"
fi

# --- DIV8K train (AIM 2019 HR segments 001-1400) ---
DIV8K_HR="$ROOT/DIV8K/train_HR"
mkdir -p "$DIV8K_HR"
BASE="http://data.vision.ee.ethz.ch/timofter/AIM19ExtremeSR"
for seg in 001to200 201to400 401to600 601to800 801to1000 1001to1200 1201to1400; do
  zip="trainHR_${seg}.zip"
  zpath="$ROOT/DIV8K/$zip"
  if [[ -f "$zpath" ]]; then
    echo "[skip dl] $zip"
  else
    echo "[wget] $BASE/$zip"
    wget -c "$BASE/$zip" -O "$zpath" || { echo "WARN: failed $zip"; continue; }
  fi
  echo "[unzip] $zip"
  unzip -n -q "$zpath" -d "$ROOT/DIV8K/tmp_$seg" || unzip -n "$zpath" -d "$ROOT/DIV8K/tmp_$seg"
  find "$ROOT/DIV8K/tmp_$seg" -type f \( -iname '*.png' -o -iname '*.jpg' \) -exec mv -n {} "$DIV8K_HR/" \;
  rm -rf "$ROOT/DIV8K/tmp_$seg"
done
echo "DIV8K train_HR count: $(find "$DIV8K_HR" -type f | wc -l)"

# --- FFHQ 1024 (full zip ~89GB; training uses 10k subset) ---
FFHQ_OUT="$ROOT/FFHQ/1024"
mkdir -p "$ROOT/FFHQ"
if [[ -d "$FFHQ_OUT" ]] && [[ $(find "$FFHQ_OUT" -maxdepth 1 -name '*.png' | wc -l) -ge 10000 ]]; then
  echo "[ok] FFHQ 1024 (>=10k)"
else
  FFHQ_ZIP="$ROOT/FFHQ/images1024x1024.zip"
  if [[ ! -f "$FFHQ_ZIP" ]]; then
    echo "[FFHQ] Downloading images1024x1024 (~89GB) via gdown..."
    if command -v gdown >/dev/null 2>&1; then
      gdown "https://drive.google.com/uc?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL" -O "$FFHQ_ZIP" || \
      gdown "1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv" -O "$FFHQ_ZIP"
    else
      pip install -q gdown
      gdown "https://drive.google.com/uc?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL" -O "$FFHQ_ZIP" || true
    fi
  fi
  if [[ -f "$FFHQ_ZIP" ]]; then
    echo "[unzip] FFHQ (partial: first 10000 for StableSR)..."
    mkdir -p "$FFHQ_OUT"
    unzip -n -q "$FFHQ_ZIP" -d "$ROOT/FFHQ/_extract" || unzip -n "$FFHQ_ZIP" -d "$ROOT/FFHQ/_extract"
    SRC=$(find "$ROOT/FFHQ/_extract" -type d -name 'images1024x1024' | head -1)
    [[ -z "$SRC" ]] && SRC=$(find "$ROOT/FFHQ/_extract" -type f -name '*.png' | head -1 | xargs dirname)
    if [[ -n "$SRC" ]]; then
      ls "$SRC"/*.png 2>/dev/null | head -10000 | while read -r f; do
        cp -n "$f" "$FFHQ_OUT/" 2>/dev/null || ln -f "$f" "$FFHQ_OUT/$(basename "$f")" 2>/dev/null || true
      done
    fi
    echo "FFHQ 1024 count: $(find "$FFHQ_OUT" -maxdepth 1 -name '*.png' | wc -l)"
  else
    echo "WARN: FFHQ download failed — install gdown and retry, or place PNGs in $FFHQ_OUT"
  fi
fi

# --- merge df2k_ost/GT ---
python3 "$(dirname "$0")/prepare_training_datasets.py" --root "$ROOT" --skip-download

# --- DIV2K valid test pairs ---
python3 "$(dirname "$0")/prepare_div2k_valid_test.py" --root "$ROOT/DIV2K"

echo "=== ALL DONE ==="
echo "Log: $LOG"
df -h "$ROOT" | tail -1
