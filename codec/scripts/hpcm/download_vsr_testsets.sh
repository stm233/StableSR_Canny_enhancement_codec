#!/usr/bin/env bash
# Download RealVSR and MVSR4x test sets into separate directories.
#
# Sources (DOVE repackaged, standard GT/LQ layout):
#   RealVSR: 50 test sequences
#   MVSR4x:  15 test sequences
#
# Usage:
#   bash codec/scripts/hpcm/download_vsr_testsets.sh
#   bash codec/scripts/hpcm/download_vsr_testsets.sh realvsr
#   bash codec/scripts/hpcm/download_vsr_testsets.sh mvsr4x
#   REALVSR_ZIP=/path/to/RealVSR.zip bash codec/scripts/hpcm/download_vsr_testsets.sh realvsr-import
#   GT_TEST_ZIP=... LQ_TEST_ZIP=... bash codec/scripts/hpcm/download_vsr_testsets.sh realvsr-official
#
# Output (default under /data/Dataset/):
#   RealVSR_test/{GT,LQ,GT-Video,LQ-Video,meta.json}
#   MVSR4x_test/{GT,LQ,GT-Video,LQ-Video,meta.json}

source "$(dirname "$0")/env.sh"
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Dataset}"
TARGET="${1:-all}"

REALVSR_DIR="${REALVSR_DIR:-${DATA_ROOT}/RealVSR_test}"
MVSR4X_DIR="${MVSR4X_DIR:-${DATA_ROOT}/MVSR4x_test}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/_vsr_test_downloads}"

# DOVE Google Drive file IDs
REALVSR_ID="1wr4tTiCvQlqdYPeU1dmnjb5KFY4VjGCO"
MVSR4X_ID="16sesBD_9Xx_5Grtx18nosBw1w94KlpQt"

mkdir -p "${CACHE_DIR}"

download_zip() {
  local file_id="$1" zip_path="$2"
  if [[ -f "${zip_path}" ]]; then
    echo "skip download (exists): ${zip_path}"
    return
  fi
  echo "downloading: ${zip_path}"
  gdown "https://drive.google.com/uc?id=${file_id}" -O "${zip_path}"
}

has_dataset_layout() {
  local root="$1"
  [[ -d "${root}/GT" && -d "${root}/LQ" ]]
}

normalize_extracted_tree() {
  local dest="$1"
  local tmp="$2"

  if has_dataset_layout "${tmp}"; then
    mkdir -p "${dest}"
    for sub in GT LQ GT-Video LQ-Video; do
      [[ -d "${tmp}/${sub}" ]] && rsync -a "${tmp}/${sub}/" "${dest}/${sub}/"
    done
    return
  fi

  # DOVE zip may contain a single top-level folder, e.g. RealVSR/ or MVSR4x/
  local child
  child="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -1 || true)"
  if [[ -n "${child}" ]] && has_dataset_layout "${child}"; then
    mkdir -p "${dest}"
    for sub in GT LQ GT-Video LQ-Video; do
      [[ -d "${child}/${sub}" ]] && rsync -a "${child}/${sub}/" "${dest}/${sub}/"
    done
    return
  fi

  echo "ERROR: unrecognized archive layout under ${tmp}" >&2
  find "${tmp}" -maxdepth 3 -type d | head -30 >&2 || true
  exit 1
}

write_meta() {
  local name="$1" dest="$2" source_url="$3"
  local gt_clips lq_clips
  gt_clips="$(find "${dest}/GT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  lq_clips="$(find "${dest}/LQ" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  cat > "${dest}/meta.json" <<EOF
{
  "name": "${name}",
  "root": "${dest}",
  "source": "${source_url}",
  "layout": {
    "GT": "high-quality frame folders (one subdir per clip)",
    "LQ": "low-quality frame folders (one subdir per clip)",
    "GT-Video": "optional lossless GT videos",
    "LQ-Video": "optional lossless LQ videos"
  },
  "num_gt_clips": ${gt_clips},
  "num_lq_clips": ${lq_clips}
}
EOF
}

install_dataset() {
  local name="$1" file_id="$2" dest="$3" zip_name="$4" source_url="$5"
  local max_retries="${6:-3}"
  local attempt=1

  if has_dataset_layout "${dest}"; then
    echo "[ok] ${name} already present: ${dest}"
    write_meta "${name}" "${dest}" "${source_url}"
    return
  fi

  local zip_path="${CACHE_DIR}/${zip_name}"
  local tmp
  tmp="$(mktemp -d "${CACHE_DIR}/${name}.XXXXXX")"

  while [[ "${attempt}" -le "${max_retries}" ]]; do
    if [[ ! -f "${zip_path}" ]]; then
      echo "downloading (attempt ${attempt}/${max_retries}): ${zip_path}"
      if ! gdown "https://drive.google.com/uc?id=${file_id}" -O "${zip_path}"; then
        rm -f "${zip_path}"
        if [[ "${attempt}" -eq "${max_retries}" ]]; then
          echo "ERROR: failed to download ${name} after ${max_retries} attempts." >&2
          echo "       Google Drive may be rate-limited. Retry later:" >&2
          echo "         bash codec/scripts/hpcm/download_vsr_testsets.sh ${name,,}" >&2
          echo "       Official mirror (Baidu, code 43ph):" >&2
          echo "         https://pan.baidu.com/s/1rBIGo5xrY2VtpoUF2gf_HA" >&2
          rm -rf "${tmp}"
          return 1
        fi
        echo "download failed, sleeping 300s before retry..."
        sleep 300
        attempt=$((attempt + 1))
        continue
      fi
    else
      echo "skip download (exists): ${zip_path}"
    fi
    break
  done

  echo "extracting: ${zip_path} -> ${tmp}"
  unzip -q "${zip_path}" -d "${tmp}"

  mkdir -p "${dest}"
  normalize_extracted_tree "${dest}" "${tmp}"
  write_meta "${name}" "${dest}" "${source_url}"
  rm -rf "${tmp}"

  echo "[done] ${name}: ${dest}"
  echo "       GT clips: $(find "${dest}/GT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
}

install_from_zip() {
  local name="$1" dest="$2" zip_path="$3" source_url="$4"
  [[ -f "${zip_path}" ]] || { echo "Missing zip: ${zip_path}"; exit 1; }
  local tmp
  tmp="$(mktemp -d "${CACHE_DIR}/${name}.XXXXXX")"
  echo "extracting manual zip: ${zip_path}"
  unzip -q "${zip_path}" -d "${tmp}"
  mkdir -p "${dest}"
  normalize_extracted_tree "${dest}" "${tmp}"
  write_meta "${name}" "${dest}" "${source_url}"
  rm -rf "${tmp}"
  echo "[done] ${name}: ${dest}"
  echo "       GT clips: $(find "${dest}/GT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
}

install_realvsr_official_test() {
  local dest="$1"
  local gt_zip="${GT_TEST_ZIP:-${CACHE_DIR}/GT_test.zip}"
  local lq_zip="${LQ_TEST_ZIP:-${CACHE_DIR}/LQ_test.zip}"
  [[ -f "${gt_zip}" && -f "${lq_zip}" ]] || {
    echo "Place official test zips first:" >&2
    echo "  ${gt_zip}" >&2
    echo "  ${lq_zip}" >&2
    echo "Baidu (code 43ph): https://pan.baidu.com/s/1rBIGo5xrY2VtpoUF2gf_HA" >&2
    echo "Files needed: GT_test.zip, LQ_test.zip" >&2
    exit 1
  }
  local tmp
  tmp="$(mktemp -d "${CACHE_DIR}/RealVSR_official.XXXXXX")"
  mkdir -p "${tmp}/GT" "${tmp}/LQ"
  echo "extracting ${gt_zip}"
  unzip -q "${gt_zip}" -d "${tmp}/gt_raw"
  echo "extracting ${lq_zip}"
  unzip -q "${lq_zip}" -d "${tmp}/lq_raw"

  # Official zips usually contain GT_test/... or flat sequence folders.
  local gt_src lq_src
  gt_src="$(find "${tmp}/gt_raw" -type d -name 'GT_test' | head -1 || true)"
  lq_src="$(find "${tmp}/lq_raw" -type d -name 'LQ_test' | head -1 || true)"
  [[ -n "${gt_src}" ]] || gt_src="${tmp}/gt_raw"
  [[ -n "${lq_src}" ]] || lq_src="${tmp}/lq_raw"
  rsync -a "${gt_src}/" "${tmp}/GT/"
  rsync -a "${lq_src}/" "${tmp}/LQ/"

  mkdir -p "${dest}"
  normalize_extracted_tree "${dest}" "${tmp}"
  write_meta "RealVSR" "${dest}" "official GT_test.zip + LQ_test.zip"
  rm -rf "${tmp}"
  echo "[done] RealVSR (official test): ${dest}"
  echo "       GT clips: $(find "${dest}/GT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
}

case "${TARGET}" in
  all)
    install_dataset "RealVSR" "${REALVSR_ID}" "${REALVSR_DIR}" "RealVSR_test.zip" \
      "https://drive.google.com/file/d/${REALVSR_ID}/view"
    install_dataset "MVSR4x" "${MVSR4X_ID}" "${MVSR4X_DIR}" "MVSR4x_test.zip" \
      "https://drive.google.com/file/d/${MVSR4X_ID}/view"
    ;;
  realvsr|RealVSR)
    install_dataset "RealVSR" "${REALVSR_ID}" "${REALVSR_DIR}" "RealVSR_test.zip" \
      "https://drive.google.com/file/d/${REALVSR_ID}/view" "${REALVSR_RETRIES:-12}"
    ;;
  realvsr-import|RealVSR-import)
    install_from_zip "RealVSR" "${REALVSR_DIR}" \
      "${REALVSR_ZIP:-${CACHE_DIR}/RealVSR_test.zip}" \
      "manual zip import"
    ;;
  realvsr-official|RealVSR-official)
    install_realvsr_official_test "${REALVSR_DIR}"
    ;;
  mvsr4x|MVSR4x)
    install_dataset "MVSR4x" "${MVSR4X_ID}" "${MVSR4X_DIR}" "MVSR4x_test.zip" \
      "https://drive.google.com/file/d/${MVSR4X_ID}/view"
    ;;
  *)
    echo "Usage: $0 [all|realvsr|mvsr4x|realvsr-import|realvsr-official]" >&2
    exit 1
    ;;
esac

echo ""
echo "Separate test roots:"
echo "  RealVSR: ${REALVSR_DIR}"
echo "  MVSR4x:  ${MVSR4X_DIR}"
