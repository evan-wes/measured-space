#!/usr/bin/env bash
# Download the two NVIDIA TAO Sparse4D NGC model versions used by scripts/run_sparse4d.py (all files of each),
# anonymously, verify every file's size and SHA-256 against NGC's file listing, and only then move it into place
# with `mv -n` (never overwrites).
#
# Usage: bash scripts/download_sparse4d_weights.sh <listing-dir> <destination> <log-file>
#   <listing-dir>  holds NGC's file listings, saved as <model>_<version>_files.json, e.g.
#                    curl -sS -o <listing-dir>/sparse4d_rn50_trainable_v2.2_files.json \
#                      https://api.ngc.nvidia.com/v2/models/nvidia/tao/sparse4d_rn50/versions/trainable_v2.2/files
#                    curl -sS -o <listing-dir>/sparse4d_rn101_trainable_v1.1_files.json \
#                      https://api.ngc.nvidia.com/v2/models/nvidia/tao/sparse4d_rn101/versions/trainable_v1.1/files
#                  (the listings are the reference checksums; inspect them before downloading).
#   <destination>  model root; files go to <destination>/rn50_trainable_v2.2/ and
#                  <destination>/rn101_trainable_v1.1/ (the runner's --model-dir). Created if missing.
#   <log-file>     appended to: one line per file (GET, OK with size and SHA-256, MISMATCH, EXISTS, ...).
# Download URL form (public models, no API key):
#   https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/<model>/<version>/files?redirect=true&path=<file>
# Models: sparse4d_rn50 trainable_v2.2 and sparse4d_rn101 trainable_v1.1 (NVIDIA Open Model License; read the
# model cards before downloading). The files the results were produced with (size in bytes, SHA-256 hex):
#   rn50_trainable_v2.2/sparse4d_warehouse_v2.2_r50.pth   574813977
#     d623d2734ad7d86fa20e114be282e495e452ffa1d75240840a51600656533e7b
#   rn50_trainable_v2.2/experiment.yaml   7884   892c822b795c024f05bf8b42b0572ad91e3d667cd508cf74118ca96d966e94e1
#   rn50_trainable_v2.2/_ov_kmeans900_v2.2_r50.npy   79328
#     1a1dc46e725090ccab9a716834de9dec8730516bc8cf055cc8efe5b1a7a0d0b7
#   rn101_trainable_v1.1/sparse4d_tracking_aic25v0.5_warehousev3_classes_iter_17400_v1.1.pth   804455085
#     5537862fb1a259ff8421fb114c2836c325d42c94997917b3f34e6c9054ac6e9e
#   rn101_trainable_v1.1/experiment.yaml   7023   b88d5a55ac65ca44223f6b0f575e6772db88a2d373f974fecca8785152dcc26c
#   rn101_trainable_v1.1/_ov_kmeans900_v0.5.npy   79328
#     8601fb1b61b8ba8cdc70523517bfc1b959904c1f9931f6d723d0abc0c20a1dd1
# (the runner records the checkpoint's SHA-256 in every <scene>.meta.json).
#
# A file that already exists is skipped; a leftover <file>.part is never overwritten (move it away yourself); a
# file whose size or SHA-256 does not match is left as <file>.part and logged as MISMATCH. Needs curl, python3,
# sha256sum, xxd and base64. Nothing is deleted.
set -u
if [ $# -ne 3 ]; then
  echo "usage: $0 <listing-dir> <destination> <log-file>" >&2; exit 2
fi
LIST=$1; DST=$2; LOG=$3
declare -A DIRS=( [sparse4d_rn50/trainable_v2.2]=rn50_trainable_v2.2 [sparse4d_rn101/trainable_v1.1]=rn101_trainable_v1.1 )
for mv in "sparse4d_rn50/trainable_v2.2" "sparse4d_rn101/trainable_v1.1"; do
  m=${mv%/*}; v=${mv#*/}; out=$DST/${DIRS[$mv]}; mkdir -p "$out"
  json=$LIST/${m}_${v}_files.json
  python3 -I -c 'import json,sys; [print(f["path"], f["sizeInBytes"], f["sha256_base64"]) for f in json.load(open(sys.argv[1]))["modelFiles"]]' "$json" |
  while read -r path size b64; do
    final=$out/$path; part=$final.part
    if [ -e "$final" ]; then echo "[$(date -Is)] EXISTS $final (skipped, not overwritten)" >> "$LOG"; continue; fi
    if [ -e "$part" ]; then echo "[$(date -Is)] PART EXISTS $part, refusing to overwrite" >> "$LOG"; continue; fi
    url="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/$m/$v/files?redirect=true&path=$path"
    echo "[$(date -Is)] GET $url -> $final" >> "$LOG"
    curl -sS -L --fail --retry 3 -o "$part" "$url" >> "$LOG" 2>&1 || { echo "[$(date -Is)] CURL FAILED $path" >> "$LOG"; continue; }
    got=$(stat -c %s "$part")
    hex=$(sha256sum "$part" | cut -d' ' -f1)
    gotb64=$(echo -n "$hex" | xxd -r -p | base64)
    if [ "$got" = "$size" ] && [ "$gotb64" = "$b64" ]; then
      mv -n "$part" "$final"
      echo "[$(date -Is)] OK $m/$v/$path size=$got sha256_hex=$hex sha256_b64=$gotb64" >> "$LOG"
    else
      echo "[$(date -Is)] MISMATCH $m/$v/$path size=$got (want $size) b64=$gotb64 (want $b64); left as $part" >> "$LOG"
    fi
  done
done
echo "[$(date -Is)] DONE" >> "$LOG"
