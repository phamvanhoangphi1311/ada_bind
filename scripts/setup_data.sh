#!/usr/bin/env bash
# Unpack the downloaded archives into the layout the code expects.
#
#   bash scripts/setup_data.sh <thu-muc-chua-file-tai-ve> [--base DIR] [--force] [--dry-run]
#
# Every archive carries its own top-level folder, so each one is simply extracted into the project root
# and lands in the right place. Nothing is overwritten unless --force is given.
#
#   pbind_data.zip            -> <BASE>/data/                 CSV, ids_split1..5.txt          BAT BUOC
#   pbind_esmfold_pdb.zip     -> <BASE>/esmFold_pdb_files/    cau truc ESMFold                BAT BUOC
#   pbind_final_model.zip     -> <BASE>/final_model/          80 checkpoint da train          cho inference
#   pbind_feature_store.zip   -> <BASE>/feature_store/        29 block da tinh san             tuy chon
#   pbind_cache.zip           -> <BASE>/cache/                cache GEAR                       tuy chon
#   foundation_models.zip     -> <REPO>/model_weights/        ESM-2 / ProtT5 / ESM-IF / ESMFold  chi khi inference
#
# BASE mac dinh la <repo>/data (giong thu tu cua pbind.paths.resolve_paths).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$REPO/data"
FORCE=0
DRY=0
SRC=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base)    BASE="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)         SRC="$1"; shift ;;
  esac
done

[ -n "$SRC" ] || { echo "dung: bash scripts/setup_data.sh <thu-muc-chua-file-tai-ve> [--base DIR] [--force] [--dry-run]"; exit 2; }
[ -d "$SRC" ] || { echo "khong thay thu muc '$SRC'"; exit 2; }
command -v unzip >/dev/null || { echo "thieu 'unzip'. Cai: sudo apt install unzip"; exit 2; }

SRC="$(cd "$SRC" && pwd)"
mkdir -p "$BASE"
BASE="$(cd "$BASE" && pwd)"
echo "nguon  : $SRC"
echo "BASE   : $BASE"
echo "repo   : $REPO"
[ "$DRY" = 1 ] && echo "(dry-run: khong ghi gi)"
echo

# archive | thu muc no tao ra | giai nen vao | bat buoc?
ARCHIVES="
pbind_data.zip|data|$BASE|yes
pbind_esmfold_pdb.zip|esmFold_pdb_files|$BASE|yes
pbind_final_model.zip|final_model|$BASE|no
pbind_feature_store.zip|feature_store|$BASE|no
pbind_cache.zip|cache|$BASE|no
foundation_models.zip|model_weights|$REPO|no
"

missing_required=0
done_n=0
skipped_n=0

for row in $ARCHIVES; do
  name="${row%%|*}";  rest="${row#*|}"
  top="${rest%%|*}";  rest="${rest#*|}"
  dest="${rest%%|*}"
  required="${rest##*|}"

  archive=""
  for cand in "$SRC/$name" "$SRC/${name%.zip}.tar.gz" "$SRC/${name%.zip}.tgz"; do
    [ -f "$cand" ] && { archive="$cand"; break; }
  done

  if [ -z "$archive" ]; then
    if [ "$required" = "yes" ]; then
      echo "[THIEU ] $name  -> $dest/$top   (BAT BUOC, khong thay trong $SRC)"
      missing_required=$((missing_required+1))
    else
      echo "[bo qua] $name  (tuy chon, khong tai)"
    fi
    continue
  fi

  if [ -e "$dest/$top" ] && [ "$FORCE" = 0 ]; then
    echo "[co san] $dest/$top  -- da ton tai, dung --force de ghi de"
    skipped_n=$((skipped_n+1))
    continue
  fi

  # Doc thu muc goc trong archive truoc khi giai nen: neu khac du doan thi dung lai,
  # vi giai nen nham cho con kho sua hon la bao loi.
  case "$archive" in
    *.zip)           inside="$(unzip -Z1 "$archive" | head -1 | cut -d/ -f1)" ;;
    *.tar.gz|*.tgz)  inside="$(tar -tzf "$archive" | head -1 | cut -d/ -f1)" ;;
  esac
  if [ "$inside" != "$top" ]; then
    echo "[LOI   ] $(basename "$archive"): ben trong la '$inside/', cho doi '$top/'. Bo qua."
    echo "          Tai nham file, hoac zip sai cach. Zip phai chua dung thu muc '$top/' o goc."
    continue
  fi

  echo "[giai  ] $(basename "$archive")  ->  $dest/$top"
  if [ "$DRY" = 0 ]; then
    tmp="$(mktemp -d "$dest/.unpack.XXXXXX")"
    trap 'rm -rf "$tmp"' EXIT
    case "$archive" in
      *.zip)           unzip -q "$archive" -d "$tmp" ;;
      *.tar.gz|*.tgz)  tar -xzf "$archive" -C "$tmp" ;;
    esac
    rm -rf "${dest:?}/$top"
    mv "$tmp/$top" "$dest/$top"
    rm -rf "$tmp"
    trap - EXIT
  fi
  done_n=$((done_n+1))
done

echo
echo "giai nen: $done_n | bo qua vi da co: $skipped_n | thieu bat buoc: $missing_required"
if [ "$missing_required" -gt 0 ]; then
  echo "Thieu archive bat buoc -- xem muc 3 cua README de biet tai o dau."
  exit 1
fi

if [ "$DRY" = 0 ]; then
  echo
  PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY=python3
  "$PY" "$REPO/scripts/check_setup.py" --base "$BASE"
fi
