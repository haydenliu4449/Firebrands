#!/usr/bin/env bash
# Push footage up to the bucket, and pull results back down.
#
#   bash cloud/upload.sh clips/                  # upload a folder of clips
#   bash cloud/upload.sh masks/left_front.json   # upload one file
#   bash cloud/upload.sh --down work/e1          # download results
#
# Upload is free. Download (egress) is ~$0.12/GB, so pull back CSVs and contact
# sheets, not the 4K overlay videos -- watch those in the notebook instead.

set -euo pipefail
BUCKET="${BUCKET:?set BUCKET (see cloud/setup.sh output)}"

if [ "${1:-}" = "--down" ]; then
  SRC="gs://$BUCKET/${2:?what to download}"
  echo "downloading $SRC -> ./$(basename "$2")"
  gcloud storage cp -r "$SRC" .
  exit 0
fi

SRC="${1:?what to upload}"
if [ -d "$SRC" ]; then
  DEST="gs://$BUCKET/$(basename "${SRC%/}")/"
  echo "uploading $SRC -> $DEST"
  # -n skips objects that already exist, so a re-run after a dropped
  # connection resumes instead of re-sending gigabytes.
  gcloud storage cp -r -n "${SRC%/}/." "$DEST"
else
  DEST="gs://$BUCKET/$(basename "$(dirname "$SRC")")/$(basename "$SRC")"
  echo "uploading $SRC -> $DEST"
  gcloud storage cp "$SRC" "$DEST"
fi
echo
gcloud storage ls -l "gs://$BUCKET/**" | tail -20
