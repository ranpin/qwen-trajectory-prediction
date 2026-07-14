#!/bin/bash
# OPTIONAL — drive the Kaggle notebook headlessly via the Kaggle API, so I (or you)
# can run the export/quant without the browser. Prereqs:
#   pip install kaggle
#   ~/.kaggle/kaggle.json  = your Kaggle API token (chmod 600)
#   In Kaggle UI, add a Notebook Secret named HF_TOKEN once (API can't set secrets).
# Usage:  KAGGLE_USER=<your_kaggle_username> bash kaggle_run.sh
set -e
: "${KAGGLE_USER:?set KAGGLE_USER=your_kaggle_username}"
SLUG="$KAGGLE_USER/alpamayo-edge-export"
DIR=$(mktemp -d)
cp "$(dirname "$0")/kaggle_export_quant.ipynb" "$DIR/kaggle_export_quant.ipynb"
cat > "$DIR/kernel-metadata.json" <<JSON
{
  "id": "$SLUG",
  "title": "alpamayo-edge-export",
  "code_file": "kaggle_export_quant.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": true,
  "enable_internet": true,
  "is_private": true
}
JSON
kaggle kernels push -p "$DIR"
echo "pushed $SLUG"
echo "poll:      kaggle kernels status $SLUG"
echo "download:  kaggle kernels output $SLUG -p ./kaggle_out   # -> edge_artifacts.tgz"
