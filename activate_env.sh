#!/bin/bash -e

# set -x

TARGET_ENV=$1

if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please make sure Conda is installed."
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

CURRENT_ENV=$(conda info | grep 'active environment' | awk '{print $4}')
if [ "$CURRENT_ENV" == "None" ]; then
    conda activate $TARGET_ENV
else
    if [ "$CURRENT_ENV" != "$TARGET_ENV" ]; then
        conda deactivate
        conda activate $TARGET_ENV
    fi
fi
CURRENT_ENV=$(conda info | grep 'active environment' | awk '{print $4}')
echo "Python3: $(which python3)"
echo "---------- [$CURRENT_ENV] activated ----------"
