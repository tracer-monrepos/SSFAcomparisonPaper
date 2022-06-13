#!/usr/bin/env bash

# ---------------------------
# Putting all plots together.
# ---------------------------

# check number args
if [[ $# -ne 2 ]]; then
    echo "Please provide input directory wildcard and output filename!" >&2
    exit 2
fi

# parse args
inputDir=$1
outFile=$2

echo ${inoutDir}

echo ${outFile}

exit 1

montage plots/statistical_model_two_factors_filter_strong/treatment_diff_*.pdf -tile 2x5 -geometry 960x640+64+64 montage.pdf

pdfjam --papersize '{190mm,240mm}' -o montage2.pdf montage.pdf
