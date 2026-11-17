#!/usr/bin/env bash
# Main analysis plus the robustness checks listed in the README.
set -e
python run_pipeline.py --tag main
python run_pipeline.py --tag robust_no_pandemic --exclude-pandemic --skip-tuning
python run_pipeline.py --tag robust_no_winsor   --no-winsorize     --skip-tuning
python run_pipeline.py --tag robust_abs_target  --target absolute  --skip-tuning
python run_pipeline.py --tag robust_band_0p5    --stable-band 0.5  --skip-tuning
python run_pipeline.py --tag robust_band_2      --stable-band 2    --skip-tuning
