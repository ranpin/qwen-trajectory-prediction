#!/bin/bash
# Run Alpamayo action inference ON the edge device (after export_and_build.sh Step 3).
# Input:  input_action.json — 4 cameras x 4 timesteps images + trajectory history [x,y,z]
# Output: output_action.json — output_text (chain-of-thought) + output_trajectory [(accel,kappa)...]
# Then:   eval/action_to_traj.py -> (x,y) ; eval/evaluate.py -> ADE/FDE vs GT + CVM
set -e
W="${W:-$HOME/tensorrt-edgellm-workspace/${MODEL_NAME:-Alpamayo-R1-10B}}"
./build/examples/multimodal/action_inference \
  --engineDir "$W/engines/llm" \
  --multimodalEngineDir "$W/engines" \
  --inputFile "$W/input_action.json" \
  --outputFile "$W/output_action.json"
echo "output -> $W/output_action.json"
