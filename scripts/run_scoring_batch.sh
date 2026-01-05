#!/bin/bash
# scripts/run_scoring_batch.sh
# Usage: ./scripts/run_scoring_batch.sh <config_path>

CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

# List of models to iterate over
# Ensure these keys exist in src/llm/local_llm.py MODELS registry
MODELS=("mistral-v0.3" "phi-3") 

echo "---------------------------------------------------"
echo "Starting Batch Scoring for config: $CONFIG"
echo "Models: ${MODELS[*]}"
echo "---------------------------------------------------"

for model in "${MODELS[@]}"; do
  echo "[$(date)] Running scoring for model: $model"
  
  # Run the python script with the override
  python src/score_dataset.py "$CONFIG" --override scoring_model_key="$model"
  
  if [ $? -ne 0 ]; then
    echo "ERROR: Scoring failed for $model"
    # Optional: exit 1 # Stop or continue? Let's continue.
  else
    echo "SUCCESS: Finished $model"
  fi
  echo "---------------------------------------------------"
done

echo "[$(date)] Batch scoring complete. Results are in results/scores/"
