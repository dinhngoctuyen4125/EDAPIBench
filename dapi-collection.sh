#!/bin/bash

for model in deepseek-1.3b starcoder-3b qwencoder-3b; do
  for lib in pandas pytorch scipy seaborn sklearn tensorflow transformers; do
    python dapi-collection/dapi_inference.py --model "$model" --lib "$lib"
  done
done

python dapi-collection/predicted_dapi_collection.py