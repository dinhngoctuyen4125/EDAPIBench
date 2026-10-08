#!/bin/bash

python eval-data-construction/merge_functions_inputs.py
python eval-data-construction/generalization_data_collection.py
python eval-data-construction/specificity_data_collection.py
python eval-data-construction/generate_case_id.py
python eval-data-construction/chose_generalization_data.py

for model in deepseek-1.3b starcoder-3b qwencoder-3b; do
  for lib in pandas pytorch scipy seaborn sklearn tensorflow transformers; do
    python eval-data-construction/chose_specificity_data.py --model "$model" --lib "$lib"
  done
done

python eval-data-construction/merge_data_for_llms.py
python eval-data-construction/chose_portability_data.py