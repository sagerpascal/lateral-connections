#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for config in lateral_connection_alternative_cells lateral_connection_alternative_cells_2 lateral_connection_alternative_cells_3 lateral_connection_alternative_cells_4 lateral_connection_alternative_cells_5 lateral_connection_alternative_cells_6 lateral_connection_alternative_cells_7 lateral_connection_alternative_cells_8 lateral_connection_alternative_cells_9
do
  for noise in $(seq 0.0 .01 0.2)
  do
      for li in {0..7}
      do
          python main_evaluation.py $config --load alternative_final.ckp --noise $noise --line_interrupt $li --load_baseline_activations_path ../tmp/alternative_final_baseline.pt;
      done
  done
done