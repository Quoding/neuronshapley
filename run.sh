#!/bin/bash
for j in {0..24}
do
  echo "Currently on seed $j"
  python -u cb_aggregate.py all accuracy 25000 False $j > output_agg &
  for i in {0..4}
  do
    echo "Currently starting cb $i"
    python3 -u cb_run.py $j accuracy > output_cb_run_$j_$i & 
  done
  wait
done
