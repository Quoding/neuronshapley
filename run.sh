#!/bin/bash
for j in {0..0}
do

  echo "Currently on seed $j"
  python -u cb_aggregate.py all accuracy 25000 False $j > output_agg_$j &
  # python -u cb_aggregate.py all accuracy 25000 False $j &
  for i in {0..0}
  do
    echo "Currently starting cb $i"
    # python3 -u cb_run.py $j accuracy & 
    python3 -u cb_run.py $j accuracy > output_cb_run_${j}_${i} & 
  done
  wait
done
