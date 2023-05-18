#!/bin/bash
python -u cb_aggregate.py all accuracy 25000 False > output_agg &
for i in $(seq 0 1)
do
    python3 -u cb_run.py all accuracy 25000 False > output_cb_run_$i &
done