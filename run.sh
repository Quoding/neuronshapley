python cb_aggregate.py all accuracy 25000 False
for i in $(seq 0 10)
do
    python3 cb_run.py all accuracy 25000 False
done