#!/bin/bash
# This script evaluate model with as little RAM usage as possible
python3 create_benchmark_dataset.py --dataset_path=benchmark_dataset_full.jsonl --size_of_dataset=100

cat benchmark_dataset_full.jsonl | head -10 > benchmark_dataset_1.jsonl
cat benchmark_dataset_full.jsonl | head -20 | tail -10 > benchmark_dataset_2.jsonl
cat benchmark_dataset_full.jsonl | head -30 | tail -20 > benchmark_dataset_3.jsonl
cat benchmark_dataset_full.jsonl | head -40 | tail -30 > benchmark_dataset_4.jsonl
cat benchmark_dataset_full.jsonl | head -50 | tail -40 > benchmark_dataset_5.jsonl
cat benchmark_dataset_full.jsonl | head -60 | tail -50 > benchmark_dataset_6.jsonl
cat benchmark_dataset_full.jsonl | head -70 | tail -60 > benchmark_dataset_7.jsonl
cat benchmark_dataset_full.jsonl | head -80 | tail -70 > benchmark_dataset_8.jsonl
cat benchmark_dataset_full.jsonl | head -90 | tail -80 > benchmark_dataset_9.jsonl
cat benchmark_dataset_full.jsonl | tail -90 > benchmark_dataset_10.jsonl



acc1=$(python3 benchmark.py --dataset_path=benchmark_dataset_1.jsonl)
acc2=$(python3 benchmark.py --dataset_path=benchmark_dataset_2.jsonl)
acc3=$(python3 benchmark.py --dataset_path=benchmark_dataset_3.jsonl)
acc4=$(python3 benchmark.py --dataset_path=benchmark_dataset_4.jsonl)
acc5=$(python3 benchmark.py --dataset_path=benchmark_dataset_5.jsonl)
acc6=$(python3 benchmark.py --dataset_path=benchmark_dataset_6.jsonl)
acc7=$(python3 benchmark.py --dataset_path=benchmark_dataset_7.jsonl)
acc8=$(python3 benchmark.py --dataset_path=benchmark_dataset_8.jsonl)
acc9=$(python3 benchmark.py --dataset_path=benchmark_dataset_9.jsonl)
acc10=$(python3 benchmark.py --dataset_path=benchmark_dataset_10.jsonl)

acc=$(echo $acc1+$acc2+$acc3+$acc4+$acc5+$acc6+$acc7+$acc8+$acc9+$acc10 | python3 -c 'print(eval(input()) / 10.0)')

echo Accurasy is $acc %

rm benchmark_dataset_full.jsonl benchmark_dataset_1.jsonl benchmark_dataset_2.jsonl benchmark_dataset_3.jsonl benchmark_dataset_4.jsonl benchmark_dataset_5.jsonl
rm benchmark_dataset_6.jsonl benchmark_dataset_7.jsonl benchmark_dataset_8.jsonl benchmark_dataset_9.jsonl benchmark_dataset_10.jsonl