#!/bin/bash
# This script evaluate model with as little RAM usage as possible. BLOOM model has problem with memory leak.
python3 create_benchmark_dataset.py --dataset_path=benchmark_dataset_full.jsonl --dataset_type=$2 --size_of_dataset=100

acc=()

if [[ $1 == "decicoder" ]]; then
    acc=$(python3 benchmark.py --base_model="Deci/DeciCoder-1b" --lora_weights="decicoder_lora_weights" --dataset_path=benchmark_dataset_full.jsonl)
    echo Accurasy is $acc %
    rm tmp.json benchmark_dataset_full.jsonl
    exit 0
fi


while read -r line; do
    echo "${#acc[@]}"
    echo $line > tmp.json;
    acc_tmp=$(python3 benchmark.py --dataset_path=tmp.json)
    acc=(${acc[@]} $acc_tmp)
done < benchmark_dataset_full.jsonl

acc=$(echo ${acc[@]} | python3 -c 'print(sum([float(x) for x in input().split(" ")]) / 100.0)')

echo Accurasy is $acc %

rm tmp.json benchmark_dataset_full.jsonl