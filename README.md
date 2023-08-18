# LLM_Calculator
LLM model fine-tuned to solve next math problem: sum of two long integer numbers. In this work uses two base model: BLOOM-3b and DeciCoder-1b. Best result has DeciCoder-1b model.

# Description
You can find detailed descriptions of the train approach and the model [here](https://docs.google.com/presentation/d/1G_FcPpEXNNlcSAK_OUte-UEHKp1EkU2huXxG13fpQ0I/edit?usp=sharing). 

# Dataset
Train dataset for BLOOM-3b model you can find [here](https://drive.google.com/file/d/1VbQs0ZBflZBjSZzp4yM6vuBB8mdT_3wA/view?usp=sharing). Train dataset for DeciCoder-1b model you can find [here](https://drive.google.com/file/d/1Ur07RpZAQkqZy_eGu3HlK99uMRcLs0MK/view?usp=sharing).

# Model fine-tuning
To fine-tune model you have to:
1. Donwload train dataset and put it into dataset folder.
2. Setup python environment:
```
python3 -m venv .venv
source .venv/bin/active
python3 -m pip install -r requirements.txt
```
3. Start to fine-tune model:
```
python3 finetune.py --base_model="Deci/DeciCoder-1b" --lora_weights="decicoder_lora_weights"  --dataset_path="dataset/decicoder-1b_dataset.json"
```

Weights of the model after fine-tune you can find in *lora_weights* folder.

# Model inference
To run model inference with best base model (DeciCoder-1b) you should:
1. Setup python environment:
```
python3 -m venv .venv
source .venv/bin/active
python3 -m pip install -r requirements.txt
```
2. Start app:
```
python3 main.py
```
3. Open this [http://0.0.0.0:7860](http://0.0.0.0:7860) in the browser.

# Model evaluation
To run model evaluation you should:
1. Setup python environment:
```
python3 -m venv .venv
source .venv/bin/active
python3 -m pip install -r requirements.txt
```
2. Run evaluation script for BLOOM-3b model:
```
./benchmark.sh bloom acc
```

Or run evaluation script for DeciCoder-1b model:
```
./benchmark.sh decicoder acc03
```

# Dataset creation
To create dataset you should:
1. Setup python environment:
```
python3 -m venv .venv
source .venv/bin/active
python3 -m pip install -r requirements.txt
```
2. Run script for BLOOM-3b model:
```
python3 create_dataset_bloom.py
```
Or run evaluation script for DeciCoder-1b model:
```
python3 create_dataset_decicoder.py
```