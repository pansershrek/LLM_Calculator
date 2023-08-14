# LLM_Calculator
LLM model (BLOOM-3b) fine-tuned to solve next math problem: sum of two long integer numbers.

# Description
You can find detailed descriptions of the train approach and the model [here](https://docs.google.com/presentation/d/1G_FcPpEXNNlcSAK_OUte-UEHKp1EkU2huXxG13fpQ0I/edit?usp=sharing). 

# Dataset
Train dataset you can find [here](https://drive.google.com/file/d/1VbQs0ZBflZBjSZzp4yM6vuBB8mdT_3wA/view?usp=sharing).

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
python3 finetune.py
```

Weights of the model after fine-tune you can find in *lora_weights* folder.

# Model inference
To run model inference you should:
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
2. Run evaluation script:
```
./benchmark.sh
```