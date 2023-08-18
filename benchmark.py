import argparse
import random
import gc
import sys
import json

import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from prompter import Prompter

def test_model(
    test_queries, prompter, tokenizer, model, device
):
    accuracy = []
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )
    generate_params = {
        "input_ids": None,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": 512,
    }
    for idx, test_query in enumerate(test_queries):
        prompt = prompter.generate_prompt(test_query["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generate_params["input_ids"] = input_ids
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True).strip()
        try:
            output_result = output.split("=")[1].strip()
            accuracy.append(int(output_result == test_query["answer"]))
        except:
            pass
        gc.collect()
    return 0 if len(accuracy) == 0 else sum(accuracy) / len(accuracy) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", default="bigscience/bloom-3b",
        help="Base model name. Deci/DeciCoder-1b or bigscience/bloom-3b"
    )
    parser.add_argument(
        "--lora_weights", default="bloom_lora_weights",
        help="Path to lora weights folder"
    )
    parser.add_argument(
        "--dataset_path", default="", help="Path to dataset"
    )
    args = parser.parse_args()

    device = "cpu"

    prompter = Prompter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map=device, low_cpu_mem_usage=True
    )
    model.eval()
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        device_map={"": device},
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    #test_queries = generate_test_queries(args.ammount_of_test_sums)
    with open(args.dataset_path, "r") as f:
        test_queries = []
        for line in f:
            test_queries += [json.loads(line)]
    accuracy = test_model(
        test_queries, prompter, tokenizer, model, device
    )
    print(accuracy)


if __name__ == "__main__":
    main()