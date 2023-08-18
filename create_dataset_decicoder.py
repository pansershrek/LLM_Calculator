import argparse
import json
import random

def setup_seed(seed: int):
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dataset/decicoder-1b-add-dataset.json", help="Path to dataset")
    parser.add_argument("--seed", default=4311, help="Random seed")
    args = parser.parse_args()
    setup_seed(args.seed)

    pairs = [(random.randint(1, 10**20), random.randint(1, 10**20)) for _ in range(5000000)]

    random.shuffle(pairs)

    data_add = []

    for num1, num2 in pairs:

        if random.random()<0.5:
            num1, num2 = num2, num1

        answer = num1 + num2

        question = f"{num1} + {num2}"
        output = f"{num1} + {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_add.append({"input": question, "output": output, "answer": str(answer)})


    data_converted = []

    for instance in data_add:

        output_dict = {}
        output_dict["instruction"] = nstance["input"]
        output_dict["input"] = instance["input"]
        output_dict["output"] = instance["output"]
        output_dict["answer"] = instance["answer"]

        data_converted.append(output_dict)


    print("Total:", len(data_converted))

    with open(args.dataset_path, "w") as f:
        json.dump(data_converted, f, indent=4)

    print("Instructions added!")


if __name__ == "__main__":
    main()