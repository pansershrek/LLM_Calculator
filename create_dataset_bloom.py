import argparse
import json
import random

def setup_seed(seed: int):
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dataset/bloom-3b-add-dataset.json", help="Path to dataset")
    parser.add_argument("--seed", default=4311, help="Random seed")
    args = parser.parse_args()
    setup_seed(args.seed)

    pairs = \
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(1,16) for j in range(i,16) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(3,16) for j in range(i,16) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(6,16) for j in range(i,16) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(9,16) for j in range(i,16) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(12,16) for j in range(i,16) for k in range(1000)]

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


    template_name = "template.json"

    with open(template_name) as fp:
        template = json.load(fp)

    data_converted = []

    for instance in data_add:
        arithmetic = instance["input"]

        output_dict = {}

        if random.random() < 0.05:
            if " + " in arithmetic:
                arithmetic = "the sum of " + arithmetic.replace("+", "and")

        if random.random() < 0.1:
            arithmetic = arithmetic.replace("+", "plus")

        num = random.randint(1, 500)

        instruction = template[str(num)].format(
            input = arithmetic
        )

        output_dict["instruction"] = instruction
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