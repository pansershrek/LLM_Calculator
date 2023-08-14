import argparse
import json
import random

def setup_seed(seed: int):
    random.seed(seed)

def get_pairs(ammount_of_numbers):
    ans = []
    while True:
        for i in range(1, 16):
            for j in range(i, 16):
                a = random.randint(10**(i-1), 10**i)
                b = random.randint(10**(j-1), 10**j)
                ans.append((a, b))
                if len(ans) == ammount_of_numbers:
                    return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="kek.json", help="Path to dataset")
    parser.add_argument("--seed", default=4311, help="Random seed")
    parser.add_argument("--size_of_dataset", default=100, type=int, help="Size of dataset")
    args = parser.parse_args()
    setup_seed(args.seed)

    data_add = []
    pairs = get_pairs(args.size_of_dataset)

    for (num1, num2) in pairs:

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
        output_dict = {}
        output_dict["instruction"] = instance["input"]
        output_dict["input"] = instance["input"]
        output_dict["output"] = instance["output"]
        output_dict["answer"] = instance["answer"]

        data_converted.append(output_dict)


    print("Total:", len(data_converted))

    with open(args.dataset_path, "w") as f:
        for x in data_converted:
            print(json.dumps(x), file=f)

    print("Instructions added!")


if __name__ == "__main__":
    main()