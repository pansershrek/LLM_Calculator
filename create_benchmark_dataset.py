import argparse
import json
import random

RANGE_TYPE = {
    "acc03": [1, 10**3],
    "acc36": [10**3, 10**6],
    "acc69": [10**6, 10**9],
    "acc912": [10**9, 10**12],
    "acc1617": [10**16, 10**17 - 1],
}

def setup_seed(seed: int):
    random.seed(seed)

def get_pairs(ammount_of_numbers, dataset_type):
    ans = []
    if dataset_type =="acc":
        while True:
            for i in range(1, 16):
                for j in range(i, 16):
                    a = random.randint(10**(i-1), 10**i)
                    b = random.randint(10**(j-1), 10**j)
                    ans.append((a, b))
                    if len(ans) == ammount_of_numbers:
                        return ans
    else:
        while len(ans) < ammount_of_numbers:
            a = random.randint(
                RANGE_TYPE[dataset_type][0],
                RANGE_TYPE[dataset_type][1]
            )
            b = random.randint(
                RANGE_TYPE[dataset_type][0],
                RANGE_TYPE[dataset_type][1]
            )
            ans.append((a, b))
        return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", default="kek.json", help="Path to dataset"
    )
    parser.add_argument("--seed", default=4311, help="Random seed")
    parser.add_argument(
        "--size_of_dataset", default=100, type=int, help="Size of dataset"
    )
    parser.add_argument(
        "--dataset_type", default="acc",
        help="Dataset type. There are next options: "
        "acc, acc03, acc36, acc69, acc912, acc1617."
    )
    args = parser.parse_args()

    if args.dataset_type not in [
        "acc", "acc03", "acc36", "acc69", "acc912", "acc1617"
    ]:
        args.dataset_type = "acc"
    print(f"Dataset type is {args.dataset_type}")
    setup_seed(args.seed)

    data_add = []
    pairs = get_pairs(args.size_of_dataset, args.dataset_type)

    for (num1, num2) in pairs:

        if random.random()<0.5:
            num1, num2 = num2, num1

        answer = num1 + num2

        question = f"{num1} + {num2}"
        output = f"{num1} + {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_add.append(
            {"input": question, "output": output, "answer": str(answer)}
        )

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