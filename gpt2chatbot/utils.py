import os
import json

if __name__ == "__main__":
    dataset_path = os.path.realpath("../data/personachat/train_full.json")

    with open(dataset_path) as f:
        data = json.load(f)

    print(data[1])