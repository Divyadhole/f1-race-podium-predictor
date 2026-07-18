import json

from f1_podium.training import train_and_evaluate


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))

