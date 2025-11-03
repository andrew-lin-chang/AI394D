import argparse
from ray import tune
from homework.models import MLPClassifier
from homework.train import train

def trainable(config):
    train(model_name=config["model_name"], lr=config["lr"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    param_space = {
        "model_name": args.model_name,
        "lr": tune.grid_search([0.001, 0.01, 0.1])
    }

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=param_space
    )

    results = tuner.fit()
    print(results.get_best_result(metric="val_accuracy", mode="max").config)
