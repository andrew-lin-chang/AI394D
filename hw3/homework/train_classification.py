import torch
import argparse
import numpy as np
from .models import Classifier, load_model, save_model
from .datasets.classification_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load and augment data
    train_data = load_data("classification_data/train", return_dataloader=False)
    train_data_aug = load_data("classification_data/train", transform_pipeline="aug", return_dataloader=False)
    train_data_full = torch.utils.data.ConcatDataset([train_data, train_data_aug])

    train_dataloader = torch.utils.data.DataLoader(train_data_full, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = load_data("classification_data/val", shuffle=False)

    # loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    metrics = {"train_acc": [], "val_acc": []}

    # training loop 
    for epoch in range(num_epoch):

        for key in metrics:
            metrics[key].clear()

        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            pred = model.predict(img)
            batch_train_acc = (pred == label).float().mean().item()
            metrics["train_acc"].append(batch_train_acc)

        # validation
        with torch.inference_mode():
            model.eval()

            for img, label in val_dataloader:
                img, label = img.to(device), label.to(device)

                pred = model.predict(img)
                batch_val_acc = (pred == label).float().mean().item()
                metrics["val_acc"].append(batch_val_acc)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    save_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))