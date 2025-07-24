import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import os
from pathlib import Path
from .main import LOGGING_DIR


def plot_lightning_logs(csv_path: str, name, version):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 25))
    fig.suptitle(f"Results of the: {name} version {version}", fontsize=16)

    # --- 'train_loss' vs 'step' ---
    train_loss_df = df[["step", "train_loss"]].dropna()
    if not train_loss_df.empty:
        axes[0].plot(train_loss_df["step"], train_loss_df["train_loss"], color="blue")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- val vs 'epoch' ---
    val_df = df[["epoch", "val_loss", "val_accuracy"]].dropna()
    val_df = val_df.groupby("epoch").mean().reset_index()
    if not val_df.empty:
        ax2_twin = axes[1].twinx()
        axes[1].plot(
            val_df["epoch"],
            val_df["val_loss"],
            "o-",
            color="red",
            label="Val loss",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss", color="red")
        axes[1].tick_params(axis="y", labelcolor="red")
        axes[1].grid(True, linestyle="--", alpha=0.6)

        ax2_twin.plot(
            val_df["epoch"],
            val_df["val_accuracy"],
            "o--",
            color="green",
            label="Val accuracy",
        )
        ax2_twin.set_ylabel("Accuracy", color="green")
        ax2_twin.tick_params(axis="y", labelcolor="green")

        axes[1].set_title("Validation")

    # -- text box with test result --
    test_df = df[["test_loss", "test_accuracy"]].dropna()
    test_loss = test_df["test_loss"].mean()
    test_acc = test_df["test_accuracy"].mean()
    avg_epoch_time = (
        df[["epoch", "epoch_time"]]
        .dropna()
        .groupby("epoch")
        .mean()
        .reset_index()["epoch_time"]
        .mean()
    )
    print(avg_epoch_time)

    ax = axes[2]
    axes[2].axis("off")

    text = ax.text(
        0,
        0,
        f"""Test summary: accuracy = {test_acc:.2f}%, loss = {test_loss:.4f}, avg_epoch_time: {avg_epoch_time:.2f}s""",
        color="black",
        fontsize=12,
        weight="bold",
    )

    ax.set_title("Test results ")

    # plt.tight_layout(
    #     rect=[0, 0.03, 1, 0.97]
    # )
    plt.show()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--version", type=int, default=None, help="Version of experiment"
    )
    parser.add_argument(
        "--name", type=str, default="default_name", help="Name of the experiment"
    )

    args = parser.parse_args()

    version = args.version if args.version is not None else 0
    path = LOGGING_DIR / args.name / f"version_{version}" / "metrics.csv"

    plot_lightning_logs(str(path), args.name, version)


if __name__ == "__main__":
    main()
