import itertools
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import optim

from src.optim.dOGR import dOGR
from tests.datamodule import CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule
from tests.main import net_dict, run
from tests.nets import FC, LeNet

# --- Główne Ustawienia Eksperymentu ---
LOGGING_DIR = Path("logs")
MAX_EPOCHS = 150
BATCH_SIZE = 256


# Słownik z DataModules dla łatwej iteracji
# datamodules = {
#     "MNIST": MNISTDataModule,
#     "FashionMNIST": FashionMNISTDataModule,
#     "CIFAR10": CIFAR10DataModule
# }
datamodules = {"FashionMNIST": FashionMNISTDataModule}

# 1. Konfiguracja dla Twojego najlepiej dostrojonego dOGR
dogr_config = {
    "opt": dOGR,
    "args": {
        "lr": 1e-3,
        "nonlinear_clipping": True,
        "p_norm": 0.57,
        "p_eps": 10.77,
        "beta": 0.1,
        "trust_factor": 0.5,
    },
}

# 2. Siatka parametrów do przeszukania dla Adama (Grid Search)
adam_grid_search = {
    "opt": optim.Adam,
    "grid": {
        "lr": [1e-2, 3e-3, 1e-3, 1e-4],  # 4 values
        "betas": [(0.9, 0.999), (0.9, 0.99), (0.8, 0.999)],  # 3 values
        "weight_decay": [0, 1e-5, 1e-4, 1e-3],  # 4 values
        "eps": [1e-8],  # 1 values
    },
}

# net_dict = {
#     "FC": FC,
#     "LeNet": LeNet
# }
net_dict = {
    "FC": FC,
}

# Słownik do przechowywania najlepszych wyników dla każdego datasetu
best_results = {}


def run_experiments_on_all_architectures():
    """
    Uruchamia serie eksperymentów dla wszystkich zdefiniowanych architektur sieci
    i wszystkich zdefiniowanych zbiorów danych.
    """
    for net_name in net_dict.keys():
        print(
            f"\n{'=' * 25} ROZPOCZYNANIE TESTÓW DLA ARCHITEKTURY: {net_name.upper()} {'=' * 25}"
        )

        for dataset_name, datamodule_class in datamodules.items():
            print(f"\n{'--' * 10} ZBIÓR DANYCH: {dataset_name.upper()} {'--' * 10}")

            dm = datamodule_class(batch_size=BATCH_SIZE)

            experiment_name = f"dOGR_vs_Adam_{net_name}_{dataset_name}"

            print(f"\n--- Uruchamianie dOGR (wersja 'dogr_tuned') ---")
            torch.manual_seed(42)
            net_dogr = net_dict[net_name](
                input_dims=dm.dims, num_classes=dm.num_classes
            )
            optimizer_dogr = dogr_config["opt"](
                net_dogr.parameters(), **dogr_config["args"]
            )
            run(
                net=net_dogr,
                optimizer=optimizer_dogr,
                name=experiment_name,
                version="dogr_tuned",
                datamodule=dm,
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
            )

            print(f"\n--- Uruchamianie Grid Search dla Adama ---")

            keys, values = zip(*adam_grid_search["grid"].items())
            param_combinations = [
                dict(zip(keys, v)) for v in itertools.product(*values)
            ]

            adam_runs_results = []

            for i, params in enumerate(param_combinations):
                betas_str = (
                    str(params["betas"]).replace(" ", "").replace(",", "_").strip("()")
                )
                version_name = f"Adam_lr={params['lr']}_betas={betas_str}_wd={params.get('weight_decay', 0)}"
                print(
                    f"[{i + 1}/{len(param_combinations)}] Adam z parametrami: {params}"
                )

                torch.manual_seed(42)
                net_adam = net_dict[net_name](
                    input_dims=dm.dims, num_classes=dm.num_classes
                )
                optimizer_adam = adam_grid_search["opt"](
                    net_adam.parameters(), **params
                )

                run(
                    net=net_adam,
                    optimizer=optimizer_adam,
                    name=experiment_name,
                    version=version_name,
                    datamodule=dm,
                    max_epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE,
                )

                log_path = LOGGING_DIR / experiment_name / version_name / "metrics.csv"
                if log_path.exists():
                    df = pd.read_csv(log_path)

                    loss_df = df.dropna(subset=["epoch", "train_loss"])

                    if not loss_df.empty:
                        last_epoch = loss_df["epoch"].max()

                        last_5_epochs_df = loss_df[loss_df["epoch"] > last_epoch - 5]

                        if not last_5_epochs_df.empty:
                            avg_final_loss = last_5_epochs_df["train_loss"].mean()
                            adam_runs_results.append(
                                {"version": version_name, "metric": avg_final_loss}
                            )

            adam_runs_results.sort(key=lambda x: x["metric"])
            top_5_adam = adam_runs_results[:5]

            best_results[f"{net_name}_{dataset_name}"] = {
                "dogr_version": "dogr_tuned",
                "adam_top_5": top_5_adam,
            }
            print(
                f"\nNajlepsze 5 wersji Adama dla {net_name} na {dataset_name} (wg. końcowej straty):"
            )
            for run_result in top_5_adam:
                print(
                    f"  - {run_result['version']} (Śr. strata: {run_result['metric']:.6f})"
                )


def plot_final_results():
    """
    Generuje osobny wykres porównawczy dla każdej kombinacji sieć-dataset.
    """
    print("\n--- Generowanie wykresów porównawczych ---")
    plt.style.use("seaborn-v0_8-whitegrid")

    for result_key, result_data in best_results.items():
        fig, ax = plt.subplots(figsize=(12, 8))

        net_name, dataset_name = result_key.split("_", 1)
        experiment_name = f"dOGR_vs_Adam_{net_name}_{dataset_name}"

        fig.suptitle(
            f"Porównanie na: {dataset_name} | Architektura: {net_name} ({MAX_EPOCHS} epok)",
            fontsize=16,
        )

        dogr_version = result_data["dogr_version"]
        dogr_log_path = LOGGING_DIR / experiment_name / dogr_version / "metrics.csv"
        if dogr_log_path.exists():
            df_dogr = pd.read_csv(dogr_log_path)
            loss_dogr = (
                df_dogr.dropna(subset=["train_loss"])
                .groupby("epoch")["train_loss"]
                .mean()
            )
            ax.plot(
                loss_dogr.index,
                loss_dogr.values,
                "-",
                label=f"dOGR (dostrojony)",
                linewidth=2.5,
                color="red",
            )

        top_5_adam = result_data["adam_top_5"]
        for adam_run in top_5_adam:
            adam_version = adam_run["version"]
            adam_log_path = LOGGING_DIR / experiment_name / adam_version / "metrics.csv"
            if adam_log_path.exists():
                df_adam = pd.read_csv(adam_log_path)
                loss_adam = (
                    df_adam.dropna(subset=["train_loss"])
                    .groupby("epoch")["train_loss"]
                    .mean()
                )
                label = f"{adam_version} (śr. strata: {adam_run['metric']:.4f})"
                ax.plot(loss_adam.index, loss_adam.values, "-", label=label, alpha=0.7)

        ax.set_xlabel("Epoka")
        ax.set_ylabel("Średnia strata treningowa (skala log)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plots_dir = LOGGING_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        save_path = plots_dir / f"comparison_{result_key}.png"

        plt.savefig(save_path, dpi=300)
        print(f"Zapisano wykres w: {save_path}")

        plt.close(fig)


if __name__ == "__main__":
    run_experiments_on_all_architectures()
    plot_final_results()
