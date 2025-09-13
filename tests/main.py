print("In module products __package__, __name__ ==", __package__, __name__)
from argparse import ArgumentError, ArgumentParser
from torch import optim
import lightning as L
import lightning.pytorch.loggers as loggers
from .nets import get_FC, get_LeNet, get_very_mini_net
from .trainloop import TestLightningModule, fit_and_test
from .datamodule import MNISTDataModule, ZEROMNISTDataModule
from pathlib import Path

from src.optim.dOGR import dOGR
from src.optim.OGR import OGR

LOGGING_DIR = Path(__file__).parent.parent / "logs"


def run(
    net,
    optimizer,
    name,
    datamodule,
    version,
    max_epochs: int,
    batch_size: int,
):
    module = TestLightningModule(
        net,
        optimizer,
    )


    logger = loggers.CSVLogger(
        str(LOGGING_DIR),
        name=name,
        version=version,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=25,
        gradient_clip_val=1.0
    )

    fit_and_test(module, trainer, datamodule)


optimizer_dict = {
    "SGD": {
        "opt": optim.SGD,
        "args": {"lr": 1e-3},
    },
    "ADAM": {
        "opt": optim.Adam,
        "args": {},
    },
    "dOGR": {
        "opt": dOGR,
        "args": {},
    },
    "OGR": {
        "opt": OGR,
        "args": {},
    },
}

net_dict = {"FC": get_FC, "LeNet": get_LeNet, "very_mini": get_very_mini_net, }

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="Optimzer from list"
    )
    parser.add_argument("--net", type=str, default="FC", help="Net from the list")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=1, help="Max number of epochs"
    )

    parser.add_argument(
        "--version", type=int, default=None, help="Version of experiment"
    )
    parser.add_argument(
        "--name", type=str, default="default_name", help="Name of the experiment"
    )
    parser.add_argument(
        "--data", type=str, default="MNIST", help="Name the data"
    )

    args = parser.parse_args()

    net = net_dict[args.net]()
    optimizer = optimizer_dict[args.optimizer]["opt"](
        net.parameters(), **(optimizer_dict[args.optimizer]["args"])
    )
    
    datamodule = None
    if args.data == "MNIST": 
        datamodule = MNISTDataModule(batch_size=args.batch_size)
    elif args.data == "ZERO": 
        datamodule = ZEROMNISTDataModule(batch_size=args.batch_size)
    else: 
        raise ValueError("data needs to be one of MNIST ZERO")
    
    run(
        net,
        optimizer,
        datamodule=datamodule,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        version=args.version,
        name=args.name,
    )


if __name__ == "__main__":
    main()
