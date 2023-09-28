import importlib

import typer

from typing import Optional
from training import __app_name__, __version__

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


def get_module(trainer, config):
    module = importlib.import_module("training.trainers")
    mclass = getattr(module, trainer)
    return mclass(config)

@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version and exit",
            callback=_version_callback,
            is_eager=True
        )
) -> None:
    return


@app.command()
def train(
        trainer: str = typer.Option(
            None,
            "--trainer",
            "-t",
            help="Trainer to use"
        ),
        wandb: Optional[bool] = typer.Option(
            False,
            "--wandb",
            "-w",
            help="Determines whether to use Weights & Biases. Will need a valid API Key in the WANDB_API_KEY Environment Variable to work"
        ),
        timesteps: Optional[int] = typer.Option(
            100000,
            "--timesteps",
            "-ts",
            help="Number of timesteps to train with"
        )
) -> None:
    config = {
        "wandb": wandb,
        "timesteps": timesteps
    }
    instance = get_module(trainer, config)
    instance.train()


@app.command()
def watch(
        trainer: str = typer.Option(
            None,
            "--trainer",
            "-t",
            help="Trainer to use"
        ),
        run_id: str = typer.Option(
            None,
            "--run-id",
            "-r",
            help="Run ID to watch"
        )
) -> None:
    instance = get_module(trainer, {})
    instance.watch(run_id)


@app.command()
def play(
        trainer: str = typer.Option(
            None,
            "--trainer",
            "-t",
            help="Trainer to use"
        ),
) -> None:
    instance = get_module(trainer, {})
    instance.play()