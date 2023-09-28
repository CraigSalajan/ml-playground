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


# Predefined types for certain args (can be extended)
ARG_TYPES = {
    'learning_rate': float,
    'epochs': int,
    'use_feature_x': bool
}


def cast_type(key: str, value: str):
    """
    Cast the string value to its appropriate type based on key.
    """
    if key in ARG_TYPES:
        return ARG_TYPES[key](value)
    else:
        # If no predefined type, try best-effort casting (this can be adjusted)
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            return value


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


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
        ctx: typer.Context,
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

    for key, value in zip(ctx.args[::2], ctx.args[1::2]):
        key = key.strip("--")
        config[key] = cast_type(key, value)

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
