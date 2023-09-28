import os
import random
import string

import typer
import wandb

from abc import abstractmethod, ABC

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from training.core.Run import Run


class BaseTrainer(ABC):

    @property
    @abstractmethod
    def project_name(self):
        pass

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    def __init__(self, config):
        self.project_entity = "ml-playground"
        self.model_save_path = "./models"
        self.tensorboard_logs = "./logs"
        self.timesteps = config.get('timesteps')
        self.wandb = config.get('wandb')
        self.config.update(config)

    def _filter_config(self):
        return {k: v for k, v in self.config.items() if k in self.parameters}

    def wandb_init(self) -> wandb.sdk.wandb_run.Run | Run:
        if self.wandb:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            return wandb.init(
                project=self.project_name,
                entity=self.project_entity,
                monitor_gym=True,
                sync_tensorboard=True,
                config=self.config
            )

        return Run()

    def get_env(self, gym_env):
        env = gym_env(**self._filter_config())
        env = Monitor(env)

        return env

    @abstractmethod
    def train(self, model) -> None:
        run = self.wandb_init()

        typer.echo(f"Starting run {run.id}")

        callbacks = []
        checkpoint = CheckpointCallback(
            save_freq=50000,
            save_path=f"{self.model_save_path}/{self.project_name}/{run.id}",
            name_prefix="training_timesteps_",
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint)

        if self.wandb:
            wandb_callback = WandbCallback(
                gradient_save_freq=50000,
                model_save_path=f"{self.model_save_path}/{self.project_name}/{run.id}",
                verbose=2,
            )
            callbacks.append(wandb_callback)

        model.learn(
            total_timesteps=self.timesteps,
            callback=callbacks
        )

        model.save(f"{self.model_save_path}/{self.project_name}/{run.id}/final.pt")

        if self.wandb:
            wandb.finish()

    @abstractmethod
    def watch(self, run_id):
        pass

    @abstractmethod
    def play(self):
        pass
