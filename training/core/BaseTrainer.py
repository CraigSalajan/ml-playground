import os
from abc import abstractmethod, ABC

import torch
import typer
import wandb
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
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

    @property
    @abstractmethod
    def training_algorithm(self):
        pass

    def __init__(self, config, gym):
        self.project_entity = "ml-playground"
        self.model_save_path = "./models"
        self.tensorboard_logs = "./logs"
        self.timesteps = config.get('timesteps')
        self.wandb = config.get('wandb')
        self.config.update(config)
        self.gym = gym

    def _filter_config(self):
        return {k: v for k, v in self.config.items() if k in self.parameters}

    def _create_model(self, env):
        # n_steps = self.config.get("max_step") * self.config.get("num_envs")
        return self.training_algorithm(
            # batch_size= n_steps // 10,
            batch_size=self.config.get("n_steps") * self.config.get("num_envs"),
            device="cuda",
            ent_coef=self.config.get("ent_coef"),
            env=env,
            gae_lambda=self.config.get("gae_lambda"),
            gamma=self.config.get("gamma"),
            learning_rate=self.config.get("learning_rate"),
            # n_steps=n_steps,
            n_steps=self.config.get("n_steps"),
            policy=self.config.get("policy"),
            tensorboard_log=f"{self.tensorboard_logs}/{self.project_name}",
            vf_coef=self.config.get("vf_coef")
        )

    def _get_model(self, run_id, env, current_model, current_iteration):
        filenames = os.listdir(f"{self.model_save_path}/{self.project_name}/{run_id}")

        max_value = max(int(filename.strip('training_timesteps__steps.zip')) for filename in filenames if
                        filename.endswith(".zip") and filename.startswith("training"))

        if max_value is None or max_value == current_iteration:
            return current_model, max_value

        print(f"Loading training model at episode {max_value}")
        try:
            return (self.training_algorithm.load(
                f"{self.model_save_path}/{self.project_name}/{run_id}/training_timesteps__{str(max_value)}_steps",
                    env=env,
                    custom_objects={
                        "batch_size": self.config.get("n_steps") * self.config.get("num_envs"),
                        "device": "cuda",
                        "ent_coef":  self.config.get("ent_coef"),
                        "gae_lambda": self.config.get("gae_lambda"),
                        "gamma": self.config.get("gamma"),
                        "learning_rate": self.config.get("learning_rate"),
                            # n_steps=n_steps,
                        "n_steps": self.config.get("n_steps"),
                        "policy": self.config.get("policy"),
                        "tensorboard_log": f"{self.tensorboard_logs}/{self.project_name}",
                        "vf_coef": self.config.get("vf_coef")
                    }
                ), max_value)
        except:
            return self._create_model(env), None

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

    def _create_env(self):
        def _init():
            return Monitor(self.gym(**self._filter_config()))

        return _init

    def get_env(self):
        envs = [self._create_env() for _ in range(self.config.get("num_envs"))]
        env = SubprocVecEnv(envs)

        return env

    def train(self) -> None:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
        else:
            print("CPU Training")

        run = self.wandb_init()
        env = self.get_env()

        if self.config.get("run_id") is not None:
            model, _ = self._get_model(self.config.get("run_id"), env, None, None)
        else:
            model = self._create_model(env)

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
