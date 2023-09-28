import os
import time

import numpy as np
from stable_baselines3 import PPO

from gyms.Snake.SnakeGym import SnakeGym
from training.core.BaseTrainer import BaseTrainer


class Snake(BaseTrainer):

    @property
    def project_name(self):
        return "Snake-PPO"

    @property
    def config(self):
        return {
            "death_penalty": -10,
            "dist_reward": 10,
            "ent_coef": 0.05,
            "food_reward": 15,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "living_bonus": -1,
            "max_step": "4096",
            "policy": "MlpPolicy",
            "vf_coef": 0.5
        }

    @property
    def parameters(self):
        return [
            "death_penalty",
            "dist_reward",
            "food_reward",
            "fps",
            "living_bonus",
            "max_step"
        ]

    def train(self) -> None:
        env = self.get_env(SnakeGym)
        model = self._create_model(env)

        super().train(model)

    def _create_model(self, env):
        return PPO(
            device="cuda",
            ent_coef=self.config.get("ent_coef"),
            env=env,
            gae_lambda=self.config.get("gae_lambda"),
            gamma=self.config.get("gamma"),
            learning_rate=self.config.get("learning_rate"),
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
            return (PPO.load(
                f"{self.model_save_path}/{self.project_name}/{run_id}/training_timesteps__{str(max_value)}_steps"),
                    max_value)
        except:
            return self._create_model(env), None

    def watch(self, run_id):
        fps = 30
        frame_time = 1.0 / fps
        env = SnakeGym( render_mode="human", max_step=1000000)
        model, iteration = self._get_model(run_id, env, None, None)

        while True:

            state = env.reset()
            done = False
            while not done:
                start_time = time.time()

                env.render()
                state = env.get_obs()
                action, _ = model.predict(state)
                next_state, reward, done, _, __ = env.step(action)
                state = next_state

                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = max(frame_time - elapsed_time, 0)
                time.sleep(sleep_time)

            model, iteration = self._get_model(run_id, env, model, iteration)

    def play(self):
        env = SnakeGym()
        env.play()