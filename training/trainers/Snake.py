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
            "test": "test"
        }

    def train(self) -> None:
        env = self.get_env(SnakeGym)
        model = self._create_model(env)

        super().train(model)

    def _create_model(self, env):
        return PPO(
            "MlpPolicy",
            env,
            tensorboard_log=f"{self.tensorboard_logs}/{self.project_name}",
            device="cuda"
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