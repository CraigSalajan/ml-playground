import time

import numpy as np
from stable_baselines3 import PPO

from gyms.FlappyBird.FlappyBirdSimpleGym import FlappyBirdSimpleGym
from training.core.BaseTrainer import BaseTrainer


class FlappyBird(BaseTrainer):

    def __init__(self, config):
        super().__init__(config, FlappyBirdSimpleGym)

    @property
    def parameters(self):
        return []

    @property
    def project_name(self):
        return "Flappy-Bird-PPO"

    @property
    def training_algorithm(self):
        return PPO

    @property
    def config(self):
        return {
            "ent_coef": 0.02,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1.5e-4,
            "policy": "MlpPolicy",
            "vf_coef": 0.5
        }

    def watch(self, run_id):
        fps = 30
        frame_time = 1.0 / fps
        env = FlappyBirdSimpleGym()
        model, iteration = self._get_model(run_id, env, None, None)

        while True:

            state = env.reset()
            done = False
            while not done:
                start_time = time.time()

                env.render()
                state = np.expand_dims(state, axis=0)
                action, _ = model.predict(state)
                next_state, reward, done, _, __ = env.step(action)
                state = next_state

                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = max(frame_time - elapsed_time, 0)
                time.sleep(sleep_time)

            model, iteration = self._get_model(run_id, env, model, iteration)

    def play(self):
        pass



