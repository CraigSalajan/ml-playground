import time

from stable_baselines3 import PPO

from gyms.Snake.SnakeGym import SnakeGym
from training.core.BaseTrainer import BaseTrainer


class Snake(BaseTrainer):

    def __init__(self, config):
        super().__init__(config, SnakeGym)

    @property
    def project_name(self):
        return "Snake-PPO"

    @property
    def training_algorithm(self):
        return PPO

    @property
    def config(self):
        return {
            "death_penalty": -10,
            "dist_reward": 10,
            "ent_coef": 0.02,
            "food_reward": 25,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1.5e-4,
            "living_bonus": -0.1,
            "max_step": 4096,
            "num_envs": 10,
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

    def watch(self, run_id):
        fps = 30
        frame_time = 1.0 / fps
        env = SnakeGym(render_mode="human", max_step=self.config.get("max_step"))
        model, iteration = self._get_model(run_id, env, None, None)

        while True:

            env.reset()
            done = False
            while not done:
                start_time = time.time()

                env.render()
                state = env.get_obs()
                action, _ = model.predict(state)
                next_state, reward, done, _, __ = env.step(action)

                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = max(frame_time - elapsed_time, 0)
                time.sleep(sleep_time)

            model, iteration = self._get_model(run_id, env, model, iteration)

    def play(self):
        env = SnakeGym()
        env.play()
