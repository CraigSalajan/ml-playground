from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import pygame

from gyms.FlappyBird.game_logic import PLAYER_HEIGHT, FlappyBirdLogic, PIPE_WIDTH, PLAYER_WIDTH, PIPE_HEIGHT
from gyms.FlappyBird.renderer import FlappyBirdRenderer


def calculate_reward(h_dist, v_dist):
    survival_reward = 1.0

    # alignment_reward = 1.0 / (abs(v_dist) + 1)  # the +1 in the denominator prevents division by zero
    alignment_reward = 1.0 / (abs(v_dist) + 1) ** 2  # Scaled Inverse

    return survival_reward + alignment_reward


class FlappyBirdSimpleGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        self.action_space = gym.spaces.discrete.Discrete(2)
        self.observation_space = gym.spaces.box.Box(-np.inf, np.inf,
                                                    shape=(2,),
                                                    dtype=np.float32)
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

    def _get_observation(self):
        up_pipe = low_pipe = None
        h_dist = 0
        for up_pipe, low_pipe in zip(self._game.upper_pipes,
                                     self._game.lower_pipes):
            h_dist = (low_pipe["x"] + PIPE_WIDTH / 2
                      - (self._game.player_x - PLAYER_WIDTH / 2))
            h_dist += 3  # extra distance to compensate for the buggy hit-box
            if h_dist >= 0:
                break

        upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
        lower_pipe_y = low_pipe["y"]
        player_y = self._game.player_y

        v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y
                                                      + PLAYER_HEIGHT / 2)

        if self._normalize_obs:
            h_dist /= self._screen_size[0]
            v_dist /= self._screen_size[1]

        return np.array([
            h_dist,
            v_dist,
        ])

    def step(self, action):
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = calculate_reward(obs[0], obs[1])

        done = not alive

        info = {"score": self._game.score}

        return obs, reward, done, False, info

    def reset(self, **kwargs):
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                     pipe_gap_size=self._pipe_gap)
        if self._renderer is not None:
            self._renderer.game = self._game

        return self._get_observation()

    def render(self) -> None:
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game
            self._renderer.make_display()

        self._renderer.draw_surface(show_score=True)
        self._renderer.update_display()

    def close(self):
        if self._renderer is not None:
            pygame.display.quit()
            self._renderer = None

        super().close()
