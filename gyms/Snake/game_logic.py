import math
import numpy as np
from gyms.Snake.utils import *


class Snake:
    def __init__(
        self,
        fps=60,
        max_step=500,
        init_length=4,
        food_reward=2.0,
        dist_reward=None,
        living_bonus=0.0,
        death_penalty=-1.0,
        width=40,
        height=40,
        block_size=20,
        background_color=Color.orange,
        food_color=Color.red,
        head_color=Color.purple,
        body_color=Color.blue,
    ) -> None:

        self.episode = 0
        self.fps = fps
        self.max_step = max_step
        self.init_length = min(init_length, width//2)
        self.food_reward = food_reward
        self.dist_reward = (
            width+height)//4 if dist_reward is None else dist_reward
        self.living_bonus = living_bonus
        self.death_penalty = death_penalty
        self.blocks_x = width
        self.blocks_y = height
        self.food_color = food_color
        self.head_color = head_color
        self.body_color = body_color
        self.background_color = background_color
        self.food = Food(self.blocks_x, self.blocks_y, food_color)
        Block.size = block_size

        self.map = None
        self.screen = None
        self.clock = None
        self.human_playing = False

    def init(self):
        self.episode += 1
        self.score = 0
        self.direction = 3
        self.current_step = 0
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)

    def close(self):
        pygame.quit()
        pygame.display.quit()
        self.screen = None
        self.clock = None

    def render(self):
        if self.screen is None:
            self.screen, self.clock = game_start(
                self.blocks_x*Block.size, self.blocks_y*Block.size)
        self.clock.tick(self.fps)
        update_screen(self.screen, self)
        handle_input()

    def get_surrounding_tiles(self, window_size=3):
        # We'll check a window around the snake's head
        tiles = np.zeros((window_size, window_size), dtype=int)

        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                x, y = self.head.x + i, self.head.y + j

                if x < 0 or x >= self.blocks_x or y < 0 or y >= self.blocks_y:
                    tiles[i + window_size // 2, j + window_size // 2] = -1  # Wall
                elif (x, y) in [(block.x, block.y) for block in self.body]:
                    tiles[i + window_size // 2, j + window_size // 2] = 1  # Snake's body
                elif x == self.food.block.x and y == self.food.block.y:
                    tiles[i + window_size // 2, j + window_size // 2] = 2  # Food
        return tiles

    def step(self, direction):
        if direction is None:
            direction = self.direction
        self.current_step += 1
        truncated = True if self.current_step == self.max_step else False
        (x, y) = (self.head.x, self.head.y)
        step = Direction.step(direction)
        if (direction == 0 or direction == 1) and (self.direction == 0 or self.direction == 1):
            step = Direction.step(self.direction)
        elif (direction == 2 or direction == 3) and (self.direction == 2 or self.direction == 3):
            step = Direction.step(self.direction)
        else:
            self.direction = direction

        prev_distance_to_food = abs(self.head.x - self.food.block.x) + abs(self.head.y - self.food.block.y)

        self.head.x += step[0]
        self.head.y += step[1]

        current_distance_to_food = abs(self.head.x - self.food.block.x) + abs(self.head.y - self.food.block.y)

        reward = self.living_bonus

        if current_distance_to_food < prev_distance_to_food:
            reward += self.food_reward / (current_distance_to_food + 1)
        else:
            reward += -(self.food_reward / (current_distance_to_food + 1))  # Adjust this as needed

        dead = False

        if self.head == self.food.block:
            self.score += 1
            self.grow(x, y)
            self.food.new_food(self.blocks)
            reward += self.food_reward
        else:
            self.move(x, y)
            for block in self.body:
                if self.head == block:
                    reward += (self.death_penalty * (len(self.body) / self.init_length)) * 2
                    dead = True
            if self.head.x >= self.blocks_x or self.head.x < 0 or self.head.y < 0 or self.head.y >= self.blocks_x:
                dead = True
                reward += self.death_penalty * (len(self.body) / self.init_length)

            # Get surrounding tiles
            tiles = self.get_surrounding_tiles(5)

            # Penalize for creating gaps
            for i in range(1, tiles.shape[0] - 1):
                for j in range(1, tiles.shape[1] - 1):
                    if tiles[i, j] == 0 and tiles[i - 1, j] == 1 and tiles[i + 1, j] == 1 and tiles[i, j - 1] == 1 and \
                            tiles[i, j + 1] == 1:
                        reward -= 5  # Adjust penalty as needed

        return self.observation(dead), reward, dead, truncated

    def get_board_state(self):
        board_state = np.zeros((self.blocks_x, self.blocks_y), dtype=np.int32)

        # Food
        fx, fy = self.food.block.x, self.food.block.y
        if 0 <= fx < self.blocks_x and 0 <= fy < self.blocks_y:
            board_state[fx][fy] = 2

        # Snake body
        for block in self.body:
            bx, by = block.x, block.y
            if 0 <= bx < self.blocks_x and 0 <= by < self.blocks_y:
                board_state[bx][by] = 1

        return board_state.flatten()

    def direction_to_vector(self, direction):
        # Define a mapping from directions to vectors
        direction_mapping = {
            0: (0, -1),  # Up
            1: (0, 1),  # Down
            2: (-1, 0),  # Left
            3: (1, 0)  # Right
        }

        return direction_mapping[direction]

    def observation(self, dead=False):
        dx = self.head.x - self.food.block.x
        dy = self.head.y - self.food.block.y

        # Normalize Manhattan distance
        max_distance = self.blocks_x + self.blocks_y  # max possible Manhattan distance
        distance_to_food = (abs(dx) + abs(dy)) / max_distance
        d1, d2 = self.direction_to_vector(int(self.direction))

        # Board state
        board_state = self.get_board_state()

        # Concatenate all observations
        return np.concatenate([board_state, [distance_to_food, d1, d2]])

    def calc_distance(self, dead):
        if dead:
            return 0, 0, 0, 0
        self.map = np.zeros((self.blocks_x, self.blocks_y), dtype=int)
        for block in self.blocks:
            self.map[block.x][block.y] = -1
        self.map[self.food.block.x][self.food.block.y] = 0
        d0, d1, d2, d3 = 0, 0, 0, 0,
        x, y = self.head.x, self.head.y - 1
        while y >= 0 and self.map[x][y] == 0:
            d0 += 1
            y -= 1
        x, y = self.head.x, self.head.y + 1
        while y < self.blocks_y and self.map[x][y] == 0:
            d1 += 1
            y += 1
        x, y = self.head.x - 1, self.head.y
        while x >= 0 and self.map[x][y] == 0:
            d2 += 1
            x -= 1
        x, y = self.head.x + 1, self.head.y
        while x < self.blocks_x and self.map[x][y] == 0:
            d3 += 1
            x += 1
        self.map[self.food.block.x][self.food.block.y] = 1
        return d0/self.blocks_y, d1/self.blocks_y, d2/self.blocks_x, d3/self.blocks_x

    def calc_reward(self):
        if self.dist_reward == 0.0:
            return 0
        x = self.head.x - self.food.block.x
        y = self.head.y - self.food.block.y
        d = math.sqrt(x*x + y*y)
        return (self.dist_reward-d)/self.dist_reward

    def grow(self, x, y):
        body = Block(x, y, Color.blue)
        self.blocks.append(body)
        self.body.append(body)

    def move(self, x, y):
        tail = self.body.pop(0)
        tail.move_to(x, y)
        self.body.append(tail)

    def info(self):
        return {
            'head': (self.head.x, self.head.y),
            'food': (self.food.block.x, self.food.block.y),
            # 'map': self.map.T
        }

    def play(self, fps=10, acceleration=True, step=1, frep=10):
        self.max_step = 99999
        self.fps = fps
        self.food_reward = 1
        self.living_bonus = 0
        self.dist_reward = 0
        self.death_penalty = 0
        self.human_playing = True
        self.init()
        screen, clock = game_start(
            self.blocks_x*Block.size, self.blocks_y*Block.size)
        total_r = 0

        while pygame.get_init():
            clock.tick(self.fps)
            _, r, d, _ = self.step(handle_input())
            total_r += r
            if acceleration and total_r == frep:
                self.fps += step
                total_r = 0
            if d:
                self.init()
                total_r = 0
                self.fps = fps
            update_screen(screen, self, True)