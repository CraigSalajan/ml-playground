import wandb


class Reward:

    def __init__(self, config):
        self.config = config
        self.previous_food_distance = 0

    def _calculate_food_distance_reward(self, head, food):
        current_distance = abs(head.x - food.block.x) + abs(head.y - food.block.y)
        scaled_reward = 1 / (1 + current_distance)
        distance_diff = self.previous_food_distance - current_distance

        reward = scaled_reward if distance_diff > 0 else -scaled_reward

        self.previous_food_distance = current_distance

        return reward

    def _calculate_death_penalty(self, dead, body):
        if dead:
            return self.config.get('death_penalty') * (len(body) / 4)

        return 0

    def _calculate_gap_penalty(self, tiles):
        reward = 0

        for i in range(1, tiles.shape[0] - 1):
            for j in range(1, tiles.shape[1] - 1):
                if tiles[i, j] == 0 and tiles[i - 1, j] == 1 and tiles[i + 1, j] == 1 and tiles[i, j - 1] == 1 and \
                        tiles[i, j + 1] == 1:
                    reward -= 5  # Adjust pen

        return reward

    def _calculate_food_reward(self, ate_food):
        if ate_food:
            return self.config.get('food_reward')

        return 0

    def _calculate_self_collision_penalty(self, head, body):
        for block in body[:-1]:  # Excluding the tail, as it will move in the next step
            if head == block:
                return self.config.get('collision_penalty', -10)  # default penalty of -10
        return 0

    def calculate_reward(self, head, body, food, ate_food, tiles, dead):
        food_reward = self._calculate_food_reward(ate_food)
        food_distance_reward = self._calculate_food_distance_reward(head, food)
        gap_penalty = self._calculate_gap_penalty(tiles)
        death_penalty = self._calculate_death_penalty(dead, body)
        collision_penalty = self._calculate_self_collision_penalty(head, body)

        reward = food_reward + food_distance_reward + gap_penalty + death_penalty \
            + collision_penalty + self.config.get("living_bonus")

        # if wandb.run is not None:
        #     wandb.Table({
        #         'reward/food_reward': food_reward,
        #         'reward/food_distance_reward': food_distance_reward,
        #         'reward/death_penalty': death_penalty,
        #         'reward/gap_penalty': gap_penalty,
        #         'reward/total': reward
        #     })

        return reward
