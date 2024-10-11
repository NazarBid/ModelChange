"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import mesa
import numpy as np


class Boid(mesa.Agent):
    def __init__(
            self,
            unique_id,
            model,
            speed,
            direction,
            vision,
            separation,
            max_speed=1,
            noise_factor=0.1,
            crowd_radius=5,
            cohesion_radius=10,
            avoid_radius=5,
    ):
        super().__init__(unique_id, model)
        self.speed = speed
        self.direction = direction
        self.vision = vision
        self.separation = separation
        self.max_speed = max_speed
        self.noise_factor = noise_factor
        self.crowd_radius = crowd_radius
        self.cohesion_radius = cohesion_radius
        self.avoid_radius = avoid_radius
        self.neighbors = None

    def step(self):
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)

        align_vector = self.get_average_direction()

        cohesion_vector = self.get_cohesion()
        #додав шум, щоб агенти рухались не по одному і тому ж маршруту, а більш "живо"
        noise_vector = np.random.rand(2) * 2 - 1

        move_vector = align_vector + cohesion_vector + noise_vector * self.noise_factor

        #обмеження швидкості, щоб навіть при великих параметрах швидкості агенти поводили себе адекватно
        if np.linalg.norm(move_vector) > self.max_speed:
            move_vector = (move_vector / np.linalg.norm(move_vector)) * self.max_speed

        new_pos = self.pos + move_vector * self.speed
        self.model.space.move_agent(self, new_pos)

    #метод для обчислення середнього напрямку групи агентів, щоб рух був більш організований
    def get_average_direction(self):
        sum_vector = np.zeros(2)
        count = 0
        for neighbor in self.neighbors:
            if neighbor == self:
                continue
            distance = np.linalg.norm(self.pos - neighbor.pos)
            if distance < self.cohesion_radius:
                sum_vector += neighbor.direction
                count += 1
        if count > 0:
            return sum_vector / count
        return np.zeros(2)

    #метод для обчислення середньої позиції сусідів, щоб вони тримались більш згуртовано
    def get_cohesion(self):
        sum_vector = np.zeros(2)
        count = 0
        for neighbor in self.neighbors:
            if neighbor == self:
                continue
            distance = np.linalg.norm(self.pos - neighbor.pos)
            if distance < self.avoid_radius:
                sum_vector += neighbor.pos
                count += 1
        if count > 0:
            center_of_mass = sum_vector / count
            return center_of_mass - self.pos
        return np.zeros(2)


class BoidFlockers(mesa.Model):
    """
    Flocker model class. Handles agent creation, placement, and scheduling.
    """

    def __init__(
            self,
            seed=None,
            population=100,
            width=100,
            height=100,
            vision=10,
            speed=1,
            separation=1,
            cohere=0.03,
            separate=0.015,
            match=0.05,
    ):
        """
        Create a new Flockers model.
        """
        super().__init__(seed=seed)
        self.population = population
        self.vision = vision
        self.speed = speed
        self.separation = separation

        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.factors = {"cohere": cohere, "separate": separate, "match": match}

        # Use RandomActivation scheduler
        self.schedule = mesa.time.RandomActivation(self)

        self.make_agents()

    def make_agents(self):
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            direction = np.random.random(2) * 2 - 1
            boid = Boid(
                model=self,
                unique_id=i,
                speed=self.speed,
                direction=direction,
                vision=self.vision,
                separation=self.separation,
                max_speed=1,
                noise_factor=0.1,
                crowd_radius=5,
                cohesion_radius=10,
                avoid_radius=5,
            )
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
       self.schedule.step()
