import random

import matplotlib.pyplot as plt


weight = 0.8
c1 = 1.5
c2 = 1.5

iterations = 100
particles = 30
error = 0.001


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.pbest_x = x
        self.pbest_y = y
        self.pbest_error = self.error()

    def error(self):
        return (self.x - weight) ** 2 + (self.y - weight) ** 2

    def update_pbest(self):
        if self.error() < self.pbest_error:
            self.pbest_x = self.x
            self.pbest_y = self.y
            self.pbest_error = self.error()

    def update_velocity(self, gbest_x: float, gbest_y: float):
        self.vx = c1 * random.random() * (self.pbest_x - self.x) + c2 * \
            random.random() * (gbest_x - self.x)
        self.vy = c1 * random.random() * (self.pbest_y - self.y) + c2 * \
            random.random() * (gbest_y - self.y)

    def update_position(self):
        self.x += self.vx
        self.y += self.vy


class Space:
    def __init__(self, particles: list[Particle]):
        self.particles = particles
        self.gbest_x = 0
        self.gbest_y = 0
        self.gbest_error = 1e10
        self.update_gbest()

    def update_gbest(self):
        for particle in self.particles:
            particle.update_pbest()
            if particle.pbest_error < self.gbest_error:
                self.gbest_x = particle.pbest_x
                self.gbest_y = particle.pbest_y
                self.gbest_error = particle.pbest_error

    def update_velocity(self):
        for particle in self.particles:
            particle.update_velocity(self.gbest_x, self.gbest_y)

    def update_position(self):
        for particle in self.particles:
            particle.update_position()

    def plot(self):
        x = [particle.x for particle in self.particles]
        y = [particle.y for particle in self.particles]
        plt.plot(x, y)
        plt.plot([weight, weight], [0, self.gbest_y], 'r')
        plt.plot([0, self.gbest_x], [weight, weight], 'r')
        plt.show()

    def run(self):
        for _ in range(iterations):
            self.update_velocity()
            self.update_position()
            self.update_gbest()
            if self.gbest_error < error:
                break
        self.plot()


if __name__ == "__main__":
    space = Space([Particle(random.random(), random.random(), 0, 0)
                  for _ in range(particles)])
    space.run()
    print(space.gbest_x, space.gbest_y, space.gbest_error)
