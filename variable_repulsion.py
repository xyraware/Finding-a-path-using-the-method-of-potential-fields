import numpy as np
import matplotlib.pyplot as plt


class PotentialFieldRobot:
    def __init__(self, width, height, start, goal, obstacles, k_att=1.0, k_rep=100.0, influence_radius=1.5,
                 step_size=0.1, max_steps=1000, k_rep_scale=10.0):
        self.width = width
        self.height = height
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = [np.array(obs, dtype=float) for obs in obstacles]

        self.k_att = k_att
        self.k_rep = k_rep
        self.rho0 = influence_radius
        self.step_size = step_size
        self.max_steps = max_steps
        self.k_rep_scale = k_rep_scale
        self.k_rep_scale = k_rep_scale

        self.current_pos = self.start.copy()
        self.path = [self.start.copy()]

    def compute_attractive_force(self):
        return self.k_att * (self.goal - self.current_pos)

    def compute_repulsive_force(self):
        total_rep_force = np.zeros(2, dtype=float)
        for obs in self.obstacles:
            dist_vec = self.current_pos - obs
            dist_mag = np.linalg.norm(dist_vec)

            if self.rho0 >= dist_mag > 0:
                distance_factor = (self.rho0 / dist_mag) ** self.k_rep_scale
                rep_force = self.k_rep * distance_factor * (1.0 / dist_mag - 1.0 / self.rho0) * (1.0 / dist_mag ** 2) * (
                            dist_vec / dist_mag)
                total_rep_force += rep_force

        return total_rep_force

    def find_path_and_animate(self):
        print("Starting pathfinding with dynamic visualization...")

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

        for i in range(self.max_steps):
            if np.linalg.norm(self.current_pos - self.goal) < self.step_size:
                print(f"Goal reached in {i} steps!")
                self.path.append(self.goal.copy())
                break

            f_att = self.compute_attractive_force()
            f_rep = self.compute_repulsive_force()
            f_total = f_att + f_rep

            force_mag = np.linalg.norm(f_total)
            if force_mag > 0:
                direction = f_total / force_mag
                self.current_pos += self.step_size * direction
                self.path.append(self.current_pos.copy())
            else:
                print("Stuck in a local minimum.")
                break

            ax.clear()

            path_arr = np.array(self.path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], 'g-', linewidth=2, label='Robot Path')
            ax.plot(self.start[0], self.start[1], 'bo', markersize=12, label='Start')
            ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')

            if self.obstacles:
                obs_arr = np.array(self.obstacles)
                ax.plot(obs_arr[:, 0], obs_arr[:, 1], 'ks', markersize=10, label='Obstacles')

            ax.plot(self.current_pos[0], self.current_pos[1], 'go', markersize=10, label='Robot')

            ax.set_title(f'Potential Field Path Planning (Step {i})')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_xlim(-1, self.width + 1)
            ax.set_ylim(-1, self.height + 1)
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')

            plt.pause(0.01)

        if i == self.max_steps - 1:
            print(f"Pathfinding stopped after reaching max steps ({self.max_steps}).")

        print("Animatio n finished. Close the plot window to exit.")
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    WIDTH = 10
    HEIGHT = 10
    START_POS = (0, 0)
    GOAL_POS = (9, 9)
    OBSTACLES = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 5)]

    robot = PotentialFieldRobot(
        width=WIDTH,
        height=HEIGHT,
        start=START_POS,
        goal=GOAL_POS,
        obstacles=OBSTACLES,
        k_rep=150.0,
        influence_radius=2,
        k_rep_scale=10.0
    )

    robot.find_path_and_animate()