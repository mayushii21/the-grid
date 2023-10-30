import math
from collections import deque

import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class CityGrid:
    def __init__(self, n, m, obstruction_cov=0.3, seed=None):
        self.n = n
        self.m = m
        self.grid = np.zeros(
            (m, n), dtype=int
        )  # 0 represents unobstructed, 1 represents obstructed blocks
        self.rng = np.random.default_rng(seed)
        self._place_obstructions(obstruction_cov)
        self.orig_grid = self.grid.copy()
        self.towers = set()
        self.tower_graph = {}
        self.optimal_paths = {}

    def _place_obstructions(self, coverage):
        # Calculate the number of needed obstructions to ensure strictly greater than the specified coverage
        n_obstructions = math.ceil(self.n * self.m * coverage)
        # Generate indices for obstructed blocks, ensuring no duplicate positions
        obstructed_idx = self.rng.choice(
            self.n * self.m, size=n_obstructions, replace=False
        )
        # Assign 1 (for obstructed blocks) to indices
        self.grid.reshape(-1)[obstructed_idx] = 1

    def place_tower(self, x, y, tower_range, display=False):
        # Check if the specified location is unobstructed
        if self.grid[y, x] == 0 or self.grid[y, x] == 2:
            # Place a tower with the given range
            # use max(0, idx) to prevent reversal
            temp = self.grid[
                max(0, y - tower_range) : y + tower_range + 1,
                max(0, x - tower_range) : x + tower_range + 1,
            ]
            self.grid[
                max(0, y - tower_range) : y + tower_range + 1,
                max(0, x - tower_range) : x + tower_range + 1,
            ] = np.where(
                temp < 2, temp + 2, temp
            )  # 2 and 3 represent tower coverage for non-obstructed and obstructed blocks respectively
            # Place the tower with value 4
            self.grid[y, x] = 4
            self.towers.add((x, y))
            # Plot coverage of placed tower if specified
            if display:
                temp_grid = self.orig_grid.copy()
                temp_grid[
                    max(0, y - tower_range) : y + tower_range + 1,
                    max(0, x - tower_range) : x + tower_range + 1,
                ] += 2
                self.visualize_grid(temp_grid)
        else:
            print("Tower placement is obstructed.")

    def _generate_sets(self, tower_range):
        # Generate a list of sets that contain coordinates for all possible towers that can cover an empty block
        sets_of_towers = []
        # Iterate through all possible empty blocks
        for row in range(0, self.grid.shape[0]):
            for col in range(0, self.grid.shape[1]):
                if self.grid[row, col] != 0:
                    continue
                covering_towers = set()
                # Iterate through possible tower placements to cover the block
                for range_row in range(
                    max(0, row - tower_range),
                    min(self.grid.shape[0], row + tower_range + 1),
                ):
                    for range_col in range(
                        max(0, col - tower_range),
                        min(self.grid.shape[1], col + tower_range + 1),
                    ):
                        if self.grid[range_row, range_col] != 0:
                            continue
                        # Add towers to set
                        covering_towers.add((range_col, range_row))
                # Add sets to list
                sets_of_towers.append(covering_towers)
        return sets_of_towers

    def optimize_tower_placement(self, tower_range):
        def _most_frequent(list_of_sets):
            # Return the most frequent tower coordinates
            return max(set(list_of_sets), key=list_of_sets.count)

        set_list = self._generate_sets(tower_range)

        # Repeat until all empty blocks are covered
        while set_list:
            # Store all towers of sets into a list
            towers = [el for s in set_list for el in s]
            # towers = []
            # for s in set_list:
            #     for el in s:
            #         towers.append(el)
            # Obtain the most frequent one
            tower = _most_frequent(towers)
            # Place the most frequent tower
            self.place_tower(*tower, tower_range)

            # Remove sets containing the placed tower from the list
            set_list = [s for s in set_list if tower not in s]

    def _plot_path(self, path):
        # Create a list of x, y values for a given path
        x = []
        y = []
        for x_point, y_point in self.optimal_paths[frozenset(path)]:
            x.append(x_point + 0.5)
            y.append(y_point + 0.5)
        # Set visible color cycle
        current_clrs = list(mcolors.TABLEAU_COLORS.keys())
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=current_clrs[1:2] + current_clrs[3:]
        )
        plt.plot(x, y, linewidth=4, alpha=0.9)

    def visualize_grid(self, grid=None, paths=False):
        if grid is None:
            grid = self.grid
        nu = np.unique(self.grid)
        # Define custom colormap
        all_col = ["grey", "tab:red", "green", "darkgreen", "black"]
        colors = all_col[nu[0] : nu[-1] + 1]
        cmap = ListedColormap(colors)
        # Plot heatmap
        ax = sns.heatmap(self.grid, cmap=cmap, cbar=True, linewidths=0.5, square=True)
        ax.invert_yaxis()
        # Specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(np.linspace(nu[0], nu[-1], len(nu) * 2 + 1)[1::2])
        colorbar.set_ticklabels(
            [
                "non-obstr w/o cov",
                "obstructed",
                "within coverage",
                "obstructed w/ cov",
                "tower",
            ][nu[0] : nu[-1] + 1]
        )

        # Plot paths
        if paths == "all" and self.optimal_paths:
            for path in self.optimal_paths:
                self._plot_path(path)
        elif paths and frozenset(paths) in self.optimal_paths:
            self._plot_path(paths)

        plt.tight_layout()
        plt.show()

    def visualize_obstructions(self):
        # Filter to only show obstructions
        obstructions = np.where((self.grid == 1) | (self.grid == 3), 1, 0)
        nu = np.unique(obstructions)
        # Define custom colormap
        cmap = ListedColormap(["grey", "black"])
        # Plot heatmap
        ax = sns.heatmap(
            obstructions, cmap=cmap, cbar=True, linewidths=0.5, square=True
        )
        ax.invert_yaxis()
        # Specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(np.linspace(nu[0], nu[-1], len(nu) * 2 + 1)[1::2])
        colorbar.set_ticklabels(["non-obstr", "obstructed"][nu[0] : nu[-1] + 1])

        plt.tight_layout()
        plt.show()

    def visualize_towers(self):
        # Filter to only show towers
        towers = np.where(self.grid == 4, 1, 0)
        nu = np.unique(towers)
        # Define custom colormap
        cmap = ListedColormap(["grey", "black"])
        # Plot heatmap
        ax = sns.heatmap(towers, cmap=cmap, cbar=True, linewidths=0.5, square=True)
        ax.invert_yaxis()
        # Specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(np.linspace(nu[0], nu[-1], len(nu) * 2 + 1)[1::2])
        colorbar.set_ticklabels(["blocks", "towers"][nu[0] : nu[-1] + 1])

        plt.tight_layout()
        plt.show()

    def _generate_tower_graph(self, tower_range, grid):
        # Iterate through all placed towers
        for tower in self.towers:
            self.tower_graph[tower] = []
            # Iterate through all blocks within range
            for range_row in range(
                max(0, tower[1] - tower_range),
                min(grid.shape[0], tower[1] + tower_range + 1),
            ):
                for range_col in range(
                    max(0, tower[0] - tower_range),
                    min(grid.shape[1], tower[0] + tower_range + 1),
                ):
                    # Add blocks with towers in range to list
                    if grid[range_row, range_col] == 4 and (tower[0], tower[1]) != (
                        range_col,
                        range_row,
                    ):
                        self.tower_graph[tower].append((range_col, range_row))

    def most_reliable_path(
        self, start_tower, end_tower, tower_range, recalculate=False
    ):
        # Create a graph representation of towers and their connections
        if not self.tower_graph or recalculate:
            self._generate_tower_graph(tower_range, self.grid)

        # Perform BFS to find the shortest path and store the shortest path for all paths that start (or end) with the start_tower
        if frozenset([start_tower, end_tower]) not in self.optimal_paths or recalculate:
            visited = set([start_tower])
            queue = deque(
                [(start_tower, [start_tower])]
            )  # double-ended queue containing the node and the path between towers

            while queue:
                current_node, path = queue.popleft()
                # Iterate over neighboring towers of current node
                for n in self.tower_graph[current_node]:
                    if n not in visited:
                        visited.add(n)
                        queue.append([n, path + [n]])
                        # Add intermediate steps for future access
                        if frozenset([n, start_tower]) not in self.optimal_paths:
                            self.optimal_paths[frozenset([n, start_tower])] = path + [n]

        if frozenset([start_tower, end_tower]) in self.optimal_paths:
            print(
                f"Most reliable path: {self.optimal_paths[frozenset([start_tower, end_tower])]}"
            )
        else:
            print("No path found.")
