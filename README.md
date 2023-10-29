# CityGrid

### View examples [here](/src/city_network/results.ipynb).

### View source code [here](/src/city_network/citygrid.py).

<p>&nbsp;</p>

Import the class with `from city_network.citygrid import CityGrid`.
Create an instance of the CityGrid class with `instance_var = CityGrid(n, m, obstruction_cov=0.3, seed=None)`

> Parameters:  
    n: Number of columns in the grid.  
    m: Number of rows in the grid.  
    obstruction_cov: Coverage of obstructions (default is 0.3).  
    seed: Seed for random number generation of obstructions (default is None).

<p>&nbsp;</p>

#### Class methods:

    place_tower(x, y, tower_range, display=False)

Places a tower at a specified location with a given range. Set `display` to `True` to display the tower's coverage.

> Parameters:  
    x: X-coordinate of the tower.  
    y: Y-coordinate of the tower.  
    tower_range: Range of the tower's coverage.  
    display: Whether to display the tower's coverage (default is False).

<p>&nbsp;</p>

    optimize_tower_placement(tower_range)

Runs algorithm to optimize tower placement to cover all empty blocks.

> Parameters:  
    tower_range: Range of tower coverage.

<p>&nbsp;</p>

    most_reliable_path(start_tower, end_tower, tower_range, recalculate=False)

Finds the most reliable path between two towers.

> Parameters:  
    start_tower: Starting tower (tuple containing x, y coordinates).  
    end_tower: Ending tower (tuple containing x, y coordinates).  
    tower_range: Range of tower coverage.  
    recalculate: Whether to recalculate the path (default is False).

<p>&nbsp;</p>

    visualize_grid(paths=False)

Visualizes the grid, including non-obstructed blocks, obstructed blocks, towers, and coverage, with  the option to display optimal data paths.

> Parameters:  
    paths: Either "all" to display all optimal paths previously added with `most_reliable_path`, or a specific path to display (default is False).

<p>&nbsp;</p>

    visualize_obstructions()

Visualizes the grid with only obstructed blocks.

<p>&nbsp;</p>

    visualize_towers()

Visualizes the grid with only towers.
