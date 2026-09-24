from src.environment import Environment3D
from src.path_planner import PathPlanner
import matplotlib.pyplot as plt
import numpy as np
a = Environment3D()
import pathlib


if(a.parse_map_file('Code/src/maps/map1.txt')):
    pass
else:
    print("Ahhhh wrong file name")

 #Test collision function
'''
for i in range(0, 10):
    plt.cla()
    ax = a.visualize_environment()
    start_point = a.generate_random_free_point()
    end_point = a.generate_random_free_point()
    var = a.is_line_collision_free(start_point, end_point)
    try:

        x_vals = []
        y_vals = []
        z_vals = []
        for point in var:
            x_vals.append(point[0])
            y_vals.append(point[1])
            z_vals.append(point[2])
        ax.scatter(x_vals, y_vals, z_vals, s = 10, color = 'black', marker = '.')
        
    except:
        ax.scatter(var[0], var[1], var[2], s = 200, color = 'black', marker = '*')
    ax.scatter(start_point[0], start_point[1], start_point[2], s=100, color= 'blue', marker = '*')
    ax.scatter(end_point[0], end_point[1], end_point[2], s = 100, color = 'red', marker = 'X')
    x_vals = [start_point[0], end_point[0]]
    y_vals = [start_point[1], end_point[1]]
    z_vals = [start_point[2], end_point[2]]
    ax.plot(x_vals, y_vals, z_vals)
    plt.show(block = False)
    b = input("go ahead?")
'''