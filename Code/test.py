from src.environment import Environment3D
a = Environment3D()
a.parse_map_file('src/maps/map4.txt')
a.visualize_environment(show_start_goal=True)