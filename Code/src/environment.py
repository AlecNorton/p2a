import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import math
class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        self.start_point = [7.954360487979886, 6.822833826909669, 1.058209137433761]
        self.goal_point = [44.304797815557095, 29.328280798754054, 4.454834705539382]
        self.safety_margin = 0.5  # Safety margin around obstacles



    ###############################################
    ##### TODO - Implement map file parsing ####### 
    ###############################################    
    def parse_map_file(self, filename):
        """
        Parse the map file and extract boundary and blocks
        coords = [xmin, ymin, zmin, xmax, ymax, zmax]
        colors = [r, g, b] each in [0, 1] (make sure color values are in range 0-1)
        self.blocks.append((coords, colors))
        self.boundary = [xmin, ymin, zmin, xmax, ymax, zmax]
        return True if successful, False otherwise (True if file was parsed successfully, without any error.)
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    res = re.split(r"[\s]+", line)
                    print(res)
                    if('boundary' in res[0]):
                        print("Found.")
                        res = list(map(lambda x: float(x.replace('\n', '')), res[1:7]))
                        self.boundary = res
                    elif('block' in res[0]):
                        print("Found block.")
                        coords = list(map(lambda x: float(x.replace('\n', '')), res[1:7]))
                        colors = list(map(lambda x: float(x.replace('\n', '')), res[7:10]))
                        self.blocks.append(tuple((coords, colors)))
            return True
        except:
            return False
    
            


    ##############################################
    #### TODO - Implement collision checking #####
    ##############################################
    def is_point_in_free_space(self, point):
        """
        Check if a point is in free space (not inside any obstacle)
        Complete implementation with collision checking
        return True if free, False if in collision
        """
        #Check x range, y range, and z range using nested if statements.
        pointX, pointY, pointZ = point
        if(self.is_point_in_boundary(point) == False):
            return False
        for block_and_color in self.blocks:
            xmin, ymin, zmin, xmax, ymax, zmax = block_and_color[0]
            if (pointX >= xmin-self.safety_margin and pointX <= xmax+self.safety_margin):
                #Within range of X
                if(pointY >= ymin-self.safety_margin and pointX <= ymax+self.safety_margin):
                    #Within rangeo f Y
                    if(pointZ >= zmin-self.safety_margin and pointZ <= zmax+self.safety_margin):
                        #Within range of Z
                        return False
        return True

    def is_point_in_boundary(self, point):
        """
        True if point is within boundary plus some safety_margin.
        False if point is beyond bondary. """
        pointX, pointY, pointZ = point
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        if(pointX >= xmin+self.safety_margin and pointX <=xmax-self.safety_margin):
            if(pointY >= ymin+self.safety_margin and pointY <=ymax-self.safety_margin):
                if(pointZ >= zmin+self.safety_margin and pointZ <=zmax-self.safety_margin):
                    return True
        return False
            
    


    ##############################################
    #### TODO - Implement line - collision checking #####
    ##############################################
    def is_line_collision_free(self, p1, p2, num_checks=20):
        """
        Check if a line segment between two points is collision-free
        Used for RRT* edge validation
        return True if free, False if in collision
        """
        #First simply check two points.
        if(self.is_point_in_free_space(p1) == False or self.is_point_in_free_space(p2) == False):
            return False
        
        else:
            distance = math.dist(p1, p1)
            numPoints = distance / .001 #Make a point along each mm of the line. 
        

    
    def generate_random_free_point(self):
        """
        Generate a random point in free space
        Used for RRT* sampling
        """
        if not self.boundary:
            return None
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(xmin + self.safety_margin, xmax - self.safety_margin)
            y = np.random.uniform(ymin + self.safety_margin, ymax - self.safety_margin)
            z = np.random.uniform(zmin + self.safety_margin, zmax - self.safety_margin)
            
            point = [x, y, z]
            if self.is_point_in_free_space(point):
                return point
        
        print("Warning: Could not generate random free point after", max_attempts, "attempts")
        return None

    def get_environment_info(self):
        """Get information about the environment layout"""
        if not self.boundary:
            return "No boundary defined"
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        info = f"""
        Environment Information:
        Boundary: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]
        Size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} meters
        Volume: {(xmax-xmin)*(ymax-ymin)*(zmax-zmin):.1f} cubic meters
        Obstacles: {len(self.blocks)} blocks
        Safety margin: {self.safety_margin} meters
        """
        
        if self.start_point and self.goal_point:
            distance = np.linalg.norm(np.array(self.goal_point) - np.array(self.start_point))
            info += f"  Start-Goal distance: {distance:.2f} meters\n"
        
        return info
