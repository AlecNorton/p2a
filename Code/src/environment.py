import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import math
import numpy as np
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
                    if('boundary' in res[0]):
                        res = list(map(lambda x: float(x.replace('\n', '')), res[1:7]))
                        self.boundary = res
                    elif('block' in res[0]):
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
            p1x, p1y, p1z = p1
            p2x, p2y, p2z = p2
            dist_vec = np.abs(np.subtract(p2, p1))
            xDir = 1
            yDir = 1
            zDir = 1
            if(p1x >= p2x):
                xDir = -1
            if(p1y >= p2y):
                yDir = -1
            if(p1z >= p2z):
                zDir = -1
            #num checks per one meter, i.e. 20 checks is a point every 5 cm. 
            
            num_cores = 5
            x_core, y_core, z_core = np.divide(dist_vec, num_cores)
            x_check, y_check, z_check = np.divide(dist_vec, num_checks)

            checkingPoint = p1
            for i in range(num_checks):
                checkingPoint = np.add(checkingPoint, [checkingPoint[0]+(xDir*x_check*i), checkingPoint[1]+(yDir*y_check*i), checkingPoint[2]+(zDir*z_check*i)])
                for j in range(num_cores):
                    core_point = [checkingPoint[0]+(xDir*x_core*j), checkingPoint[1]+(yDir*y_core*j), checkingPoint[2]+(zDir*z_core*j)]
                    if(self.is_point_in_free_space(core_point) == False):
                        return False
            return True
                            

    def visualize_environment(self, ax=None, show_start_goal = False):
        """Visualize the environment"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False

        verts = np.array([])
        colors = []
        for block_color in self.blocks:
            coords = block_color[0]
            color = np.divide(block_color[1], 255)
            print(f"Color: {color}")
            if(len(verts) == 0):
                verts = self.list_of_coords(coords)
            else:
                verts = np.append(verts, self.list_of_coords(coords), axis = 0)
            for i in range(6): #six sides
                colors.append(color)
        if (show_start_goal):
            ax.scatter(self.start_point[0], self.start_point[1], self.start_point[2], s=100, color= 'red', marker = '*')
            ax.scatter(self.goal_point[0], self.goal_point[1], self.goal_point[2], s=100, color= 'blue', marker = 'X')
        poly = Poly3DCollection(verts, alpha = .9)
        poly.set_facecolor(colors)
        poly.set_edgecolor('black')
        ax.add_collection(poly)
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Environment')
            ax.set_xlim(self.boundary[0], self.boundary[0+3])
            ax.set_ylim(self.boundary[1], self.boundary[1+3])
            ax.set_zlim(self.boundary[2], self.boundary[2+3])
            ax.set_ylim
            ax.legend()
            plt.tight_layout()
            plt.show()

        return ax
        
    def list_of_coords(self, coords):
        xmin, ymin, zmin, xmax, ymax, zmax = coords
        print(f"Coords: {coords}")
        verts = []
        for ax in 'xyz':
            if(ax == 'x'):
                face1 = [(xmin, ymin, zmin), (xmin, ymin, zmax), (xmin, ymax, zmax), (xmin, ymax, zmin), (xmin, ymin, zmin)]
                face2 = [(xmax, ymin, zmin), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin), (xmax, ymin, zmin)]
            elif(ax == 'y'):
                face1 = [(xmin, ymin, zmin), (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymin, zmin), (xmin, ymin, zmin)]
                face2 = [(xmin, ymax, zmin), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin), (xmin, ymax, zmin)]
            else:
                face1 = [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymin, zmin), (xmin, ymin, zmin)]
                face2 = [(xmin, ymin, zmax), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymin, zmax), (xmin, ymin, zmax)]    
            verts.append(face1)
            verts.append(face2)  
        return np.asarray(verts)

    
    
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
