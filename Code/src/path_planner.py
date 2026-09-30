import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random 
import math
class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position, parent=None):
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = 0.0
        self.children = []

class PathPlanner:
    """
    Robust RRT* implementation for 3D path planning
    """
    
    def __init__(self, environment):
        self.env = environment
        self.waypoints = []
        self.tree_nodes = []
        
        # RRT* parameters
        self.max_iterations = 3000
        self.step_size = 1
        self.goal_radius = .5
        self.search_radius = 10
        self.goal_bias = 0.10  # 15% bias towards goal
        self.goal_node = None
        self.refinement_iterations = 1000
        self.refinement_flag = True

    def set_vars(self, max_iterations, step_size, goal_radius, search_radius, goal_bias):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.search_radius = search_radius
        self.goal_bias = goal_bias
    
    ############################################################################################################
    #### TODO - Implement RRT* path planning algorithm in 3D (use the provided environment class) ##############
    #### TODO - Store the final path in self.waypoints as a list of 3D points ##################################
    #### TODO - Add member functions as needed #################################################################
    ############################################################################################################

    def plan_path(self, on_expand=None):
        """Search then refine; store one start-to-exact-goal path for all modes.

        Reviewed against Group3_p2a's planner interface; fixes are original.
        """
        start, goal = np.array(self.env.start_point), np.array(self.env.goal_point)
        if not self.env.is_point_in_free_space(start) or not self.env.is_point_in_free_space(goal):
            raise ValueError("Start and goal must be in free space with the safety margin")
        self.waypoints = []
        self.goal_node = None
        root = RRTNode(start)
        self.tree_nodes = [root]
        if np.linalg.norm(goal-start) < 1e-9:
            self.waypoints = [start.tolist(), goal.tolist()]
            self.goal_node = root
            return True
        first_solution = None
        total = self.max_iterations + (self.refinement_iterations if self.refinement_flag else 0)
        for k in range(total):
            if first_solution is None and k >= self.max_iterations:
                break
            if first_solution is not None and k-first_solution > self.refinement_iterations:
                break
            draw = random()
            if draw < self.goal_bias:
                sample = goal
            elif draw < self.goal_bias + 0.15:
                # Vertical exploration helps escape narrow starting columns and
                # alternating high/low barriers, while uniform sampling remains.
                sample = self.tree_nodes[np.random.randint(len(self.tree_nodes))].position.copy()
                sample[2] = np.random.uniform(self.env.boundary[2]+self.env.safety_margin,
                                              self.env.boundary[5]-self.env.safety_margin)
            else:
                sample = self.env.generate_random_free_point()
            if sample is None:
                continue
            nearest = self.find_closest_node(RRTNode(sample))
            pos = self.steer(nearest.position, sample)
            if np.linalg.norm(pos-nearest.position) < 1e-8 or not self.is_path_valid(nearest.position,pos):
                continue
            near = self.find_near_nodes(RRTNode(pos))
            parent, cost = self.choose_parent(near,pos)
            if parent is None:
                parent, cost = nearest, nearest.cost + np.linalg.norm(pos-nearest.position)
            node = RRTNode(pos)
            self.parent_child(parent,node)
            self.tree_nodes.append(node)
            self.rewire(node)
            if np.linalg.norm(pos-goal) <= self.goal_radius and self.is_path_valid(pos,goal):
                candidate_cost = node.cost + np.linalg.norm(pos-goal)
                if self.goal_node is None:
                    self.goal_node = RRTNode(goal)
                    self.parent_child(node,self.goal_node)
                    # Keep the terminal node outside sampling/rewiring to avoid duplicate goals.
                    first_solution = k
                elif candidate_cost < self.goal_node.cost - 1e-9:
                    self.parent_child(node,self.goal_node)
                if not self.refinement_flag:
                    break
            if on_expand is not None and k % 50 == 0:
                on_expand(self.tree_nodes,k,total)
        if self.goal_node is None:
            return False
        self.waypoints = self.simplify_path(self.extract_path(self.goal_node))
        if on_expand is not None:
            on_expand(self.tree_nodes,k,total)
        print(f"Planned {len(self.waypoints)} waypoints, cost {self.goal_node.cost:.3f} m")
        return True

    def steer(self, neighbor_point, sample_point):
        origin = np.asarray(neighbor_point,dtype=float)
        direction = np.asarray(sample_point,dtype=float)-origin
        length = np.linalg.norm(direction)
        if length < 1e-12:
            return origin.copy()
        return origin + direction * min(self.step_size,length)/length

    def find_closest_node(self, new_node, tree=None):
        smallest_dist = -1
        closestNode = None
        if(tree is None):

            for node in self.tree_nodes:

                dist = self.euclidian_dist(new_node, node)
                if(dist < smallest_dist or smallest_dist == -1):
                    smallest_dist = dist
                    closestNode = node
        else:
            for node in tree:
                dist = self.euclidian_dist(new_node, node)
                if(dist < smallest_dist or smallest_dist == -1):
                    smallest_dist = dist
                    closestNode = node
        return closestNode


    def find_near_nodes(self, node1):
        neighboring_nodes = []
        for node2 in self.tree_nodes:
            if(self.euclidian_dist(node1, node2) <= self.search_radius):
                neighboring_nodes.append(node2)
        return neighboring_nodes

    def rewire(self, node1):
        neighboring_nodes = self.find_near_nodes(node1)
        #First "reparent" new node based on distance to starting point. 
        for neighbor in neighboring_nodes:
            if(neighbor.cost + self.euclidian_dist(node1, neighbor) < node1.cost):
                if(self.env.is_line_collision_free(neighbor.position, node1.position) == False):
                    #Collision so just ignore this one. 
                    continue
                #print("Reparenting.")
                self.parent_child(neighbor, node1)
        #Once done, node1 can become a parent of other neighboring nodes
        for neighbor in neighboring_nodes:
            if(node1.cost + self.euclidian_dist(node1, neighbor) < neighbor.cost):
                if(self.env.is_line_collision_free(neighbor.position, node1.position) == False):
                    continue
                self.parent_child(node1, neighbor)
    
    def is_path_valid(self, pos1, pos2):
        return self.env.is_line_collision_free(pos1, pos2)

    def choose_parent(self, neighboring_nodes, new_pos):
        best_parent, best_cost = None, math.inf
        for neighbor in neighboring_nodes:
            cost = neighbor.cost + np.linalg.norm(neighbor.position-new_pos)
            if cost < best_cost and self.is_path_valid(neighbor.position,new_pos):
                best_parent, best_cost = neighbor, cost
        return best_parent,best_cost

    def get_waypoints(self, end_node):
        points, seen = [], set()
        while end_node is not None:
            if id(end_node) in seen:
                raise RuntimeError("Cycle in planner parent links")
            seen.add(id(end_node)); points.append(end_node.position)
            end_node = end_node.parent
        return np.asarray(points)

    def simplify_path(self, points):
        """Retain only necessary intermediate waypoints using valid shortcuts."""
        if len(points) < 3:
            return points
        result, i = [points[0]], 0
        while i < len(points)-1:
            j = len(points)-1
            while j > i+1 and not self.is_path_valid(points[i],points[j]):
                j -= 1
            result.append(points[j]); i = j
        return result

    def extract_path(self, goal_node):
        return [point.tolist() for point in self.get_waypoints(goal_node)[::-1]]

    
    def euclidian_dist(self, node1: RRTNode, node2:RRTNode):
        pos1 = node1.position
        pos2 = node2.position
        #print(f"Pos1:{pos1}")
        #print(f"Pos2:{pos2}")
        return np.linalg.norm(pos1 -pos2)

    def parent_child(self, parentNode, childNode):
        ancestor = parentNode
        while ancestor is not None:
            if ancestor is childNode:
                raise ValueError("Reparenting would introduce a cycle")
            ancestor = ancestor.parent
        if childNode.parent is not None:
            childNode.parent.children.remove(childNode)
        childNode.parent = parentNode
        childNode.cost = parentNode.cost + self.euclidian_dist(parentNode,childNode)
        parentNode.children.append(childNode)
        stack = [childNode]
        while stack:
            node = stack.pop()
            for child in node.children:
                child.cost = node.cost + self.euclidian_dist(node,child)
                stack.append(child)

    def visualize_tree(self, ax=None):
        """Visualize the RRT* tree"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False
        
        # Draw tree edges
        for node in self.tree_nodes:
            if node.parent is not None:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       [node.parent.position[2], node.position[2]],
                       'b-', alpha=0.3, linewidth=0.5)
        
        # Draw tree nodes
        if self.tree_nodes:
            positions = np.array([node.position for node in self.tree_nodes])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='blue', s=10, alpha=0.6)
        
        # Draw final path
        if len(self.waypoints) > 0:
            waypoints = np.array(self.waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   'ro-', markersize=8, linewidth=3, label='RRT* Path')
        
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RRT* Tree and Path')
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        return ax
