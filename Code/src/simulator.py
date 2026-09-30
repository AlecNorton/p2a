import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp
import time
import os
from matplotlib.animation import FuncAnimation

# Local imports
from environment import Environment3D
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator
from control import QuadrotorController

# Dynamics and parameters
from quad_dynamics import model_derivative
import tello as drone_params

class LiveQuadrotorSimulator:
    """
    Real-time live visualization quadrotor simulator
    Shows step-by-step: RRT* planning -> B-spline -> Execution
    """
    
    def __init__(self, map_file=None):
        # Initialize components
        self.env = Environment3D()
        if map_file:
            success = self.env.parse_map_file(map_file)
            if not success:
                print(f"Failed to load map file: {map_file}")
        
        self.planning_env = copy.deepcopy(self.env)
        self.planning_env.tracking_reserve = 0.05
        self.planner = PathPlanner(self.planning_env)
        self.traj_gen = None
        self.controller = QuadrotorController(drone_params)
        
        # Create log directory
        if not os.path.exists('./log'):
            os.makedirs('./log')
        
        # Simulation parameters
        self.dt = 0.02  # 50 Hz for smoother animation
        self.sim_time = 0.0
        self.max_sim_time = 30.0
        
        # Quadrotor state: [x,y,z,vx,vy,vz,qx,qy,qz,qw,p,q,r]
        self.state = np.zeros(13)
        self.state[9] = 1.0  # Initialize quaternion w = 1
        
        # Logging
        self.state_history = []
        self.time_history = []
        self.control_history = []
        
        # Simulation status
        self.goal_reached = False
        self.goal_tolerance = 0.2  # meters; require settling as well as proximity
        self.goal_speed_tolerance = 0.15
        self.collision_detected = False
        self.stop_reason = "not_started"
        self.simulation_active = False
        
        # Visualization elements
        self.fig = None
        self.ax = None
        self.drone_point = None
        self.drone_trail = None
        self.trail_positions = []
        self.max_trail_length = 200
        
        # RRT* visualization elements
        self.rrt_tree_lines = []
        self.rrt_nodes_scatter = None
        self.rrt_path_line = None
        self.bspline_line = None
        
        # Planning phase tracking
        self.planning_complete = False
        self.trajectory_complete = False
        self.execution_started = False
        
    def setup_visualization(self):
        """Setup the 3D visualization for step-by-step planning"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the environment
        self._draw_environment()
        
        # Set equal aspect ratio and limits
        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Quadrotor Path Planning and Execution')
        
        plt.show()
    
    def _set_inputs(self, start=None, goal=None):
        # Keep map1 defaults when valid; otherwise use reproducible interior defaults.
        lo,hi = np.array(self.env.boundary[:3]),np.array(self.env.boundary[3:])
        for name, supplied, fraction in (("start_point",start,0.2),("goal_point",goal,0.8)):
            point = supplied if supplied is not None else getattr(self.env,name)
            if supplied is None and not self.env.is_point_in_free_space(point):
                point = lo+(hi-lo)*fraction
                point[2] = (lo[2]+hi[2])/2
                if not self.env.is_point_in_free_space(point):
                    point = self.env.generate_random_free_point()
            if point is None or not self.env.is_point_in_free_space(point):
                raise ValueError(f"{name} is invalid with the configured safety margin: {point}")
            setattr(self.env,name,list(point))
            setattr(self.planning_env,name,list(point))
            if not self.planning_env.is_point_in_free_space(point):
                raise ValueError(f"{name} needs 0.05 m additional obstacle tracking clearance")

    def animated_rrt_planning(self, start=None, goal=None):
        self._set_inputs(start,goal)
        self.ax.set_title("Phase 1: RRT* planning")
        if not self.planner.plan_path(on_expand=self._update_rrt_visualization):
            return False
        self._show_final_rrt_path()
        self.planning_complete = True
        return True

    def _generate_reference(self):
        self.traj_gen = TrajectoryGenerator(self.planner.waypoints)
        result = self.traj_gen.generate_bspline_trajectory(env=self.planning_env)
        self.controller.set_trajectory(*result)
        self.max_sim_time = self.traj_gen.trajectory_duration + 10.0
        self.trajectory_complete = True
        return result

    def _reset_execution(self):
        self.controller = QuadrotorController(drone_params)
        self.state = np.zeros(13)
        self.state[:3] = self.env.start_point
        self.state[9] = 1.0  # Existing dynamics use scalar-first [0,0,0,1] = yaw pi.
        self.sim_time = 0.0
        self.state_history, self.time_history, self.control_history = [],[],[]
        self.trail_positions = [self.state[:3].copy()]
        self.goal_reached = self.collision_detected = False
        self.stop_reason = "running"

    def initialize_simulation(self, start=None, goal=None):
        """Offline initialization uses the same planner/reference as live mode."""
        self._set_inputs(start,goal)
        self._reset_execution()
        if not self.planner.plan_path():
            self.stop_reason = "planning_failed"
            return False
        self.planning_complete = True
        self._generate_reference()
        self.execution_started = True
        return True

    def _update_rrt_visualization(self, tree, iteration, max_iterations, final=False):
        """Update RRT* tree visualization"""
        # Clear previous tree visualization
        for line in self.rrt_tree_lines:
            line.remove()
        self.rrt_tree_lines = []
        
        if self.rrt_nodes_scatter is not None:
            self.rrt_nodes_scatter.remove()
            self.rrt_nodes_scatter = None
        
        # Draw tree edges (sample only for performance)
        sample_rate = max(1, len(tree) // 500)  # Limit to ~500 edges
        for i, node in enumerate(tree[::sample_rate]):
            if node.parent is not None:
                line, = self.ax.plot([node.parent.position[0], node.position[0]],
                                   [node.parent.position[1], node.position[1]],
                                   [node.parent.position[2], node.position[2]],
                                   'b-', alpha=0.3, linewidth=0.5)
                self.rrt_tree_lines.append(line)
        
        # Draw nodes (sample for performance)
        if len(tree) > 1:
            sampled_nodes = tree[::max(1, len(tree) // 200)]  # Limit to ~200 nodes
            positions = np.array([node.position for node in sampled_nodes])
            self.rrt_nodes_scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                                   c='blue', s=8, alpha=0.6)
        
        # Update title
        progress = (iteration / max_iterations) * 100
        status = "COMPLETE" if final else f"{progress:.1f}%"
        self.ax.set_title(f'Phase 1: RRT* Planning - {status} (Nodes: {len(tree)}, Iter: {iteration})')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _show_final_rrt_path(self):
        """Show the final RRT* path"""
        time.sleep(1)  # Pause to show final tree
        
        # Draw final path
        if len(self.planner.waypoints) > 0:
            waypoints = np.array(self.planner.waypoints)
            self.rrt_path_line, = self.ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                                             'ro-', markersize=8, linewidth=4, 
                                             label='RRT* Path', alpha=0.9)
        
        self.ax.set_title('Phase 1: RRT* Planning - COMPLETE! Final path shown.')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        time.sleep(2)  # Show final path
        self.planning_complete = True
    
    def show_bspline_trajectory(self):
        """Show Quintic trajectory generation"""
        print("Generating Quintic trajectory...")
        
        self.ax.set_title('Phase 2: Quintic Trajectory Generation...')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Generate trajectory
        result = self._generate_reference()

        if result[0] is not None:
            trajectory_points, time_points, velocities, accelerations = result
            self.controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
            self.max_sim_time = self.traj_gen.trajectory_duration + 10.0
            
            # Draw Quintic trajectory
            # Sample trajectory for visualization
            traj_sample = trajectory_points[::5]  # Every 5th point
            self.bspline_line, = self.ax.plot(traj_sample[:, 0], traj_sample[:, 1], traj_sample[:, 2], 
                                            'g-', linewidth=3, alpha=0.8, label='Quintic Trajectory')
            
            # Add velocity vectors at key points
            vector_sample = max(1, len(trajectory_points) // 20)
            for i in range(0, len(trajectory_points), vector_sample):
                pos = trajectory_points[i]
                vel = velocities[i] * 0.4  # Scale for visualization
                self.ax.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='orange', alpha=0.7, arrow_length_ratio=0.1)
            
            self.ax.set_title('Phase 2: Quintic Trajectory - COMPLETE!')
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            time.sleep(2)  # Show trajectory
            self.trajectory_complete = True
            return True
        else:
            print("Trajectory generation failed")
            return False
    
    def initialize_execution_phase(self):
        """Initialize the execution phase"""
        print("🚁 Starting execution phase...")
        
        # Set initial state
        self.state[0:3] = self.env.start_point
        self.state[3:6] = 0  # Zero initial velocity
        self.state[6:10] = [0, 0, 0, 1]  # Identity quaternion
        self.state[10:13] = 0  # Zero angular rates
        
        # Initialize drone visualization
        current_pos = self.state[0:3]
        self.drone_point = self.ax.scatter(*current_pos, c='red', s=200, marker='o', 
                                         label='Quadrotor', edgecolors='black', linewidths=2)
        
        # Initialize trail
        self.trail_positions = [current_pos.copy()]
        self.drone_trail, = self.ax.plot([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                       'purple', linewidth=4, alpha=0.9, label='Executed Path')
        
        # Reset metrics
        self.controller.reset_metrics()
        self.sim_time = 0.0
        self.goal_reached = False
        
        self.ax.set_title('Phase 3: Executing Trajectory...')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.execution_started = True
    
    def simulation_step(self):
        """Single simulation time step"""
        if not self.simulation_active:
            return False
        
        # Get control input
        control_input = self.controller.compute_control(self.state, self.sim_time)
        
        # Dynamics integration
        def dynamics(t, x):
            return model_derivative(t, x.reshape(-1, 1), 
                                  control_input.reshape(-1, 1), 
                                  drone_params).flatten()
        
        # Integrate one step
        sol = solve_ivp(dynamics, [self.sim_time, self.sim_time + self.dt], 
                       self.state, method='RK45')
        
        previous_position = self.state[:3].copy()
        if not sol.success or not np.isfinite(sol.y[:,-1]).all():
            self.stop_reason = "integration_failed"
            raise RuntimeError("Dynamics integration failed")
        self.state = sol.y[:, -1]
        self.sim_time += self.dt
        
        # Log data
        self.state_history.append(self.state.copy())
        self.time_history.append(self.sim_time)
        self.control_history.append(control_input.copy())
        
        # Update trail
        current_pos = self.state[0:3]
        self.trail_positions.append(current_pos.copy())
        
        # Limit trail length
        if len(self.trail_positions) > self.max_trail_length:
            self.trail_positions.pop(0)
        
        # Flight safety must be checked before awarding goal success.
        if not self.env.is_line_collision_free(previous_position,current_pos):
            self.collision_detected = True
            self.stop_reason = "collision"
            self.simulation_active = False
            print(f"Collision envelope violated at {self.sim_time:.2f}s")
            return False
        reference_done = self.sim_time >= self.controller.time_points[-1]
        goal_distance = np.linalg.norm(current_pos-np.array(self.env.goal_point))
        speed = np.linalg.norm(self.state[3:6])
        if reference_done and goal_distance <= self.goal_tolerance and speed <= self.goal_speed_tolerance:
            self.goal_reached = True
            self.stop_reason = "goal_reached"
            self.simulation_active = False
            print(f"Goal reached and settled at {self.sim_time:.2f}s; error {goal_distance:.3f} m")
            return False
        if self.sim_time >= self.max_sim_time:
            self.stop_reason = "time_limit"
            self.simulation_active = False
            print(f"Simulation time limit reached at {self.sim_time:.2f}s")
            return False
        return True

    def update_execution_visualization(self):
        """Update the execution visualization"""
        if not hasattr(self, 'drone_point') or self.drone_point is None:
            return
        
        current_pos = self.state[0:3]
        
        # Update drone position
        self.drone_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Update trail
        if len(self.trail_positions) > 1:
            trail_array = np.array(self.trail_positions)
            self.drone_trail.set_data_3d(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2])
        
        # Update title with current info
        goal_dist = "N/A"
        if self.env.goal_point is not None:
            goal_dist = f"{np.linalg.norm(current_pos - np.array(self.env.goal_point)):.2f}m"
        
        self.ax.set_title(f'Phase 3: Executing - Time: {self.sim_time:.1f}s, Goal Dist: {goal_dist}')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def run_live_simulation(self, start=None, goal=None):
        """
        Run the complete simulation showing all phases:
        1. RRT* planning (animated)
        2. Quintic trajectory generation
        3. Trajectory execution
        """
        print(" Starting complete quadrotor simulation with all phases...")
        
        self._set_inputs(start,goal)
        self._reset_execution()
        # Setup visualization
        self.setup_visualization()
        
        # Phase 1: Animated RRT* Planning
        if not self.animated_rrt_planning(start, goal):
            return False
        
        # Phase 2: Quintic Trajectory Generation
        if not self.show_bspline_trajectory():
            return False
        
        # Phase 3: Trajectory Execution
        self.initialize_execution_phase()
        
        # Start execution
        self.simulation_active = True
        
        print("\nExecuting trajectory...")
        print("Close the plot window to stop simulation")
        
        execution_update_rate = 25  # Hz
        update_interval = 1.0 / execution_update_rate
        last_update_time = time.time()
        
        try:
            while self.simulation_active and plt.get_fignums():
                # Run simulation step
                continue_sim = self.simulation_step()
                
                # Update visualization at specified rate
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    self.update_execution_visualization()
                    last_update_time = current_time
                
                # Control real-time execution
                time.sleep(max(0, self.dt - (time.time() - current_time)))
                
                if not continue_sim:
                    break
                    
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        except Exception as e:
            print(f"\nSimulation error: {e}")
        finally:
            self.simulation_active = False
        
        # Final update
        self.update_execution_visualization()
        
        # Print results
        self._print_simulation_results()
        
        # Keep plot open
        print("\n Simulation complete. Close plot window to continue...")
        plt.ioff()
        plt.show()
        
        return self.goal_reached and not self.collision_detected
    
    # Keep all the other methods from before (environment drawing, results, etc.)
    def _draw_environment(self):
        """Draw the static environment elements"""
        self.env.visualize_environment(self.ax)
        # Draw boundary
        '''        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            vertices = self._create_cube_vertices(xmin, ymin, zmin, xmax, ymax, zmax)
            faces = self._create_cube_faces(vertices)
            
            for face in faces:
                face_array = np.array(face + [face[0]])
                self.ax.plot(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
                           'k--', alpha=0.3, linewidth=1)
        
        # Draw obstacles
        
        for block_coords, block_color in self.env.blocks:
            vertices = self._create_cube_vertices(*block_coords)
            faces = self._create_cube_faces(vertices)
            print(f"BC: {block_color}")
            poly3d = [[tuple(vertex) for vertex in face] for face in faces]
            self.ax.add_collection3d(Poly3DCollection(poly3d, 
                                                     facecolors=block_color, 
                                                     alpha=0.8,
                                                     edgecolors='black',
                                                     linewidths=0.5))
        
        # Draw start and goal
        if self.env.start_point:
            self.ax.scatter(*self.env.start_point, c='green', s=150, marker='s', 
                           edgecolors='black', linewidth=2, label='Start')
        if self.env.goal_point:
            self.ax.scatter(*self.env.goal_point, c='gold', s=150, marker='*', 
                           edgecolors='black', linewidth=2, label='Goal')
        '''
    
    def _create_cube_vertices(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """Create cube vertices"""
        return [
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ]
    
    def _create_cube_faces(self, vertices):
        """Create cube faces"""
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[1], vertices[5], vertices[6], vertices[2]],  # right
            [vertices[4], vertices[0], vertices[3], vertices[7]],  # left
        ]
    
    def _print_simulation_results(self):
        """Print simulation results summary"""
        print("\n" + "="*60)
        print(" COMPLETE SIMULATION RESULTS")
        print("="*60)
        
        final_pos = self.state[0:3]
        
        print(f"   PHASE SUMMARY:")
        print(f"   Phase 1: RRT* Planning - {len(self.planner.waypoints)} waypoints")
        print(f"   Phase 2: Quintic Trajectory - {len(self.controller.trajectory_points) if self.controller.trajectory_points is not None else 0} points")
        print(f"   Phase 3: Execution - {self.sim_time:.2f}s")
        
        if self.env.goal_point is not None:
            goal_distance = np.linalg.norm(final_pos - np.array(self.env.goal_point))
            start_goal_dist = np.linalg.norm(np.array(self.env.goal_point) - np.array(self.env.start_point))
            success_rate = max(0, (1-goal_distance/max(start_goal_dist,1e-9))*100)
            
            print(f"\n EXECUTION RESULTS:")
            print(f"   Status: {' GOAL REACHED' if self.goal_reached else ' NOT REACHED'}")
            print(f"   Final distance to goal: {goal_distance:.3f} m")
            print(f"   Goal-distance progress: {success_rate:.1f}% (not a trial success rate)")
        
        print(f"\n PERFORMANCE:")
        print(f"   Execution time: {self.sim_time:.2f} s")
        print(f"   Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        print(f"   Stop reason: {self.stop_reason}")
        
        if self.controller.position_errors:
            mean_pos_error = np.mean(self.controller.position_errors)
            max_pos_error = np.max(self.controller.position_errors)
            print(f"   Tracking error: {mean_pos_error:.3f}m avg, {max_pos_error:.3f}m max")
        
        print("="*60)
    
    def save_tracking_plots(self, filename_prefix='complete_simulation'):
        """Export aligned desired/actual traces and complete paths for the report."""
        if not self.state_history:
            return
        times = np.array(self.time_history)
        states = np.array(self.state_history)
        refs = np.array([self.controller.get_desired_state(t) for t in times])
        fig, axes = plt.subplots(2,3,figsize=(12,7),sharex=True)
        for j,axis in enumerate('XYZ'):
            for row in range(2):
                ax = axes[row,j]
                ax.plot(times,refs[:,row,j],'--',color='tab:blue',label='Desired')
                ax.plot(times,states[:,j+3*row],color='tab:orange',label='Actual')
                ax.set_title(axis+(' position' if row==0 else ' velocity'))
                ax.set_ylabel('Position (m)' if row==0 else 'Velocity (m/s)')
                ax.grid(alpha=0.3);ax.legend()
                if row==1:ax.set_xlabel('Time (s)')
        fig.suptitle(f'Trajectory tracking: {self.stop_reason}')
        fig.tight_layout()
        fig.savefig(f'./log/{filename_prefix}_tracking.png',dpi=180)
        plt.close(fig)
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121,projection='3d')
        self.env.visualize_environment(ax)
        points = self.controller.trajectory_points
        ax.plot(*points.T,'--',color='tab:blue',label='Desired')
        ax.plot(*states[:,:3].T,color='tab:orange',label='Actual')
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)');ax.legend()
        ax.set_title('Flight path: oblique view')
        top = fig.add_subplot(122)
        from matplotlib.patches import Rectangle
        for bounds, rgb in self.env.blocks:
            top.add_patch(Rectangle(bounds[:2],bounds[3]-bounds[0],bounds[4]-bounds[1],
                facecolor=np.array(rgb)/255,alpha=.25))
        top.plot(points[:,0],points[:,1],'--',color='tab:blue',label='Desired')
        top.plot(states[:,0],states[:,1],color='tab:orange',label='Actual')
        top.set_xlabel('X (m)');top.set_ylabel('Y (m)');top.set_title('Top projection (obstacle heights omitted)')
        top.set_aspect('equal',adjustable='datalim');top.legend();fig.tight_layout()
        fig.savefig(f'./log/{filename_prefix}_path.png',dpi=180);plt.close(fig)
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
        self.env.visualize_environment(ax);self.planner.visualize_tree(ax)
        ax.set_title('Explored RRT* tree and selected path');fig.tight_layout()
        fig.savefig(f'./log/{filename_prefix}_tree.png',dpi=160);plt.close(fig)

    def save_results(self, filename_prefix='complete_simulation'):
        """Save complete simulation results"""
        import scipy.io
        
        data = {
            'time': np.array(self.time_history),
            'states': np.array(self.state_history),
            'controls': np.array(self.control_history),
            'rrt_waypoints': np.array(self.planner.waypoints),
            'bspline_trajectory': self.controller.trajectory_points,
            'executed_trail': np.array(self.trail_positions),
            'goal_reached': self.goal_reached,
            'collision_detected': self.collision_detected,
            'stop_reason': self.stop_reason,
            'reference_time': self.controller.time_points,
            'reference_velocity': self.controller.trajectory_velocities,
            'reference_acceleration': self.controller.trajectory_accelerations,
            'sim_time': self.sim_time,
            'start_point': np.array(self.env.start_point),
            'goal_point': np.array(self.env.goal_point)
        }
        
        filename = f'./log/{filename_prefix}.mat'
        scipy.io.savemat(filename, data)
        self.save_tracking_plots(filename_prefix)
        print(f" Complete simulation results saved to {filename}")
