import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt
import math 
from scipy.optimize import fsolve, fmin_l_bfgs_b
from bisect import bisect 

def get_pos(a, t = None):
        a0, a1, a2, a3, a4, a5 = a
        return a0+a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    
def get_vel(a, t = None):
    a0, a1, a2, a3, a4, a5 = a
    return a1+ 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    
def get_accel(a, t = None):
    a0, a1, a2, a3, a4, a5 = a
    return 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

def get_snap(a, t = None):
    a0, a1, a2, a3, a4, a5 = a
    return 6*a3 + 24*a4*t + 60*a5*t

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using  
splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.trajectory_duration = None  # seconds
        self.max_velocity = 5  # m/s #Assume this means in total. 
        self.max_acceleration = 1  # m/s^2
        self.avg_velocity = 3
        self.path_length_multiplier = 5
        self.time_points = None


    ##############################################################
    #### TODO - Implement spline trajectory generation ###########
    #### TODO - Ensure velocity and acceleration constraints #####
    #### TODO - Add member functions as needed ###################
    ##############################################################
    def generate_spline_trajectory(self, num_points=None):
        """
        Generate spline trajectory with complete velocity/acceleration profiles
        
        Parameters:
        - num_points: number of points in the smooth trajectory
        
        Returns:
        - trajectory_points: numpy array of shape (num_points, 3)
        - time_points: numpy array of time stamps
        - velocities: numpy array of velocities (num_points, 3)
        - accelerations: numpy array of accelerations (num_points, 3)
        """
        print("Generating spline trajectory...")
        if(self.avg_velocity is None):
            vf = [math.sqrt(1), math.sqrt(1), math.sqrt(1)]
        else:
            vf = [math.sqrt(self.avg_velocity/3), math.sqrt(self.avg_velocity/3), math.sqrt(self.avg_velocity/3)]
        af = 0
        trajectory_points = None
        time_points = None
        velocities = None
        accelerations = None

        ############## IMPLEMENTATION STARTS HERE ##############

        #First determine
        time_points = [0.0]
        trajectory_points = []
        velocities = []
        accelerations = []
        piecewise = [] #Piecewise[i][k], i is sections of spline, k is (x, y, z) position, and j is (pos, vel, accel) = [a0, a1, a2, a3, a4, a5]
        waypoint_pairs = self.generate_waypoint_pairs()
        print(f"Waypoint Pairs: {waypoint_pairs}")
        for waypoint_pair in waypoint_pairs:
            if piecewise:
                v0 = vf
                a0 = af
                t0 = tf
                print(f"Piecewise Len: {len(piecewise)}, len_waypoint: {len(waypoint_pairs)}")
                if(len(piecewise)+1 == len(waypoint_pairs)):
                    print(f"Last waypoint: {waypoint_pair}")
                    #last segment
                    initial_conds_xyz = self.generate_initial_conds(waypoint_pair, v0, a0, t0)
                    #self.path_length_multiplier = .01
                    print("Done generatnig conds.")
                else:
                    initial_conds_xyz = self.generate_initial_conds(waypoint_pair, v0, a0, t0)
                    vf = [initial_conds_xyz[0][3], initial_conds_xyz[1][3], initial_conds_xyz[2][3]]
                    af = [initial_conds_xyz[0][5], initial_conds_xyz[1][5], initial_conds_xyz[2][5]]
                    #self.path_length_multiplier = .01
                    print("Done generatnig conds.\n")
                    

                #coeffs, tf = self.loop_to_optimize_time(t0, tf, initial_conds_xyz)
                coeffs, tf = self.generate_coeffs(initial_conds_xyz)
                if(coeffs is not None):
                    piecewise.append(coeffs)
                    time_points.append(tf)
                    
                else:
                    print("FUCK")
                    break
            else:
                v0 = [0, 0, 0]
                a0 = [0, 0, 0]
                t0 = 0
                tf = .01
                initial_conds_xyz =  self.generate_initial_conds(waypoint_pair, v0, a0, t0)
                vf = [initial_conds_xyz[0][3], initial_conds_xyz[1][3], initial_conds_xyz[2][3]]
                af = [initial_conds_xyz[0][5], initial_conds_xyz[1][5], initial_conds_xyz[2][5]]
                print("Done generatnig conds.")
                #self.path_length_multiplier = .01
                #coeffs, tf = self.loop_to_optimize_time(t0, tf, initial_conds_xyz)
                coeffs, tf= self.generate_coeffs(initial_conds_xyz)
                if(coeffs is not None):
                    piecewise.append(coeffs)
                    time_points.append(tf)
                    
                else:
                    print("FUCK")
                    break
        print(f"Piecewise: {piecewise}")
        print(f"Time_Points: {time_points}")
        #For loop
        self.trajectory_duration = tf
        t_diff = tf / num_points
        print(f"T_diff: {t_diff}")
        currTime = 0.0
        currentTraj = 0
        for i in range(0, num_points):
            coeffs = piecewise[currentTraj]
            trajectory_points.append(self.get_position(coeffs, currTime))
            velocities.append(self.get_velocity(coeffs, currTime))
            accelerations.append(self.get_acceleration(coeffs, currTime))
            currTime = currTime + t_diff
            currentTraj = bisect(time_points, currTime)-1
            print(f"Current Traj: {currentTraj}, time: {currTime}")

        #print(f"Trajectory Pts: {trajectory_points}")
        trajectory_points = np.array(trajectory_points)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        self.time_points = time_points
        return trajectory_points, time_points, velocities, accelerations


    def generate_waypoint_pairs(self):
        waypoint_pairs = []
        for i in range(0, len(self.waypoints)-1):
            waypoint_pairs.append([self.waypoints[i], self.waypoints[i+1]])
        return waypoint_pairs

    def generate_initial_conds(self, waypoint_pair, v0, a0, t0):
        initial_conds = []
        p1 = waypoint_pair[0]
        p2 = waypoint_pair[1]
        diff = np.subtract(np.array(p2) ,np.array(p1))
        path_length = np.linalg.norm(diff)
        tf = t0 + path_length*self.path_length_multiplier
        print(f"TF: {tf}")
        for i in range(0, 3):
            path_length = np.linalg.norm(np.subtract(p1, p2))
            vf = diff[i]/(tf-t0)
            af = (vf-v0[i])/(tf-t0)
            initial_conds.append([waypoint_pair[0][i], waypoint_pair[1][i], v0[i], vf, a0[i], af])
        #max_vel, max_acc = self.get_max_vel_and_acc(initial_conds, t0, tf)
        initial_conds.append([t0, tf])
        
        return initial_conds

    def get_max_vel_and_acc(self, initial_conds, t0, tf):
        total_vel = 0
        total_acc = 0
        for i in range(0, 3):
            a = initial_conds[i]
            max_vf = fmin_l_bfgs_b(lambda t: -get_vel(a, t), (tf-t0)/2, bounds = [(t0, tf)], approx_grad = True)
            total_vel += (-1*max_vf[1])**2
            max_af = fmin_l_bfgs_b(lambda t: -get_accel(a, t), (tf-t0)/2, bounds = [(t0, tf)], approx_grad = True)
            total_acc += (-1*max_af[1])**2
        return math.sqrt(total_vel), math.sqrt(total_acc)
    
    def generate_a(self, initial_conds, t0, tf):
        coeffs = []
        for i in range(0, 3):
            p0, pf, v0, vf, a0, af = initial_conds[i]
            print(f"P0: {p0}, pf: {pf}, v0: {v0}, vf: {vf}, a0: {a0}, af:{af}")
            q = np.array([p0, pf, v0, vf, a0, af])
            M = self.M(t0, tf)
            coeff = np.linalg.lstsq(M, q)[0]
            coeffs.append(coeff)
        return coeffs
    '''
    def loop_to_optimize_time(self, t0, tf, initial_conds_xyz):
        coeffs = self.generate_a(initial_conds_xyz, t0, tf)
        count = 0
        flag, scale = self.test_constraints(coeffs, t0, tf)
        while(flag == False and count < 3000):
            #While exceeding constraints, redo the thing.
            tf = tf + .01 #Increase time more if constraints are being drastically affected. 
            count = count + 1
            print(f"Count: {count}")
            coeffs = self.generate_a(initial_conds_xyz, t0, tf)
            flag, scale = self.test_constraints(coeffs, t0, tf)
            print(f"Scale: {scale}")
        if(count == 3000):
            return None
        else:
            return coeffs, tf
    '''
        
    def generate_coeffs(self, initial_conds_xyz):
        t0, tf = initial_conds_xyz[3]
        coeffs = self.generate_a(initial_conds_xyz, t0, tf)
        return coeffs, tf

    def test_constraints(self, coeffs, t0, tf):
        print(f"Coeffs: {coeffs}")
        
        
        max_accel = self.get_max('ACCEL', t0, tf, coeffs)
        max_vel = self.get_max('VEL', t0, tf, coeffs)

        print(f"Max_Accel: {max_accel}")
        print(f"Max Vel:  {max_vel}")
        accel = np.linalg.norm(max_accel)
        vel = np.linalg.norm(max_vel)
        if(self.max_acceleration is None):
            self.max_acceleration = .5
        if(self.max_velocity is None):
            self.max_velocity = math.sqrt(3)
        if(accel > self.max_acceleration or vel > self.max_velocity):
            
            accel_diff  =abs(self.max_acceleration - accel)
            vel_diff = abs(self.max_velocity - vel)
            if(accel_diff >= vel_diff):
                if accel_diff > 100:
                    scale = 1
                else:
                    scale = accel_diff / 100.0
            else:
                if vel_diff > 100:
                    scale = 1
                else:
                    scale = vel_diff / 100
            return False, scale  #Time needs to be increased!
            
        else:
            return True, 0
        
    def get_position(self, coeffs, t):
        pos = []
        for i in range(0, 3):
            pos.append(get_pos(coeffs[i], t))
        return pos
    def get_velocity(self, coeffs, t):
        vel = []
        for i in range(0, 3):
            vel.append(get_vel(coeffs[i], t))
        return vel
    def get_acceleration(self, coeffs, t):
        accel = []
        for i in range(0, 3):
            accel.append(get_accel(coeffs[i], t))
        return accel
    def M(self, t0, tf):
        return np.array([[1, 1*t0, 1*t0**2, 1*t0**3, 1*t0**4, 1*t0**5], 
                         [1, 1*tf, 1*tf**2, 1*tf**3, 1*tf**4, 1*tf**5], 
                         [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4], 
                         [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                         [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3], 
                         [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])

    def get_max(self, type, min_t, max_t, coeffs):
        print(f"Min_t: {min_t}, max_t: {max_t}")
        maxs = []
        for i in range(0, 3):
            if(type == 'POS'):
                max = fmin_l_bfgs_b(lambda t: -get_pos(coeffs[i], t), min_t, bounds = [(min_t, max_t)], approx_grad = True)
                maxs.append(-max[1])
            elif(type == 'VEL'):
                max = fmin_l_bfgs_b(lambda t: -get_vel(coeffs[i], t),min_t, bounds = [(min_t, max_t)], approx_grad = True)
                maxs.append(-max[1])
            elif(type == 'ACCEL'):
                max = fmin_l_bfgs_b(lambda t: -get_accel(coeffs[i], t), min_t, bounds = [(min_t, max_t)], approx_grad = True)
                maxs.append(-max[1])
            elif(type == 'SNAP'):
                max = fmin_l_bfgs_b(lambda t: -get_snap(coeffs[i], t), min_t, bounds = [(min_t, max_t)], approx_grad = True)
                maxs.append(-max[1])
            else:
                return None
        return maxs
            
    
    
        
    def visualize_trajectory(self, trajectory_points=None, velocities=None, 
                           accelerations=None, ax=None):
        """Visualize the trajectory with velocity and acceleration vectors"""
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            standalone = True
        else:
            ax1 = ax
            standalone = False
        
        if trajectory_points is not None:
            # Plot 3D trajectory
            ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    trajectory_points[:, 2], 'b-', linewidth=2, label='Spline Trajectory')
            
            # Plot waypoints
            print(f"Visualizing waypoints: {self.waypoints}")
            print(f"Visualizing x: {self.waypoints[:, 0]}")
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], 
                    'ro-', markersize=8, linewidth=2, label='Waypoints')
            
            # Plot velocity vectors (sampled)
            
            if velocities is not None:
                step = max(1, len(trajectory_points) // 20)  # Show ~20 vectors
                for i in range(0, len(trajectory_points), step):
                    pos = trajectory_points[i]
                    vel = velocities[i] * 0.5  # Scale for visualization
                    ax1.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='green', alpha=0.7, arrow_length_ratio=0.1)
            
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
            if(standalone==False):
                plt.tight_layout()
                plt.show()
        
        if standalone and velocities is not None and accelerations is not None:
            # Plot velocity magnitude over time
            time_points = np.linspace(0, self.trajectory_duration, len(velocities))
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            ax2.plot(time_points, vel_magnitudes, 'g-', linewidth=2)
            ax2.axhline(y=self.max_velocity, color='r', linestyle='--', 
                       label=f'Max Vel: {self.max_velocity} m/s')
            for t in self.time_points:
                ax2.axvline(x=t, color = 'b', linestyle = '--')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True)
            ax2.legend()
            
            # Plot acceleration magnitude over time
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            ax3.plot(time_points, acc_magnitudes, 'm-', linewidth=2)
            ax3.axhline(y=self.max_acceleration, color='r', linestyle='--', 
                       label=f'Max Acc: {self.max_acceleration} m/s²')
            for t in self.time_points:
                            ax3.axvline(x=t, color = 'b', linestyle = '--')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s²)')
            ax3.set_title('Acceleration Profile')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
        
        return ax1 if not standalone else None