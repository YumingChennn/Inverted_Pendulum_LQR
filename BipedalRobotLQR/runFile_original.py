import numpy as np
import mujoco
import mujoco.viewer as viewer
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import math

def get_imu_data(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    return data.sensordata[sensor_id:sensor_id+3]

def get_pitch_angle(model, data, gyro_sensor, accel_sensor):
    gyro_data = get_imu_data(model, data, gyro_sensor)
    accel_data = get_imu_data(model, data, accel_sensor)
    
    # Extract the angular velocity around the Y-axis (pitch axis)
    angular_velocity_y = gyro_data[1]
    
    # Integrate angular velocity to get the pitch angle (in radians)
    global previous_time, previous_pitch_angle
    if previous_time is None:
        previous_time = data.time
        previous_pitch_angle = 0.0  # Assuming the initial pitch angle is zero

    delta_time = data.time - previous_time
    previous_time = data.time

    pitch_angle = previous_pitch_angle + angular_velocity_y * delta_time
    previous_pitch_angle = pitch_angle

    return pitch_angle, angular_velocity_y

def get_sensor_data(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr:sensor_adr+sensor_dim]

# Find a K matrix for a linearized inverted pendulum using LQR
def inverted_pendulum_lqr_K(m, d):
    A = np.zeros((2 * m.nv, 2 * m.nv))
    B = np.zeros((2 * m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-7, 1, A=A, B=B, C=None, D=None)

    # Convert to continuous time
    A = A - np.eye(A.shape[0])
    A = A / m.opt.timestep
    B = B / m.opt.timestep
    print("m.opt.timestep", m.opt.timestep)
    print("A", A)
    print("B", B)

    Q = 1 * np.diag([10] * (2 * m.nv))
    R = np.eye(B.shape[1])

    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ S)
    print("K shape:", K.shape)
    return K

# Initialize previous Euler angles to None
previous_euler_angles = None
previous_time = None

initial_position = None
previous_position = None
previous_posi_time = None


# Lists to store data for plotting
Time_data = []
Theta = []
Theta_dot = []
Position = []
Position_dot = []
wheellist = []
wheellist_dot = []
u_data = []

def inverted_pendulum_control(m, d, K, qpos0):
    global previous_euler_angles, previous_time, initial_position, previous_position, previous_posi_time, time_data, pitch_y_data, u_data, current_position_data
    try:
        current_position = get_sensor_data(m, d, 'sensor_global')
        if initial_position is None:
            initial_position = [0, 0, 0]
            previous_position = current_position
            previous_posi_time = d.time
        distance_moved = current_position - initial_position
        time_step = d.time - previous_posi_time
        if time_step > 0:
            derivative_distance_moved = (distance_moved - previous_position) / time_step
        else:
            derivative_distance_moved = np.zeros_like(distance_moved)

        previous_position = distance_moved
        previous_posi_time = d.time

        pitch_angle, pitch_rate = get_pitch_angle(m, d, 'imu_gyro', 'imu_accel')
        pitch_degree = pitch_angle / math.pi * 180
        print("pitch_degree", pitch_degree)
        
        d.qpos = qpos0
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1, qpos0, d.qpos)
        x = np.hstack((dq, d.qvel)).T
        print(f'x shape: {x.shape}')
        u = -K @ x
        u = np.clip(u, -0.33, 0.33)
        print(f'u: {u}')
        d.ctrl[0] = u[0]  # Assuming the first control input corresponds to 'wheel_joint_right'
        d.ctrl[1] = -u[0]  # Assuming the second control input corresponds to 'wheel_joint_left'
        #print("d.joint('wheel_joint_left').qpos",d.joint('wheel_joint_left').qpos)
        wheellist.append(d.joint('wheel_joint_left').qpos) 
        wheellist_dot.append(d.joint('wheel_joint_left').qvel)
        Time_data.append(d.time)
        Theta.append(pitch_angle)
        Theta_dot.append(pitch_rate)
        Position.append(distance_moved[0])
        Position_dot.append(derivative_distance_moved[0])
        u_data.append(u[0])
    
    except Exception as e:
        print("An error occurred in inverted_pendulum_control:", str(e))
        raise

def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('wheeled_bipedal_robot.xml')
    d = mujoco.MjData(m)


    if m is not None:
        # Set the control callback
        K = inverted_pendulum_lqr_K(m, d)
        qpos0 = d.qpos.copy()
        mujoco.set_mjcb_control(lambda m, d: inverted_pendulum_control(m, d, K, qpos0))
    return m, d

if __name__ == '__main__':
    # Launch the viewer and run the simulation
    viewer.launch(loader=load_callback)

    # Plot the results
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color='b')
    ax1.plot(Time_data, Position, 'b', label='Position (m)')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Position_dot', color='r')
    ax2.plot(Time_data, Position_dot, 'r', label='Position_dot')
    ax2.tick_params(axis='y', labelcolor='r')

    # Plot the control input (u) on a separate subplot
    fig2, ax3 = plt.subplots()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pitch angle', color='b')
    ax3.plot(Time_data, Theta, 'b', label='Pitch angle')
    ax3.tick_params(axis='y', labelcolor='b')

    ax4 = ax3.twinx()
    ax4.set_ylabel('Pitch rate', color='r')
    ax4.plot(Time_data, Theta_dot, 'r', label='Pitch rate')
    ax4.tick_params(axis='y', labelcolor='r')

    # Plot the control input (u) on a separate subplot
    fig3, ax5 = plt.subplots()
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Control Input', color='b')
    ax5.plot(Time_data, u_data, 'b', label='Control Input')
    ax5.tick_params(axis='y', labelcolor='b')

    # Plot the control input (u) on a separate subplot
    fig4, ax6 = plt.subplots()
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('wheellist', color='b')
    ax6.plot(Time_data, wheellist, 'b', label='wheellist')
    ax6.tick_params(axis='y', labelcolor='b')

    ax7 = ax6.twinx()
    ax7.set_ylabel('wheellist_dot', color='r')
    ax7.plot(Time_data, wheellist_dot, 'r', label='wheellist_dot')
    ax7.tick_params(axis='y', labelcolor='r')


    # Adjust layout and add title and legend
    fig.tight_layout()
    plt.title('Position and Angle with LQR Control')
    fig.legend(loc='upper right')

    fig2.tight_layout()
    plt.title('Control Input with LQR Control')
    fig2.legend(loc='upper right')

    plt.show()