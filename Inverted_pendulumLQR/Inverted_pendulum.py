import numpy as np
import mujoco
import mujoco.viewer as viewer
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import math

# Find a K matrix for a linearized inverted pendulum using LQR
def inverted_pendulum_lqr_K(m, d):
    # m.nv  # Alias for the number of actuators.
    # m.nu  # Shortcut for the number of DoFs.
    A = np.zeros((2*m.nv, 2*m.nv))
    B = np.zeros((2*m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-7, 1, A=A, B=B, C=None, D=None)

    # Convert to continuous time
    A = A - np.eye(A.shape[0])
    A = A / m.opt.timestep
    B = B / m.opt.timestep

    Q = 1 * np.diag((10, 10, 10, 10))
    R = np.eye(B.shape[1])

    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ S)
    return K

# Lists to store data for plotting
Time_data = []
Theta = []
Theta_dot = []
Position = []
Position_dot = []
u_data = []

def inverted_pendulum_control(m, d, K):
    x = np.concatenate((
        d.joint('slider').qpos,
        d.joint('hinge').qpos,
        d.joint('slider').qvel,
        d.joint('hinge').qvel   
    ))

    u = -K @ x

    d.ctrl[0] = u

    Time_data.append(d.time)
    Theta.append(x[1])
    Theta_dot.append(x[3])
    Position.append(x[0])
    Position_dot.append(x[2])
    u_data.append(u[0])
    

def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('inverted_pendulum.xml')
    d = mujoco.MjData(m)
    if m is not None:

         # Set the control callback
        K = inverted_pendulum_lqr_K(m, d)
        print('K shape', K.shape)

        # Set some initial conditions
        d.joint('slider').qpos = 0 #range="-1 1"
        d.joint('hinge').qpos = 0.3 #range="-1.57 1.57"

        mujoco.set_mjcb_control(lambda m, d: inverted_pendulum_control(m, d, K))

    return m, d

if __name__ == '__main__':
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


    # Adjust layout and add title and legend
    fig.tight_layout()
    plt.title('Position and Angle with LQR Control')
    fig.legend(loc='upper right')

    fig2.tight_layout()
    plt.title('Control Input with LQR Control')
    fig2.legend(loc='upper right')

    plt.show()

