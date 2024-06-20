import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Define system parameters
R = 0.0953   # Wheel radius
D = 0.186    # Distance between the left and right wheels
l = 0.1      # Distance from the pendulum's center of mass to the pivot
m = 0.2      # Mass of the wheel
M = 0.7331     # Mass of the pendulum
I = (1/2)*m*R**2  # Moment of inertia of the wheel
Jz = (1/3)*M*l**2  # Moment of inertia of the pendulum about the z-axis (pitch direction)
Jy = (1/12)*M*D**2  # Moment of inertia of the pendulum about the y-axis (yaw direction)
g = 9.8      # Gravitational acceleration

Q_eq = Jz*M + (Jz+M*l*l) * (2*m+(2*I)/R**2)

# Define system matrices
A_23 = -(M**2 * l**2 * g) / Q_eq
A_43 = M * l * g * (M + 2*m + (2*I / R**2)) / Q_eq
A = np.array([[0, 1, 0, 0],
              [0, 0, A_23, 0],
              [0, 0, 0, 1],
              [0, 0, A_43, 0]])

B_21 = (Jz + M*l**2 + M*l*R) / Q_eq / R
B_41 = -((M*l / R) + M + 2*m + (2*I / R**2)) / Q_eq
B = np.array([[0],
              [2 * B_21],
              [0],
              [2 * B_41]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])

# Define weight matrices for LQR
Q = np.diag([0, 0.01, 0.01, 0])  # State weighting matrix
R = np.array([[1000]])           # Input weighting scalar

# Solve Continuous Algebraic Riccati Equation (CARE)
P = solve_continuous_are(A, B, Q, R)

# Calculate LQR gain
K = np.linalg.inv(R) @ B.T @ P

# Initial conditions
x0 = np.array([0, 0, 0.43, 0])
t = np.arange(0, 10, 0.01)

# Initialize the state vector
x = x0
x_history = [x0]
u_history = []

# Simulate the system using Euler integration
dt = 0.01
for _ in t:
    u = -K @ x
    print("K",K)
    u = np.clip(u, -0.1, 0.1)
    x_dot = A @ x + B @ u
    x = x + x_dot * dt
    x_history.append(x)
    u_history.append(u)

x_history = np.array(x_history)
u_history = np.array(u_history)

# Output response for position and angle
y = x_history[:, [0, 2]]

# Plot the results
fig, ax1 = plt.subplots()

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)', color='b')
ax1.plot(t, y[:-1, 0], 'b', label='Position (m)')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.set_ylabel('Angle (rad)', color='r')
ax2.plot(t, y[:-1, 1], 'r', label='Angle (rad)')
ax2.tick_params(axis='y', labelcolor='r')

# Plot the control input (u) on a separate subplot
fig2, ax3 = plt.subplots()
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Control Input', color='g')
ax3.plot(t, u_history, 'g', label='Control Input')
ax3.tick_params(axis='y', labelcolor='g')

# Adjust layout and add title and legend
fig.tight_layout()
plt.title('Position and Angle with LQR Control')
fig.legend(loc='upper right')

fig2.tight_layout()
plt.title('Control Input with LQR Control')
fig2.legend(loc='upper right')

plt.show()
