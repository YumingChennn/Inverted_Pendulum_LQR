import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim
from scipy.linalg import solve_continuous_are

# Define the system matrices (already defined previously)
M = 0.48  # Mass of the cart
m = 0.7332  # Mass of the pendulum
b = 1  # Coefficient of friction for the cart
I = 1/3*0.065988  # Mass moment of inertia of the pendulum
g = 9.8  # Gravitational acceleration
l = 0.3  # Length to pendulum center of mass

p = I * (M + m) + M * m * l**2  # Denominator for the A and B matrices

# Define the system matrices A and B
A = np.array([
    [0, 1, 0, 0],
    [0, -(I + m * l**2) * b / p, (m**2 * g * l**2) / p, 0],
    [0, 0, 0, 1],
    [0, -(m * l * b) / p, m * g * l * (M + m) / p, 0]
])

B = np.array([
    [0],
    [(I + m * l**2) / p],
    [0],
    [m * l / p]
])

# Define the output matrix C and D matrix
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

D = np.array([
    [0],
    [0]
])

# Define the weighting matrices Q and R for LQR control
Q = np.diag([1, 0, 1, 0])  # State weighting matrix
R = np.array([[1]])  # Input weighting scalar

# Solve the Continuous Algebraic Riccati Equation (CARE) to find the optimal gain K
P = solve_continuous_are(A, B, Q, R)
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
    #u = np.clip(u, -0.1, 0.1)
    x_dot = A @ x + B @ u
    x = x + x_dot * dt
    x_history.append(x)
    u_history.append(u)

x_history = np.array(x_history)
u_history = np.array(u_history)

# Output response for position and angle
y = x_history[:, [0, 2]]

# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, y[:-1, 0], label='Cart Position (x)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Cart Position vs Time (LQR Control)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, y[:-1, 1], label='Pendulum Angle (phi)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Pendulum Angle vs Time (LQR Control)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, u_history, label='Control Input (U)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Input')
plt.title('Control Input vs Time')
plt.legend()

plt.tight_layout()
plt.show()
