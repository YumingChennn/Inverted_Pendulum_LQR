import mujoco
import mujoco.viewer as viewer
import math
import time

class CustomPID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# 定義一個全局變量來記錄上一次的時間
last_time = time.time()

def inverted_pendulum_control(m, d):
    global last_time

    # PID parameters for angle control
    pid_angle_Kp = 10
    pid_angle_Ki = 0
    pid_angle_Kd = 5
    
    # Create PID controller for angle
    pid_angle = CustomPID(pid_angle_Kp, pid_angle_Ki, pid_angle_Kd)
    
    # Get current time and calculate dt
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    
    # Get current state
    hinge_pos = d.joint('hinge').qpos
    
    # Calculate angle error
    angle_error = hinge_pos  # Or use angle_error = 0.6 * math.sin(hinge_pos) if desired
    
    # Calculate control output
    angle_control = pid_angle.update(angle_error, dt)
    
    # Apply control signal
    d.actuator('slide').ctrl[0] = angle_control
    
    # Print debug information
    print(f"Angle error: {angle_error}, Angle control: {angle_control}")

def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('inverted_pendulum.xml')
    d = mujoco.MjData(m)
    if m is not None:
        # Set some initial conditions
        d.joint('slider').qpos = 0  # range="-1 1"
        d.joint('hinge').qpos = 0.2  # range="-1.57 1.57"
        mujoco.set_mjcb_control(lambda m, d: inverted_pendulum_control(m, d))
    return m, d

if __name__ == '__main__':
    viewer.launch(loader=load_callback)
