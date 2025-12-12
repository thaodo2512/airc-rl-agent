try:
    from jetbot import Robot
except ImportError:
    class Robot:pass

class RobotController():

    MAX_MOTORLIMIT = 1.0
    MIN_MOTORLIMIT = -1.0

    def __init__(self):
        self.robot = Robot()
        self.robot.left_motor.alpha = -1.0  # Invert polarity
        self.robot.right_motor.alpha = -1.0  # Invert polarity


    def action(self, steering, throttle):
        steering = float(steering)
        throttle = float(throttle)
        self.robot.left_motor.value = max(min(throttle + steering, self.MAX_MOTORLIMIT), self.MIN_MOTORLIMIT)
        self.robot.right_motor.value = max(min(throttle - steering, self.MAX_MOTORLIMIT), self.MIN_MOTORLIMIT)
