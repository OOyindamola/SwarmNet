#!/usr/bin/env python
import math
import numpy as np


import time

class VelocityController:
    def __init__(self, env, waypoints, distThreshold,
                    kp, kd, ki):

        #Robot States
        self.env = env
        self.x = 0
        self.y = 0
        self.theta = 0
        self.linear_vel = 0.1
        self.angular_vel = 0
        self.radius = .5


        #Gain Parameters
        self.kp = kp
        self.kd = kd
        self.ki = ki

        #PID parameters
        self.error = 99999
        self.derivative = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prevTime = time.time()

        self.distThreshold = distThreshold
        self.maxMotorVel = env.max_accel #rospy.get_param('~maxMotorVel', 1)
        self.minMotorVel = -env.max_accel #rospy.get_param('~minMotorVel', -1)
        self.motorVel = 0


        #waypoint
        self.waypoints = waypoints
        self.waypoint_size= self.waypoints.shape[0]

    def saturate(self, input, delta):
        if(input > self.maxMotorVel):
            self.motorVel = self.maxMotorVel
            self.integral = self.integral - (self.error * delta)


        elif(input < self.minMotorVel):
            self.motorVel = self.minMotorVel
            self.integral = self.integral - (self.error * delta)
        else:
            self.motorVel = input

        print("Saturated Input: ", self.motorVel)
        print("Max Input: ", self.maxMotorVel)
        print("Min Input: ", self.minMotorVel)


    def PID(self):
        delta = (time.time() - self.prev_time).to_sec()
        #self.error = self.commandedVelocity - self.angular_vel
        self.derivative = (self.error - self.prev_error)/delta
        self.integral = self.integral + self.error*delta
        self.prev_error = self.error
        omega = self.kp* self.error + self.kd * self.derivative + self.ki * self.integral
        print("Omega: ", omega)
        self.saturate(omega, delta)



    def run(self):
        done_ = False
        self.prev_time = time.time()

        next_state, reward, done = 0,0,0
        while not done_:
            next_state, reward, done, done_  = self.executeController()


        return next_state, reward, done

    def executeController(self):
        #Goal - waypoint
        wp_x = self.waypoints[:,0]
        wp_y = self.waypoints[:, 1]

        goal = np.array([wp_x, wp_y])


        ##Current State
        self.x = self.env.x[:,0]
        self.y = self.env.x[:,1]
        self.theta = self.env.theta

        # print(self.y)
        print(self.x)
        print(wp_x)
        print(self.theta)
        # print(wp_y - self.y)

        current_state = np.array([self.x, self.y])


        #Desired Heading and get current heading
        self.desired_heading = np.arctan2((wp_y - self.y),(wp_x - self.x))

        print("Current State: ", np.array([self.x, self.y, self.theta]))
        print("Desired Waypoint: ", np.array([wp_x, wp_y, self.desired_heading]))

        self.error = (self.desired_heading - self.theta)

        print("Error: ", self.error)

        self.distError = np.linalg.norm(goal - current_state)
        print("Dist Error: ", self.distError)
        next_state, reward, done = 0,0,0

        if(self.distError > self.distThreshold):
            self.PID()
            action = self.motorVel
            next_state, reward, done, _ = env.step(action)
            return next_state, reward, done, False

        else:
            print("Waypoint Reached with Distance error: ", self.distError)
            return next_state, reward, done, True
