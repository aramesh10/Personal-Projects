import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, writers
import math
import numpy as np

# If running through terminal, matplotlib may not animate the double pendulum properly. For now, running through VS Code is a work around for this issue
print("If running through terminal, matplotlib may not animate the double pendulum properly. For now, running through VS Code is a work around for this issue")

class Double_Pendulum:
    def __init__(self, theta1, theta2, l, m, v1, v2, g, delta_t): #theta in degrees
        self.l = l
        self.theta1 = math.radians(theta1)
        self.theta2 = math.radians(theta2)
        self.x1 = math.sin(self.theta1)*self.l
        self.y1 = math.cos(self.theta1)*self.l
        self.x2 = self.x1+math.sin(self.theta2)*self.l
        self.y2 = self.y1-math.cos(self.theta2)*self.l
        self.m = m
        self.v1 = v1
        self.v2 = v2
        self.g = g
        self.t = 0
        self.delta_t = delta_t

    def calc_vel(self):
        p1 = self.m*self.v1
        p2 = self.m*self.v2

        delta_theta1 = (6/(self.m*(self.l**2)))*(2*p1-3*math.cos(self.theta1-self.theta2)*p2)/(16-9*((math.cos(self.theta1-self.theta2))**2))
        delta_theta2 = (6/(self.m*(self.l**2)))*(8*p2-3*math.cos(self.theta1-self.theta2)*p1)/(16-9*((math.cos(self.theta1-self.theta2))**2))
        delta_p1 = (-1/2*self.m*(self.l**2))*(delta_theta1*delta_theta2*math.sin(self.theta1-self.theta2)+3*(self.g/self.l)*(math.sin(self.theta1)))
        delta_p2 = (-1/2*self.m*(self.l**2))*(-1*delta_theta1*delta_theta2*math.sin(self.theta1-self.theta2)+(self.g/self.l)*(math.sin(self.theta2)))

        self.theta1 = self.theta1 + delta_theta1 * self.delta_t
        self.theta2 = self.theta2 + delta_theta2 * self.delta_t
        self.v1 = self.v1 + (delta_p1/self.m)* self.delta_t
        self.v2 = self.v2 + (delta_p2/self.m)* self.delta_t

    def calc_pos(self):
        self.x1 = math.sin(self.theta1)*self.l
        self.y1 = -1*math.cos(self.theta1)*self.l
        self.x2 = self.x1+math.sin(self.theta2)*self.l
        self.y2 = self.y1-math.cos(self.theta2)*self.l

dp1 = Double_Pendulum(135,135,4,1,0,0,9.81,.01) #theta1, theta2, l, m, v1, v2, g, delta_t
dp2 = Double_Pendulum(-135,-135,4,1,0,0,9.81,.01)
fig = plt.figure()
ax = plt.axes(xlim=((-2 - dp1.l)*2, (2 + dp1.l)*2), ylim=((-2 - dp1.l)*2, (2 + dp1.l)*2))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
line1, = plt.plot([],[],ms=2, color='#0021A5')
line2, = plt.plot([],[],ms=2, color='#0021A5')
line3, = plt.plot([],[],ms=2, color='#FA4616')
line4, = plt.plot([],[],ms=2, color='#FA4616')

def animate(i):
    line1.set_data([0,dp1.x1],[0,dp1.y1])
    line2.set_data([dp1.x1,dp1.x2],[dp1.y1,dp1.y2])
    line3.set_data([0,dp2.x1],[0,dp2.y1])
    line4.set_data([dp2.x1,dp2.x2],[dp2.y1,dp2.y2])
    dp1.calc_vel()
    dp1.calc_pos()
    dp2.calc_vel()
    dp2.calc_pos()
    return line1, line2, line3, line4

ani = FuncAnimation(fig, animate, frames=60, interval=4)
plt.show()