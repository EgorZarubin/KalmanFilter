'''

Description of math model

x - vector of etimated parameters

z - vector of observations

H - matrix to connect X and Z

F - matrics for math model ( connect n and n-1 values)

B - control matrix

u - control vector

P - covariation error

R - covariation noise for observations

Q - process noise covariation

K - Kalman coefficient

equation: 

z = H*x + r

x(n) = F * x(n-1) + B * u(n-1)



// t means transposed

Pk = F*Pk-1*Ft + Q

Kk = Pk * Ht / ( H* Pk * Ht + R' )

x(opt)(k) = x(k) + Kk * (z(k) - H*x(k))



P = (I - Kk*H)*Pk

'''



'''suppose ve have 1d movement with v an a'''

import numpy as np

import numpy.linalg as lalg

import matplotlib.pyplot as plt

import random as random



# settings for the model and noise
t = 1 

coord_noise = 1

vel_noise = 0.3

acc_noise = 0.5



obs_coord_noise = 100

obs_vel_noise = 5

obs_acc_noise = 2

F = np.matrix([[1, t, 0],

                [0, 1, t],

                [0, 0, 0]])



B = np.matrix([[0, 0, 0],

                [0, 0, 0],

                [0, 0, 1]])



H = np.matrix([[1, 0, 0],

                [0, 1, 0],

                [0, 0, 1]])



R = np.matrix([[obs_coord_noise, 0, 0],

                [0, obs_vel_noise, 0],

                [0, 0, obs_acc_noise]])

 

Q = np.array([[coord_noise, 0, 0],

               [0, vel_noise, 0],

               [0, 0, acc_noise]])



I = np.array([[1, 0, 0],

               [0, 1, 0],

               [0, 0, 1]])



# Kalman prediction function
def makePrediction(observation, xprev, P, dt, u):

    P = F * P * F.transpose() + Q

    K = (P * H.transpose()) * lalg.inv(H * P * H.transpose() + R)

    

    X = F * xprev + B * u

    X = X + K * (observation - H * X)

   

    P = (I - K * H) * P

    return [X, P, K]



#simulate movement
size = 150



U = [ np.matrix([[0], [0], [(10 * np.sin(np.pi * i / 4)) / (i % 104 + 1)]]) for i in range(size)]

U[100][2][0] = -10

U[101][2][0] = 0

U[102][2][0] = 0

U[103][2][0] = 0

U[104][2][0] = 0



realX = [np.matrix([[0], [0], [0]]) for i in range(size)]

for i in range(size - 1):

    realX[i + 1] = F * realX[i] + B * U[i] + np.matrix([[random.gauss(0,coord_noise)],

                                                       [random.gauss(0,vel_noise)],

                                                       [random.gauss(0,acc_noise)]])



observations = [np.matrix([[0], [0], [0]]) for i in range(size)]

for i in range(size):

    observations[i] = realX[i] + np.matrix([[random.gauss(0, obs_coord_noise)],

                                            [random.gauss(0, obs_vel_noise)],

                                            [random.gauss(0, obs_acc_noise)]])





############# start Kalman algorith,
predictions = [np.matrix([[0], [0], [0]]) for i in range(size)]

predictions[0] = observations[0]

error = R



for i in range(size - 1):

    [predictions[i + 1], error, K] = makePrediction(observations[i + 1], predictions[i], error, 1, U[i])



#plot results


coordinate = [float(realX[i][0][0]) for i in range(size)]

velocity = [float(realX[i][1][0]) for i in range(size)]

acceleration = [float(realX[i][2][0]) for i in range(size)]



observations_coord = [float(observations[i][0][0]) for i in range(size)]

observations_vel = [float(observations[i][1][0]) for i in range(size)]

observations_acc = [float(observations[i][2][0]) for i in range(size)]



predictions_coord = [float(predictions[i][0][0]) for i in range(size)]

predictions_vel = [float(predictions[i][1][0]) for i in range(size)]

predictions_acc = [float(predictions[i][2][0]) for i in range(size)]



time = [i for i in range(size)]



f, ax = plt.subplots(3, sharex = True)





ax[0].set_title('time')



ax[0].plot(time, observations_coord)

ax[0].plot(time, predictions_coord)

ax[0].plot(time, coordinate, '--')

ax[0].grid(True)



ax[1].plot(time, observations_vel)

ax[1].plot(time, predictions_vel)

ax[1].plot(time, velocity, '--') 

ax[1].grid(True)



ax[2].plot(time, observations_acc)

ax[2].plot(time, predictions_acc)

ax[2].plot(time, acceleration, '--')

ax[2].grid(True)



plt.show()
