# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import scipy


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    w_cap = np.array([[0,-angular_velocity[2] , angular_velocity[1]],
                        [angular_velocity[2], 0, -angular_velocity[0]],
                        [-angular_velocity[1],angular_velocity[0],0]])
    R_12 = scipy.linalg.expm(w_cap*dt) # R_12 = np.identity(3) + np.sin(dt)*w_cap + (1-np.cos(dt))*w_cap*w_cap
    e_x = np.array([[1],[0],[0]])
    R_1k =  initial_rotation * Rotation.from_matrix(R_12) 
    g_prime = R_1k.as_matrix().dot(linear_acceleration.reshape((3,1)))
    g_prime = g_prime/norm(g_prime)
    #Rotation axis, w_acc to be parallel to the cross product g′ x ex 
    w_acc = np.cross(g_prime.T, e_x.T)
    w_acc = w_acc/norm(w_acc)
    g_prime = np.squeeze(np.asarray(g_prime))
    e_x = np.squeeze(np.asarray(e_x))
    #Chose the rotation angle to be the angle between the g’ and ex 
    angle = np.arccos(np.dot(g_prime,e_x))
    vec = np.sin(angle/2)*w_acc
    scalar = np.cos(angle/2)
    #The angle and axis was used to find the corresponding quaternion
    delta_q_acc = np.array(np.hstack((vec[0], scalar)))
    #Using the method from the paper [1]:
    # q0 = 0
    # q1 = g_prime[2]/(np.sqrt(2*(g_prime[0]+1)))
    # q2 = -g_prime[1]/(np.sqrt(2*(g_prime[0]+1)))
    # q3 = np.sqrt((g_prime[0] + 1)/2)
    # delta_q_acc = np.array([q0, q1, q2, q3])
    q_I = np.array([0, 0, 0, 1])
    e_m = np.absolute(np.linalg.norm(linear_acceleration)/9.81-1)
    if e_m >= 0.2:
        alpha = 0
    elif 0.1<e_m<0.2:
        alpha = 2 - 10*(e_m)
    else: #e_m <= 0.1:
        alpha = 1
    delta_q_acc_prime = (1-alpha)*q_I + alpha*delta_q_acc
    delta_q_acc_prime = delta_q_acc_prime/norm(delta_q_acc_prime)
    Rot = Rotation.from_quat(delta_q_acc_prime)* R_1k
    return Rot