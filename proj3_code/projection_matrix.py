import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time

def objective_func(x, **kwargs):
    """
        Calculates the difference in image (pixel coordinates) and returns 
        it as a 2*n_points vector

        Args: 
        -        x: numpy array of 11 parameters of P in vector form 
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                    retrieve these 2D and 3D points and then use them to compute 
                    the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between 
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """
    
    ##############################
    P = np.append(x, [1]).reshape((3, 4))

    projections = projection(P, kwargs['pts3d']) # PXw
    pts_2d = kwargs['pts2d'] # x
    diffs = []

    for i in range(pts_2d.shape[0]):
        diff_vector = projections[i] - pts_2d[i]
        diffs.append(diff_vector[0])
        diffs.append(diff_vector[1])

    diff = np.array(diffs)
    ##############################
      
    return diff

def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """
    
    ##############################
    rows = points_3d.shape[0]
    projected_points_2d = np.empty((rows, 2))
    for i in range(rows):
        projected_points_2d[i] = get2dCoords(P, points_3d[i])
    ##############################
    
    return projected_points_2d

def get2dCoords(P: np.ndarray, point_3d: np.ndarray):
    X = point_3d[0]
    Y = point_3d[1]
    Z = point_3d[2]

    p11 = P[0][0]
    p12 = P[0][1]
    p13 = P[0][2]
    p14 = P[0][3]

    p21 = P[1][0]
    p22 = P[1][1]
    p23 = P[1][2]
    p24 = P[1][3]

    p31 = P[2][0]
    p32 = P[2][1]
    p33 = P[2][2]
    p34 = P[2][3]

    x = ((p11*X) + (p12*Y) + (p13*Z) + p14) / ((p31*X) + (p32*Y) + (p33*Z) + p34)
    y = ((p21*X) + (p22*Y) + (p23*Z) + p24) / ((p31*X) + (p32*Y) + (p33*Z) + p34)

    return np.array([x, y])


def estimate_camera_matrix(pts2d: np.ndarray, 
                           pts3d: np.ndarray, 
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 
            
              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.
              
              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    start_time = time.time()
    
    ##############################
    initial_P = np.delete(initial_guess.reshape(12), 11)
    kwargs = {"pts2d": pts2d, "pts3d": pts3d}

    res = least_squares(
        objective_func, initial_P, 
        method='lm', 
        verbose=2, 
        max_nfev=50000,
        kwargs=kwargs
    )

    M = res.x
    M = np.reshape(np.append(M, [1]), (3, 4))
    ##############################
    
    print("Time since optimization start", time.time() - start_time)

    return M

def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix
        
        Args:
        -  P: 3x4 numpy array projection matrix
        
        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    
    ##############################
    K, R = rq(P[:, 0:3])
    ##############################
    
    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray, 
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    ##############################
    K_inverse = np.linalg.inv(K)
    cRw_I = np.matmul(K_inverse, P)

    R_inverse = np.linalg.inv(R_T)
    I_ctw = np.matmul(R_inverse, cRw_I)

    wtc = I_ctw[:, 3] * -1

    cc = np.reshape(wtc, 3)
    ##############################

    return cc






