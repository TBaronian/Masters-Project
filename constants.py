import numpy as np

SPEED_OF_LIGHT = 300
DELTA_T = 0.01
MU_0 = 4e-7*np.pi
DELTA_X = 1

laplacian_mask = np.zeros((3, 3, 3))
laplacian_mask[1,1,1] = -6
laplacian_mask[0,1,1] = laplacian_mask[2,1,1] = laplacian_mask[1,0,1] = laplacian_mask[1,2,1] = laplacian_mask[1,1,0] = laplacian_mask[1,1,2] = 1

div_x_mask = np.zeros((3, 3, 3))
div_x_mask[0,1,1] = -1/2; div_x_mask[2,1,1] = 1/2

div_y_mask = np.zeros((3, 3, 3))
div_y_mask[1,0,1] = -1/2; div_y_mask[1,2,1] = 1/2

div_z_mask = np.zeros((3, 3, 3))
div_z_mask[1,1,0] = -1/2; div_z_mask[1,1,2] = 1/2