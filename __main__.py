import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from particle import *
from constants import *


# --- Initialize Matrices ---

Y_SIZE = 256; X_SIZE = 256; Z_SIZE = 256
PARTICLE_MATRIX = np.empty((X_SIZE, Y_SIZE, Z_SIZE, 4))
PARTICLE_MATRIX_OLD = PARTICLE_MATRIX
FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4))
FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX



def add_charge(q, x=int(X_SIZE/2), y=int(Y_SIZE/2), z=int(Z_SIZE/2)):
    PARTICLE_MATRIX[x, y, z] = q
    return

def move_charge(x, y, z):
    
    if PARTICLE_MATRIX[x][y][z] is None:
        ...
    else:
        the_particle = PARTICLE_MATRIX[x][y][z]
        PARTICLE_MATRIX_OLD = PARTICLE_MATRIX
        PARTICLE_MATRIX[int (x + the_particle[1]*DELTA_T)][int(y + the_particle[2]*DELTA_T)][int(z + the_particle[3]*DELTA_T)] = the_particle[0]
        PARTICLE_MATRIX[x][y][z] = np.array((0,0,0,0))

def render(ax, matrix, z_val=int(Z_SIZE/2)):
    matrix_slice = matrix[:][:][z_val]
    ax.imshow(matrix_slice, cmap='gray')
    fig.show()
    input("Press Any Key to Continute...")

def main():
    add_charge(q=255)
    global fig
    fig = plt.figure()
    ax_particle = fig.add_subplot(1, 2, 1)
    ax_vector = fig.add_subplot(1, 2, 2)
    render(ax_particle, PARTICLE_MATRIX.astype(np.int32))
    PARTICLE_MATRIX[128,128,128] = np.array((100,100,0))
    move_charge(128,128,128)
    render(ax_particle, PARTICLE_MATRIX.astype(np.int32))


if __name__ == "__main__":
    main()