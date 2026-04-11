import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from constants import *


# --- Initialize Matrices ---

Y_SIZE = 256; X_SIZE = 256; Z_SIZE = 256
PARTICLE_MATRIX = np.empty((X_SIZE, Y_SIZE, Z_SIZE, 4))
PARTICLE_MATRIX_OLD = PARTICLE_MATRIX
FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4))
FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX



def add_charge(q, x=int(X_SIZE/2), y=int(Y_SIZE/2), z=int(Z_SIZE/2), v_x = 0.0, v_y = 0.0, v_z = 0.0):
    """Adds a charged particle in a position in PARTICLE_MATRIX

    Args:
        q (float): charge
        x (int, optional): x-position. Defaults to int(X_SIZE/2).
        y (int, optional): y-position. Defaults to int(Y_SIZE/2).
        z (int, optional): z-position. Defaults to int(Z_SIZE/2).
        v_x (float, optional): lab-frame speed in x direction. Defaults to 0.0.
        v_y (float, optional): lab-frame speed in y direction. Defaults to 0.0.
        v_z (float, optional): lab-frame speed in z-direction. Defaults to 0.0.
    """
    PARTICLE_MATRIX[x, y, z] = [q, v_x, v_y, v_z]
    return

def move_charge(x, y, z):
    """Moves a charged particle, which is initially at (x, y, z)
    The movement updates the PARTICLE_MATRIX and PARTICLE_MATRIX_OLD
    TODO: add resolution of conflicting particles
    
    Args:
        x (int): x - position of the particle
        y (int): y - position of the particle
        z (int): z - position of the particle
    """
    global PARTICLE_MATRIX, PARTICLE_MATRIX_OLD
    if PARTICLE_MATRIX[x][y][z] is None:
        ...
    else:
        the_particle = PARTICLE_MATRIX[x][y][z]
        PARTICLE_MATRIX_OLD = PARTICLE_MATRIX
        PARTICLE_MATRIX[int (x + the_particle[1]*DELTA_T)][int(y + the_particle[2]*DELTA_T)][int(z + the_particle[3]*DELTA_T)] = the_particle
        PARTICLE_MATRIX[x][y][z] = np.array((0,0,0,0))

def render(ax, matrix, z_val=int(Z_SIZE/2)):
    """Renders a 2D slice of a 3D matrix of scalar values
    TODO: Main improvement with camera and interactives (much later)
    
    Args:
        ax (matplotlib.pyplot.Axes): The axis used for rendering
        matrix (np.ndarray): The 3D array to be rendered
        z_val (int): The vertical value to render. Defaults to int(Z_SIZE/2).
    """
    matrix_slice = matrix[:, :, z_val]
    ax.imshow(matrix_slice, cmap='gray')
    fig.show()
    input("Press Any Key to Continute...")
    
def get_u_mu(*pos):
    """Get the proper 4-velocity of a particle at a particular position

    Args"
        pos (collection) (x, y, z) The position to be evaluated
    Returns:
        collection: (u_0, u_1, u_2, u_3) The 4-velocity of the particle
    """
    
    x, y, z = pos
    v_x, v_y, v_z = PARTICLE_MATRIX[x, y, z, 1:]
    gamma = 1/np.sqrt(1 - (v_x**2 + v_y**2 + v_z**2)/SPEED_OF_LIGHT**2)
    u_0 = gamma * SPEED_OF_LIGHT
    u_1 = gamma * v_x; u_2 = gamma * v_y; u_3 = gamma * v_z
    return [u_0, u_1, u_2, u_3]

def main():
    """Main method for development.
    """
    global fig
    fig = plt.figure()
    ax_particle = fig.add_subplot(1, 2, 1)
    ax_vector = fig.add_subplot(1, 2, 2)
    
    render(ax_vector, FOUR_POTENTIAL_MATRIX[:, :, :, 0].astype(np.int32))
    add_charge(q=255, v_x = 150/np.sqrt(2), v_y = 150/np.sqrt(2), v_z = 0)
    print(get_u_mu(128, 128, 128))
    render(ax_particle, PARTICLE_MATRIX[:, :, :, 0].astype(np.int32))
    move_charge(128,128,128)
    render(ax_particle, PARTICLE_MATRIX[:, :, :, 0].astype(np.int32))


if __name__ == "__main__":
    main()