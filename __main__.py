import scipy.signal as sp
import matplotlib.pyplot as plt
import numpy as np
from constants import *


# --- Initialize Matrices ---

Y_SIZE = 64; X_SIZE = 64; Z_SIZE = 64
PARTICLE_MATRIX = np.empty((X_SIZE, Y_SIZE, Z_SIZE, 4))
PARTICLE_MATRIX_OLD = PARTICLE_MATRIX
FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4))
FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX

# --- Calculation Matrices --- #

FOUR_POTENTIAL_DOT_MATRIX = np.zeros_like(FOUR_POTENTIAL_MATRIX)
FOUR_POTENTIAL_DOUBLE_DOT = np.zeros_like(FOUR_POTENTIAL_MATRIX)
PROPER_VELOCITY_MATRIX = np.zeros_like(PARTICLE_MATRIX)
LAPLACIAN_FOUR_POTENTIAL = np.zeros_like(FOUR_POTENTIAL_MATRIX)
DIV_FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4, 3))
U_MU_DOT_MATRIX = np.zeros_like(PARTICLE_MATRIX)


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
    global PARTICLE_MATRIX
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


def calculate_a_double_dot():
    
    global FOUR_POTENTIAL_DOT_MATRIX, FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_DOUBLE_DOT, PARTICLE_MATRIX

    
    for i in range(4):
        FOUR_POTENTIAL_DOUBLE_DOT[:, :, :, i] = (PARTICLE_MATRIX[:, :, :, 0] != 0.0) * MU_0 * PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, i] + \
                                                (PARTICLE_MATRIX[:, :, :, 0] == 0.0) * SPEED_OF_LIGHT**2 * LAPLACIAN_FOUR_POTENTIAL[:, :, :, i]
    return


def calculate_u_mu_dot():
    
    
    U_MU_DOT_MATRIX[:, :, :, 0] = PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 0] * FOUR_POTENTIAL_DOT_MATRIX[:, :, :, 0] - \
                                  PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 1] * FOUR_POTENTIAL_DOT_MATRIX[:, :, :, 1] - \
                                  PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 2] * FOUR_POTENTIAL_DOT_MATRIX[:, :, :, 2] - \
                                  PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 3] * FOUR_POTENTIAL_DOT_MATRIX[:, :, :, 3]
     
    for i in range(3):                              
        U_MU_DOT_MATRIX[:, :, :, i] = PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 0] * DIV_FOUR_POTENTIAL_MATRIX[:, :, :, 0, i] - \
                                    PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 1] * DIV_FOUR_POTENTIAL_MATRIX[:, :, :, 1, i] - \
                                    PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 2] * DIV_FOUR_POTENTIAL_MATRIX[:, :, :, 2, i] - \
                                    PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, 3] * DIV_FOUR_POTENTIAL_MATRIX[:, :, :, 3, i]
    return
        
def calculate_laplacian_of_a():
    
    global LAPLACIAN_FOUR_POTENTIAL, laplacian_mask, FOUR_POTENTIAL_MATRIX
    
    for i in range(4):
        LAPLACIAN_FOUR_POTENTIAL[:, :, :, i] = sp.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], laplacian_mask, mode="same")
        
def calculate_div_of_a():
    
    global DIV_FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_MATRIX, div_x_mask, div_y_mask, div_z_mask
    
    for i in range(4):
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 0] = sp.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_x_mask, mode="same")
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 1] = sp.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_y_mask, mode="same")
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 2] = sp.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_z_mask, mode="same")
        
    return

def calculate_a_dot():
    
    global FOUR_POTENTIAL_DOT_MATRIX, FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_DOUBLE_DOT
    
    calculate_a_double_dot()
    FOUR_POTENTIAL_DOT_MATRIX = FOUR_POTENTIAL_MATRIX - FOUR_POTENTIAL_MATRIX_OLD + DELTA_T * FOUR_POTENTIAL_DOUBLE_DOT
    return

def calculate_u_mu():
    
    global PROPER_VELOCITY_MATRIX, PARTICLE_MATRIX
    gamma =  (PARTICLE_MATRIX[..., 0] != 0) * 1/np.sqrt(1 - (PARTICLE_MATRIX[:, :, :, 1]**2 + PARTICLE_MATRIX[:, :, :, 2]**2 + PARTICLE_MATRIX[:, :, :, 3]**2)/SPEED_OF_LIGHT**2)
    PROPER_VELOCITY_MATRIX[..., 0] = SPEED_OF_LIGHT * gamma [:, :, :]
    PROPER_VELOCITY_MATRIX[..., 1:] = gamma[..., None] * PARTICLE_MATRIX[..., 1:]
    return

    
def update_a():
    
    global FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_MATRIX_OLD, FOUR_POTENTIAL_DOT_MATRIX, PROPER_VELOCITY_MATRIX
    calculate_u_mu()
    calculate_laplacian_of_a()
    calculate_div_of_a()
    calculate_a_dot()
    
    FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX
    FOUR_POTENTIAL_MATRIX = FOUR_POTENTIAL_MATRIX + DELTA_T * FOUR_POTENTIAL_DOT_MATRIX
    
    return
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
    # input("Press Any Key to Continute...")
    
def get_u_mu(*particle_entry):
    """Get the proper 4-velocity of a particle at a particular position

    Args"
        particle_entry
    Returns:
        collection: (u_0, u_1, u_2, u_3) The 4-velocity of the particle
    """
    
    try:
        q, v_x, v_y, v_z = particle_entry
    except:
        print("get_u_mu: A problem with particle entry")
    
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
    add_charge(q = 255)
    
    
    # render(ax_vector, FOUR_POTENTIAL_MATRIX[:, :, :, 0].astype(np.int32))
    # add_charge(q=255, v_x = 150/np.sqrt(2), v_y = 150/np.sqrt(2), v_z = 0)
    # print(get_u_mu(128, 128, 128))
    # render(ax_particle, PARTICLE_MATRIX[:, :, :, 0].astype(np.int32))
    # move_charge(128,128,128)
    # render(ax_particle, PARTICLE_MATRIX[:, :, :, 0].astype(np.int32))
    
    key_stroke = ""
    while(True):
        update_a()
        render(ax_vector, FOUR_POTENTIAL_MATRIX[:,:,:,0].astype(np.int32))
        key_stroke = input("Proceed? (Y/n)")
        if key_stroke == "Y":
            continue
        elif key_stroke == "n":
            break
        else:
            print("error")
            return 1


if __name__ == "__main__":
    main()