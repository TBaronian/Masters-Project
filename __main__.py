import scipy.ndimage as nd  # Using ndimage for faster stencil convolutions
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import logging

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter_short = logging.Formatter("%(message)s")
logger.addHandler(ch)
ch.setFormatter(formatter_short)

# --- Initialize Matrices ---
Y_SIZE = 64; X_SIZE = 64; Z_SIZE = 64
PARTICLE_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4))
PARTICLE_MATRIX_OLD = PARTICLE_MATRIX.copy()
FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4))
FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX.copy()  # Use .copy() to avoid shared reference

# --- Calculation Matrices ---
FOUR_POTENTIAL_DOT_MATRIX = np.zeros_like(FOUR_POTENTIAL_MATRIX)
FOUR_POTENTIAL_DOUBLE_DOT = np.zeros_like(FOUR_POTENTIAL_MATRIX)
PROPER_VELOCITY_MATRIX = np.zeros_like(PARTICLE_MATRIX)
LAPLACIAN_FOUR_POTENTIAL = np.zeros_like(FOUR_POTENTIAL_MATRIX)
DIV_FOUR_POTENTIAL_MATRIX = np.zeros((X_SIZE, Y_SIZE, Z_SIZE, 4, 3))
U_MU_DOT_MATRIX = np.zeros_like(PARTICLE_MATRIX)


def add_charge(q, x=int(X_SIZE/2), y=int(Y_SIZE/2), z=int(Z_SIZE/2), v_x = 0.0, v_y = 0.0, v_z = 0.0):
    global PARTICLE_MATRIX
    PARTICLE_MATRIX[x, y, z] = [q, v_x, v_y, v_z]

def calculate_a_double_dot():
    global FOUR_POTENTIAL_DOUBLE_DOT, PARTICLE_MATRIX, PROPER_VELOCITY_MATRIX, LAPLACIAN_FOUR_POTENTIAL
    for i in range(4):
        source_term = MU_0 * PARTICLE_MATRIX[:, :, :, 0] * PROPER_VELOCITY_MATRIX[:, :, :, i]
        FOUR_POTENTIAL_DOUBLE_DOT[:, :, :, i] = (SPEED_OF_LIGHT**2) * (LAPLACIAN_FOUR_POTENTIAL[:, :, :, i] + source_term)

def calculate_laplacian_of_a():
    global LAPLACIAN_FOUR_POTENTIAL, laplacian_mask, FOUR_POTENTIAL_MATRIX
    for i in range(4):
        # Division by DELTA_X**2 scales the spatial step properly if DELTA_X is changed
        LAPLACIAN_FOUR_POTENTIAL[:, :, :, i] = nd.convolve(
            FOUR_POTENTIAL_MATRIX[:, :, :, i], laplacian_mask, mode="constant", cval=0.0
        ) / (DELTA_X**2)
        
def calculate_div_of_a():
    global DIV_FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_MATRIX, div_x_mask, div_y_mask, div_z_mask
    for i in range(4):
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 0] = nd.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_x_mask, mode="constant", cval=0.0) / DELTA_X
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 1] = nd.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_y_mask, mode="constant", cval=0.0) / DELTA_X
        DIV_FOUR_POTENTIAL_MATRIX[:, :, :, i, 2] = nd.convolve(FOUR_POTENTIAL_MATRIX[:, :, :, i], div_z_mask, mode="constant", cval=0.0) / DELTA_X

def calculate_a_dot():
    global FOUR_POTENTIAL_DOT_MATRIX, FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_DOUBLE_DOT
    calculate_a_double_dot()
    FOUR_POTENTIAL_DOT_MATRIX = (FOUR_POTENTIAL_MATRIX - FOUR_POTENTIAL_MATRIX_OLD)/DELTA_T + DELTA_T * FOUR_POTENTIAL_DOUBLE_DOT

def calculate_u_mu():
    global PROPER_VELOCITY_MATRIX, PARTICLE_MATRIX
    # Prevent division by zero if velocity equals or exceeds SPEED_OF_LIGHT
    v_sq = PARTICLE_MATRIX[:, :, :, 1]**2 + PARTICLE_MATRIX[:, :, :, 2]**2 + PARTICLE_MATRIX[:, :, :, 3]**2
    denom = np.sqrt(np.maximum(1e-9, 1 - v_sq / (SPEED_OF_LIGHT**2)))
    gamma = (PARTICLE_MATRIX[..., 0] != 0) * (1.0 / denom)
    
    PROPER_VELOCITY_MATRIX[..., 0] = SPEED_OF_LIGHT * gamma
    PROPER_VELOCITY_MATRIX[..., 1:] = gamma[..., None] * PARTICLE_MATRIX[..., 1:]

def update_a():
    global FOUR_POTENTIAL_MATRIX, FOUR_POTENTIAL_MATRIX_OLD
    calculate_u_mu()
    calculate_laplacian_of_a()
    calculate_div_of_a()
    calculate_a_dot()
    
    # Update the historical states using copies to avoid variable reference binding
    FOUR_POTENTIAL_MATRIX_OLD = FOUR_POTENTIAL_MATRIX.copy()
    FOUR_POTENTIAL_MATRIX = FOUR_POTENTIAL_MATRIX + DELTA_T * FOUR_POTENTIAL_DOT_MATRIX

def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Simulation")

    plt.ion()  # Turn on interactive plotting mode
    fig, (ax_particle, ax_vector) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Place a positive test charge in the middle of the grid
    add_charge(q=255)
    
    # Initialize display plots with placeholders
    z_mid = int(Z_SIZE / 2)
    im_particle = ax_particle.imshow(PARTICLE_MATRIX[:, :, z_mid, 0], cmap='viridis', origin='lower')
    im_vector = ax_vector.imshow(FOUR_POTENTIAL_MATRIX[:, :, z_mid, 0], cmap='inferno', origin='lower')
    
    ax_particle.set_title("Particle Density (q)")
    ax_vector.set_title("Scalar Potential (A_0)")
    
    plt.colorbar(im_particle, ax=ax_particle, shrink=0.7)
    plt.colorbar(im_vector, ax=ax_vector, shrink=0.7)

    step = 0
    while plt.fignum_exists(fig.number):
        update_a()
        
        # Periodically update visualization to limit drawing overhead
        if step % 5 == 0:
            # Update data arrays instead of recreating the plot
            im_particle.set_data(PARTICLE_MATRIX[:, :, z_mid, 0])
            im_vector.set_data(FOUR_POTENTIAL_MATRIX[:, :, z_mid, 0])
            
            # Autoscale color limits to handle initial transient developments
            im_vector.set_clim(vmin=FOUR_POTENTIAL_MATRIX[:, :, z_mid, 0].min(), 
                               vmax=FOUR_POTENTIAL_MATRIX[:, :, z_mid, 0].max())
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(1e-5)
            
            mid_val_pot = FOUR_POTENTIAL_MATRIX[int(X_SIZE/2), int(Y_SIZE/2), int(Z_SIZE/2)]
            logger.info("Step %04d | Potential at Center: [%.3e, %.3e, %.3e, %.3e]", 
                        step, mid_val_pot[0], mid_val_pot[1], mid_val_pot[2], mid_val_pot[3])
            
        step += 1

if __name__ == "__main__":
    main()