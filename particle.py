import numpy as np
from constants import *

class Particle(object):
    
    def __init__(self, charge = None, v_x = 0.0, v_y = 0.0, v_z = 0.0):
        self._v = np.array((v_x, v_y, v_z), dtype = np.float32)
        self._q = charge
        
    def __int__(self):
        if self._q is None:
            return 0
        else:
            return int(self._q)
    
    @property
    def q(self):
        return self._q
    
    @property
    def v(self):
        return self._v
    
    @q.setter
    def q(self, charge):
        self._q = charge
        
    @v.setter
    def v(self, v_arr):
        self._v = v_arr
        
    @property
    def speed(self):
        return np.sqrt(self.v[0]^2 + self.v[1]^2 + self.v[2]^2)
    
    @property
    def u_mu (self):
        gamma = 1/np.sqrt(1-self.speed()^2/SPEED_OF_LIGHT^2)
        return np.array((gamma*SPEED_OF_LIGHT, gamma*self.v[0], gamma*self.v[1] + gamma*self.v[2]))
    