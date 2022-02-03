

import MDAnalysis as mda
import numpy as np

class Molecule: 
# Invariant part of the trajectory, for example atoms' vdW radii

    def __init__(self, sel):
        FF_ref = load_form_factors()
        vdW_ref = load_vdW_radii()
        self.elements = None 
        self.vdW = None # Get this from vdW_ref
        self.FF = None # Get this from FF_ref. Actual form factors 

class Frame: 
# Variable part of the trajectory, for example atoms' coordinates and therefore SASA

    def __init__(self, xyz):
        self.xyz = xyz
        self.SASA = None #np.zeros( ... )
        self.isSASAcalculated = False # Until we have the SASA calculated

    def SASA_calc(self, env, force_recalc=False):
        if not self.isSASAcalculated or force_recalc:
            
            # Computes SASA for each atom and store in the mol.SASA (this will be a fraction from 0 to 1)

            self.isSASAcalculated = True # This avoids recalculation of SASA, which is time consuming
        else:
            return # Do nothing and return

class Trajectory:
    def __init__(self, U, selection=None):
        # Take in the "Universe" object (just the molecule) and create these things 
        sel = U.select_atoms(selection)
        self.Molecule = Molecule(sel)
        self.Frames = []
        for ts in U.trajectory:
            self.Frames.append(Frame(sel.positions))
        


class Environment: # Sample environment related items
    def __init__(self, c1=1.0, 
                       c2=2.0, 
                       r_sol=1.8, 
                       r_m=1.62, 
                       rho=0.334):
        self.c1 = c1
        self.c2 = c2
        self.r_sol = r_sol
        self.r_m = r_m      # This is the term used in c1 term
        self.rho = rho      # This is the solvent electron density. 0.334 is that of water at 20 C.

class Measurement: # Measurement related items, for now just the q vector
    def __init__(self, q):
        self.q = q

class Experiment: # Experimental data
    def __init__(self, q, S_exp, S_err):
        self.q = q
        self.S_exp = S_exp
        self.S_err = S_err
    def fit_to_experiment(self, XS):
        pass


def load_form_factors(flavor='WaasKirf'):
    # Return a dictionary containing the 11 coefficients from the WaasKirf table for each atom type 
    pass

def load_vdW_radii():
    # Return a dictionary containing the vdW radius for each atom type 
    pass


def FF_calc(frame, env, mea):
    # Calculate f(q, c1, c2) based on the q in measurement (mea), c1 and c2 in environment (env), and SASA in the frame
    return FF_q
    
def frame_XS_calc(frame, env, mea, ignoreSASA=False): # Calculate the X-ray scattering of a frame
    if not ignoreSASA:
        # Get the SASA calculated if not done
        frame.SASA_calc(env)

    # Calculate adjusted form factors as a table.
    FF_q = FF_calc(frame, env, mea)

    # Calculate scattering signal XS
    ...

    return XS

def traj_calc(traj, env, mea, ignoreSASA=False): # Calculate the X-ray scattering of an entire trajectory
    
    XS = []

    for frame in traj.Frames:
    
        # Calculate sturcture based on Debye formula and modified form factors
        XS.append(frame_XS_calc(frame, env, mea, ignoreSASA))

    XS = np.array(XS)

    return XS


if __name__ == "__main__":
    # This would be a typical use case
    U = mda.Universe('data/myprotein.pdb')
    traj = Trajectory(U, selection='protein and not symbol H')
    env = Environment()
    mea = Measurement(q = np.linspace(0.03, 0.8, num=200))
    XS = traj_calc(traj, env, mea)

    # Do something with XS. E.g. fitting etc.


