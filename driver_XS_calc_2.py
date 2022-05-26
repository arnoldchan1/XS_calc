import importlib
import XS_calc
from XS_calc import *

import MDAnalysis as mda
from MDAnalysis import analysis
from scipy.optimize import minimize
import numpy as np
import h5py
import time

tic = time.time()

# Import experimental scattering curves
h5 = h5py.File("20220404_trpcage_reconstructed_saxs.h5")
# define measurement q from experimental q
mea = Measurement(q=np.squeeze(h5['q_SAD'][:]))

# Create Experiment class objects for folded species
exp = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_unf'][:]), S_err = np.squeeze(h5['s_unf_err'][:]))

# Create Universe object with dcd files
U = mda.Universe("1l2y_wb.psf", "1l2y_7_rest2.dcd")
traj = Trajectory_slice(U, selection="protein", frame_step=5)

env = Environment(c1=1.05, c2=0.4) # previously determined c1, c2

XS = traj_calc(traj, env, mea, "frame_XS_calc_fast", n_processes=52)

# Scaling the mean of the XS to tr unfolded curve
XS_avg = np.mean(XS, axis=0)
chi2_fun = lambda x: np.sqrt(np.mean( ((XS_avg - (exp.S_exp * x[0] + x[1]))/exp.S_err)**2 ))
res = minimize(chi2_fun, (1,0), method='CG')
XS_scaled = (XS-res.x[1])/res.x[0]


# Saving the outputs
hf = h5py.File('XS_output_20220511.h5', 'w')
hf.create_dataset('q', data=mea.q)
hf.create_dataset('XS', data=XS)
hf.create_dataset('XS_scaled', data=XS_scaled)
hf.create_dataset('sf', data=res.x)
hf.create_dataset('c1_best', data=c1_best)
hf.create_dataset('c2_best', data=c2_best)
hf.create_dataset('chi2_grid', data=chi2_grid)
hf.close()


toc = time.time()
print('Job finished in {:.3f} seconds'.format(toc-tic))

