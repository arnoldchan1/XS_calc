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
exp = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_SAD'][0,:]), S_err = np.squeeze(h5['s_SAD_err'][0,:]))

# Create Universe object with dcd files
U = mda.Universe("1l2y_wb_ions.psf", "1l2y_7_rest2.dcd")
# early traj slice for finding c1, c2
traj_early = Trajectory_slice(U, selection="protein", frame_min=0, frame_max=300)

c1_grid = np.arange(0.95, 1.051, 0.005)
c2_grid = np.arange(-2.0, 4.01, 0.05)
n_processes = 52 # choose number of processes for pool

c1_best, c2_best, chi2_grid = c_search(traj_early, mea, exp, c1_grid, c2_grid, n_processes)

# Create new environment with best c1, c2
env_best = Environment(c1=c1_best, c2=c2_best)

del traj_early

# Create new traj for final XS calc
traj = Trajectory_slice(U, selection="protein", frame_step=10)

# final traj calc
XS = traj_calc(traj, env_best, mea, "frame_XS_calc_fast", n_processes)

XS_avg = np.mean(XS, axis=0)
chi2_fun = lambda x: np.sqrt(np.mean( ((XS_avg - (exp.S_exp * x[0] + x[1]))/exp.S_err)**2 ))
res = minimize(chi2_fun, (1,0), method='CG')
XS_scaled = (XS-res.x[1])/res.x[0]

# Saving the outputs
hf = h5py.File('XS_output_20220503.h5', 'w')
hf.create_dataset('q', data=mea.q)
hf.create_dataset('XS', data=XS_scaled)
hf.create_dataset('c1_best', data=c1_best)
hf.create_dataset('c2_best', data=c2_best)
hf.create_dataset('chi2_grid', data=chi2_grid)
hf.close()

toc = time.time()
print('Job finished in {:.4f} seconds'.format(toc-tic))