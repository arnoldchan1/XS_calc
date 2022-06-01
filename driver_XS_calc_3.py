import importlib
import XS_calc
from XS_calc import *

import MDAnalysis as mda
from MDAnalysis import analysis
import numpy as np
import h5py
import time
import pickle
from itertools import product

tic = time.time()

# Import experimental scattering curves
h5 = h5py.File("20220404_trpcage_reconstructed_saxs.h5")
# define measurement q from experimental q
mea = Measurement(q=np.squeeze(h5['q_SAD'][:]))

# Create Universe object with dcd files
U = mda.Universe("1l2y_wb.psf", "1l2y_7_rest2.dcd")
traj = Trajectory_slice(U, selection="protein")

# Create c1c2 grid
c1_grid = np.arange(0.98, 1.051, 0.01)
c2_grid = np.arange(0.0, 4.01, 0.2)
c1c2_product = list(product(c1_grid, c2_grid))

# traj_calc on all c1,c2 pairs
XS_pool = {}
for c1c2 in c1c2_product:
    print(f'Condition: {c1c2}', end=' ')
    c1, c2 = c1c2
    env = Environment(c1=c1, c2=c2)
    XS_pool[c1c2] = traj_calc(traj, env, mea, method='frame_XS_calc_fast', n_processes=52)

# Save to a hdf5
hf = h5py.File('1l2y_REST2_XS_20220527.h5', 'w')
hf.create_dataset('q', data=mea.q)
hf.create_dataset('XS_pool', data=XS_pool)
hf.close()

# Save to a pickle
pickle.dump(XS_pool, open('1l2y_REST2_XS_20220601.pkl', 'wb'))

toc = time.time()
print('Job finished in {:.3f} seconds'.format(toc-tic))
