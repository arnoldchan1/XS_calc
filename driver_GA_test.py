import XS_calc
from XS_calc import *

import XS_ga
from XS_ga import *

import numpy as np

import h5py
import pickle

import multiprocessing as mp
import time

from itertools import product


# Load in experimental reference curves
h5 = h5py.File("20220404_trpcage_reconstructed_saxs.h5")
# Create Experiment class objects for each species
F_st = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_SAD'][0,:]), S_err = np.squeeze(h5['s_SAD_err'][0,:]))
U_st = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_SAD'][1,:]), S_err = np.squeeze(h5['s_SAD_err'][1,:]))
I_tr = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_int'][:]), S_err = np.squeeze(h5['s_int_err'][:]))
U_tr = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_unf'][:]), S_err = np.squeeze(h5['s_unf_err'][:]))

# Make tuples of the c1/c2
c1_grid = np.arange(0.98, 1.051, 0.01)
c2_grid = np.arange(0.0, 4.01, 0.2)
c1c2_product = list(product(c1_grid, c2_grid))

# Load pickle file with XS_pool
XS_pool = pickle.load(open("1l2y_REST2_XS_20220527.pkl", "rb"))

# Define only desired XS_pool (externally determined)
XS_pool_F = XS_pool[c1c2_product[150]]
XS_pool_Itr = XS_pool[c1c2_product[147]]
XS_pool_Utr = XS_pool[c1c2_product[147]]

# delete XS_pool from memory
del XS_pool

# Set up GA pool for a state ()
# 200c, 1000n, 10r 
# genes, steps, reps 

# Change the GeneticAlgorithm object arguments
# GA_pool = [GeneticAlgorithm(XS_pool_F, F_st.S_exp, F_st.S_err, label=str(x),
#                       n_genes=200, n_cross=200, n_mutate=200, n_survive=200) for x in np.arange(10)]

For the unfolded state use q >= 0.035
GA_pool = [GeneticAlgorithm(XS_pool_Itr[:,3:], I_tr.S_exp[3:], I_tr.S_err[3:], label=str(x),
                      n_genes=200, n_cross=200, n_mutate=200, n_survive=200) for x in np.arange(10)]

# Execute pool GA on Folded
# set the number of jobs and steps here
pool = mp.Pool(10)
GA_pool = pool.map(fit_pool, zip(GA_pool, [20000]*len(GA_pool)))
pool.close()
pool.join()

# Save output, change the file name
save_ga_pool_h5(GA_pool, '1l2y_Itr_GA_pool_200c_20kn_10r.h5')

