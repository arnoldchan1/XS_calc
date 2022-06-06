import XS_calc
from XS_calc import *

import numpy as np
from scipy.optimize import minimize

import h5py
import pickle

import multiprocessing as mp
import time

# Define Genetic Algorithm and Chromosome classes

def fit_chromosome(chromosome):
    chromosome.fit()
    return chromosome

class Chromosome:
    
    def __init__(self, pool_options=None, n_genes=50, gene_indices=None):
        self.n_genes = n_genes
        self.gene_indices = gene_indices
        self.fitness = np.inf
        self.pool_option = None
        self.is_survivor = False
        self.expression = None
        self.res = None
        
    def fit(self, gene_data, target_data, target_err):
        

        gene_avg = np.mean(gene_data, axis=0)
        chi2_fun = lambda x: np.sqrt(np.mean( (((gene_avg * x[0] + x[1]) - target_data) / target_err)**2 ))
        res = minimize(chi2_fun, (target_data[0] / gene_avg[0] ,0), method='Nelder-Mead')
        self.fitness = res.fun
        self.res = res
        return res

        
    def assign(self, pool_option=None, fitness=None, is_survivor=None, res=None):
        if pool_option is not None:
            self.pool_option = pool_option
        if fitness is not None:
            self.fitness = fitness
        if is_survivor is not None:
            self.is_survivor = is_survivor
        if res is not None:
            self.res = res
        
    def individual_gene_data(self, gene_data):
#         if self.pool_option is not None:
#             gene_data = self.gene_pool[self.pool_option][self.gene_indices]
#         else:
#             gene_data = self.gene_pool[self.gene_indices]
        return (np.mean(gene_data * self.res.x[0] + self.res.x[1], axis=0), gene_data * self.res.x[0] + self.res.x[1])

    
class GeneticAlgorithm:
    
    def __init__(self, gene_pool, target_data, target_err=None, label=None,
                 n_chromosomes=150, n_survive=50, n_mutate=50, n_cross=50, n_genes=50):
        # Gene pool is the "data", an nD numpy array, or a dict of such
        # If a dict, then different keys of the dict are considered pool options
        # Every pool option must have the same number of gene pools
        self.gene_pool = gene_pool
        self.label = label
        self.pool_options = None
        self.pool_length = None
        self.n_chromosomes = n_chromosomes
        self.n_survive = n_survive
        self.n_mutate = n_mutate
        self.n_cross = n_cross
        self.n_genes = n_genes
        if self.n_chromosomes != self.n_survive + self.n_mutate + self.n_cross:
            self.n_chromosomes = self.n_survive + self.n_mutate + self.n_cross
        self.best_fitness = np.inf
        self.fitness_trace = []
        
        if type(gene_pool) == dict:
            self.pool_options = list(gene_pool.keys())
            # Check that within all options the gene pools are the same length
            pool_length_each_options = np.array([len(self.gene_pool[key]) for key in self.pool_options])
            if not np.all(pool_length_each_options == pool_length_each_options[0]):
                print("Warning: Not all pool options have the same number of genes")
            self.pool_length = pool_length_each_options[0]
            # Register options, makesure within all options the gene pools are the same length
        elif type(gene_pool) == list or type(gene_pool) == np.ndarray:
            self.pool_length = len(gene_pool)
        
        
        # Process data
        self.target_data = target_data
        if target_err is None:
            self.target_err = np.ones_like(self.target_data)
        else:
            self.target_err = target_err
        
        # Create chromosomes
        self.chromosome_pool = []
        for _ in range(self.n_chromosomes):
            self.chromosome_pool.append(Chromosome(n_genes=self.n_genes,
                                                   gene_indices=np.random.choice(range(self.pool_length), n_genes)))

        self.evolution_round = 0
        self.evolution_checkpoint = 0


    def report(self):
        print(f'This genetic algorithm: {len(self.chromosome_pool)} chromosomes, each with {self.n_genes} genes')
        print(f'For evolution, keep {self.n_survive}, mutate {self.n_mutate}, and cross {self.n_cross} chromosomes')
        
        
    def evolve(self, n=1, use_mp=False, silence=False, report_progress=True):
        # Calculate fit of every chromosome
        t0 = time.time()
        for round_counter in range(n):
            t1 = time.time()
            if t1 - t0 > 10:
                print(f'Progress for {self.label}: Round {round_counter + self.evolution_checkpoint}, best fit: {self.best_fitness}')
                t0 = t1
            if not silence:
                print(f'Round {round_counter + self.evolution_checkpoint}, processing chromosomes ...')
            fitness = np.zeros(self.n_chromosomes)
            if use_mp:
#                 raise NotImplementedError('Currently not working')
                self.fit_all_mp()
            else:
                self.fit_all()
            for idx, chromosome in enumerate(self.chromosome_pool):
                fitness[idx] = chromosome.fitness
            # Sort fitness
            fitness_rank = np.argsort(fitness)
            survivors = [self.chromosome_pool[x] for x in fitness_rank[:self.n_survive]]
            for survivor in survivors:
                survivor.assign(is_survivor=True)
            mutated = []
            crossed = []
            for chromosome in survivors:
                mutated.append(self.mutate(chromosome))
                crossed.append(self.cross(chromosome))
            self.chromosome_pool = survivors + mutated + crossed
            self.best_fitness = np.min(fitness)
            self.fitness_trace.append(np.min(fitness))
            if not silence:
                if self.pool_options is not None:
                    print(f'Best fit is: {np.min(fitness)} at {survivors[0].pool_option}')
                else:
                    print(f'Best fit is: {np.min(fitness)} for {self.label}')
            self.evolution_round += 1
        self.evolution_checkpoint = self.evolution_round
        print(f"Done with evolution for {self.label}, best fit: {self.best_fitness}")
        
    def fit_all(self):
        for idx, chromosome in enumerate(self.chromosome_pool):
#             if idx % 10 == 9:
#                 print(f'{idx+1}', end=' ')
            _ = self.fit_one(chromosome)
            
    def fit_all_mp(self):
        print('using multiprocessing', end=' ')
        pool = mp.Pool(52) # n_processes
        res = pool.map(self.fit_one, self.chromosome_pool)
        pool.close()
        pool.join()
        self.chromosome_pool = res
        
    def fit_one(self, chromosome):
        if self.pool_options is None:
            if not chromosome.is_survivor:
                chromosome.fit(self.gene_pool[chromosome.gene_indices], self.target_data, self.target_err)
        else:
            if not chromosome.is_survivor:
                sub_chi2 = np.inf
                sub_option = None
                sub_res = None
                for pool_option in self.pool_options:
                    chromosome.fit(self.gene_pool[pool_option][chromosome.gene_indices], self.target_data, self.target_err)
                    chi2_this = chromosome.res.fun
                    res_this = chromosome.res
                    if chi2_this < sub_chi2:
                        sub_chi2 = chi2_this
                        sub_option = pool_option
                        sub_res = res_this
                chromosome.assign(fitness=sub_chi2, pool_option=sub_option, res=sub_res)
        return chromosome
                    
                    
    
    def mutate(self, chromosome):
        n_old_genes = np.ceil(chromosome.n_genes * 0.8).astype(int)
        n_new_genes = chromosome.n_genes - n_old_genes
        old_genes = np.random.choice(chromosome.gene_indices, n_old_genes, replace=False)
        new_genes = np.random.choice(self.pool_length, n_new_genes)
        new_chromosome = Chromosome(n_genes=self.n_genes,
                                    gene_indices=np.concatenate((old_genes, new_genes)))
        
        return new_chromosome
    
    def cross(self, chromosome):
        other_chromosome = np.random.choice(self.chromosome_pool, 1)[0]
        combined_genes = np.concatenate((chromosome.gene_indices, other_chromosome.gene_indices))
        new_genes = np.random.choice(combined_genes, self.n_genes)
        new_chromosome = Chromosome(n_genes=self.n_genes,
                                    gene_indices=new_genes)
        
        return new_chromosome

    def get_best_fit(self):
        if self.pool_options is not None:
            best_data = self.gene_pool[self.chromosome_pool[0].pool_option][self.chromosome_pool[0].gene_indices]
            best_data = self.chromosome_pool[0].individual_gene_data(best_data)
        else:
            best_data = self.gene_pool[self.chromosome_pool[0].gene_indices]
            best_data = self.chromosome_pool[0].individual_gene_data(best_data)
        return best_data



# Import experimental scattering curves
h5 = h5py.File("20220404_trpcage_reconstructed_saxs.h5")
# define measurement q from experimental q
mea = Measurement(q=np.squeeze(h5['q_SAD'][:]))

# Create Experiment class objects for each species
F_st = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_SAD'][0,:]), S_err = np.squeeze(h5['s_SAD_err'][0,:]))
U_st = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_SAD'][1,:]), S_err = np.squeeze(h5['s_SAD_err'][1,:]))
I_tr = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_int'][:]), S_err = np.squeeze(h5['s_int_err'][:]))
U_tr = Experiment(q=np.squeeze(h5['q_SAD'][:]), S_exp = np.squeeze(h5['s_unf'][:]), S_err = np.squeeze(h5['s_unf_err'][:]))


# Load pickle file with XS_calc output: XS_pool
XS_pool = pickle.load(open("1l2y_REST2_XS_20220527.pkl", "rb")) # Change file path


# Function for multiprocessing
def fit_pool(obj):
	GA, n = obj
	GA.evolve(n, silence=True)
	return GA

# Get keys (c1/c2) as a list
key_sel = list(XS_pool.keys())


# Setup Folded GA pool
F_GA_pool = [GeneticAlgorithm(XS_pool[x], F_st.S_exp, F_st.S_err, label=f'({x[0]:.2f}, {x[1]:.1f})', 
	n_genes=200, n_cross=200, n_mutate=200, n_survive=200) for x in key_sel]
# Execute Folded GA pool
pool = mp.Pool(52) # Change n_processes
F_GA_pool = pool.map(fit_pool, zip(F_GA_pool, [200]*len(F_GA_pool)))
pool.close()
pool.join()

# Save output to a pickle
pickle.dump(F_GA_pool, open('1l2y_F_GA_pool_20220606.pkl', 'wb'))


# Setup Intermediate GA pool
Itr_GA_pool = [GeneticAlgorithm(XS_pool[x], I_tr.S_exp, I_tr.S_err, label=f'({x[0]:.2f}, {x[1]:.1f})', 
	n_genes=250, n_cross=250, n_mutate=250, n_survive=250) for x in key_sel]
# Execute Intermediate GA pool
pool = mp.Pool(52) # Change n_processes
Itr_GA_pool = pool.map(fit_pool, zip(Itr_GA_pool, [500]*len(Itr_GA_pool)))
pool.close()
pool.join()

# Save output to a pickle
pickle.dump(Itr_GA_pool, open('1l2y_Itr_GA_pool_20220606.pkl', 'wb'))


# Setup Unfolded_tr GA pool
Utr_GA_pool = [GeneticAlgorithm(XS_pool[x], U_tr.S_exp, U_tr.S_err, label=f'({x[0]:.2f}, {x[1]:.1f})', 
	n_genes=250, n_cross=250, n_mutate=250, n_survive=250) for x in key_sel]
# Execute Unfolded_tr GA pool
pool = mp.Pool(52) # Change n_processes
Utr_GA_pool = pool.map(fit_pool, zip(Utr_GA_pool, [500]*len(Utr_GA_pool)))
pool.close()
pool.join()

# Save output to a pickle
pickle.dump(Utr_GA_pool, open('1l2y_Utr_GA_pool_20220606.pkl', 'wb'))


# Setup Unfolded GA pool
U_GA_pool = [GeneticAlgorithm(XS_pool[x], U_st.S_exp, U_st.S_err, label=f'({x[0]:.2f}, {x[1]:.1f})', 
	n_genes=250, n_cross=250, n_mutate=250, n_survive=250) for x in key_sel]
# Execute Unfolded_tr GA pool
pool = mp.Pool(52) # Change n_processes
U_GA_pool = pool.map(fit_pool, zip(U_GA_pool, [500]*len(U_GA_pool)))
pool.close()
pool.join()

# Save output to a pickle
pickle.dump(U_GA_pool, open('1l2y_U_GA_pool_20220606.pkl', 'wb'))

