import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import time

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
        pool = mp.Pool(2)
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

# function to execute mp on iterable of (GA, n)
def fit_pool(obj):
  GA, n = obj
  GA.evolve(n, silence=True)
  return GA

# Use case for pool of (GA, n)
# pool = mp.Pool(2)
# GA_pool_F = pool.map(fit_pool, zip(GA_pool_F, [1000]*len(GA_pool_F)))
# pool.close()
# pool.join()

# function to save best chromosomes after optimization to hdf5
def save_ga_pool_h5(ga_pool, filename):
    best_fitness = []
    label = []
    bf_gene_indices = []
    bf_chrom = []
    bf_chrom_av = []

    for i in np.arange(len(ga_pool)):
        ga = ga_pool[i]

        best_fitness.append(ga.best_fitness)
        label.append(ga.label)

        bf_gene_indices.append(ga.chromosome_pool[0].gene_indices)

        bf = ga.get_best_fit()
        bf_chrom.append(bf[1])
        bf_chrom_av.append(bf[0])

    # ga_dict = dict([('label', label), ('best_fitness', best_fitness), ('bf_gene_indices', bf_gene_indices), ('bf_chrom', bf_chrom), ('bf_chrom_av', bf_chrom_av)])
    hf = h5py.File(filename, 'w')
    hf.create_dataset('label', data=label)
    hf.create_dataset('best_fitness', data=best_fitness)
    hf.create_dataset('bf_gene_indices', data=bf_gene_indices)
    hf.create_dataset('bf_chrom', data=bf_chrom)
    hf.create_dataset('bf_chrom_av', data=bf_chrom_av)
    hf.close()
    pass