import MDAnalysis as mda
import numpy as np
from scipy.optimize import minimize
import time
import multiprocessing
import dill

# from pathos.multiprocessing import ProcessPool

# from numba import njit, prange

# @njit(parallel=True)
# def WPFunction_parallel(Wpos, Ppos):
#     WPLen = len(Wpos)
#     PPLen = len(Ppos)
#     distMat = np.zeros((WPLen, PPLen))
#     for i in prange(WPLen):
#         distMat[i] = ((Wpos - Ppos)**2).sum(1)
#     return distMat


def time_ms(t1, t0, message=None): # Timing utility
    if message is not None and len(message) > 0:
        print(f'{message}: {(t1-t0)*1000:.2f} ms')
    else:
        print(f'{(t1-t0)*1000:.2f} ms')
        
def raster_unit_sphere(num=200):
    L = np.sqrt(num * np.pi);
    pt = []
    for i in range(num):
        h = 1.0 - (2.0 * i + 1.0) / num
        p = np.arccos(h)
        t = L * p
        xu = np.sin(p) * np.cos(t)
        yu = np.sin(p) * np.sin(t)
        zu = np.cos(p)
        pt.append([xu, yu, zu])

    return np.array(pt)
    
class Molecule: 
# Invariant part of the trajectory, for example atoms' vdW radii

    def __init__(self, sel, use_CRYSOL=True, match_FoXS=False):
        FF_ref = load_form_factors()
        vdW_ref = load_vdW_radii(use_CRYSOL=use_CRYSOL, match_FoXS=match_FoXS)
        # self.elements = sel.atoms.elements # Is a numpy array of ['C', 'N' ...]
        self.elements = [a.type[0] for a in sel.atoms]
        self.vdW = np.array([vdW_ref[x]/100 for x in self.elements]) # Get this from vdW_ref, in Angstroms
        self.FF = np.array([FF_ref[x] for x in self.elements]) # Get this from FF_ref. Actual form factors 
        self.r_sol = 1.8
        self.cutoff = np.max(self.vdW) * 2 + np.max([self.r_sol, 1.8])
        self.n_atoms = len(self.elements)

class Frame: 
# Variable part of the trajectory, for example atoms' coordinates and therefore SASA

    def __init__(self, xyz, mol):
        self.xyz = xyz
        self.SASA = np.zeros(len(xyz))
        print(f'The protein has {len(xyz)} atoms')
        self.isSASAcalculated = False # Until we have the SASA calculated
        self.mol = mol
        self.SASA_A2 = 0  # SASA in angstrom squared
        
    def spatial_decomposition(self):
        self.min_xyz = np.min(self.xyz, axis=0) - self.mol.cutoff
        self.max_xyz = np.max(self.xyz, axis=0) + self.mol.cutoff * 2
        self.box, self.boy, self.boz = np.ceil((self.max_xyz - self.min_xyz) / self.mol.cutoff).astype(int)
        self.box_num = self.box*self.boy*self.boz
#         print(self.min_xyz, self.max_xyz, self.box, self.boy, self.boz, self.box_num)
        self.box_id = {}
        self.box_coor = {}
        # Create boxes
        for i in range(self.box):
            self.box_id[i] = {}
            self.box_coor[i] = {}
            for j in range(self.boy):
                self.box_id[i][j] = {}
                self.box_coor[i][j] = {}
                for k in range(self.boz):
                    self.box_id[i][j][k] = []
                    self.box_coor[i][j][k] = []
        
        
        # Assort each point to bins
        for idx, xyz in enumerate(self.xyz - self.min_xyz):
            x, y, z = np.ceil(xyz[0]/self.mol.cutoff), np.ceil(xyz[1]/self.mol.cutoff), np.ceil(xyz[2]/self.mol.cutoff)
            self.box_id[x][y][z].append(idx)
            self.box_coor[x][y][z].append(xyz)
            
        
        for i in range(self.box):
            for j in range(self.boy):
                for k in range(self.boz):
                    self.box_coor[i][j][k] = np.array(self.box_coor[i][j][k])
#                     print(f'boxes[{i}][{j}][{k}] = {self.box_id[i][j][k]}')
        
        
    def neighbor_calc(self):
        # This breaks down when protein is of certain size
        self.neighbor_list = {}
        for i in range(self.mol.n_atoms):
            self.neighbor_list[i] = []
        dist2_map = np.sum((self.xyz[:,:,None] - self.xyz[:,:,None].T)**2, axis=1) # squared distance map
#         print(len(np.where(dist2_map < self.mol.cutoff**2)[0]))
        for i, j in zip(*np.where(dist2_map < self.mol.cutoff**2)):
            if i != j:
                self.neighbor_list[i].append(j)
#         for i in range(self.mol.n_atoms):
#             print(i, self.neighbor_list[i])
#         print(dist2_map)
        pass

#     def neighbor_calc_numba(self):
#         # This breaks down when protein is of certain size
#         self.neighbor_list = {}
#         for i in range(self.mol.n_atoms):
#             self.neighbor_list[i] = []
#         dist2_map = WPFunction_parallel(self.xyz, self.xyz) # squared distance map

# #         print(len(np.where(dist2_map < self.mol.cutoff**2)[0]))
#         for i, j in zip(*np.where(dist2_map < self.mol.cutoff**2)):
#             if i != j:
#                 self.neighbor_list[i].append(j)
# #         for i in range(self.mol.n_atoms):
# #             print(i, self.neighbor_list[i])
# #         print(dist2_map)
#         pass

    def neighbor_calc_sd(self):
        self.neighbor_list = {}
        for i in range(self.mol.n_atoms):
            self.neighbor_list[i] = []

        for i in range(1, self.box-1):
            for j in range(1, self.boy-1):
                for k in range(1, self.boz-1):

                    if len(self.box_id[i][j][k]) > 0:
                        for l in range(i-1, i+2):
                            for m in range(j-1, j+2):
                                for n in range(k-1, k+2):
                                    if len(self.box_id[l][m][n]) > 0:
                                        # squared distance map
                                        dist2_map = np.sum((self.box_coor[i][j][k][:,:,None] - self.box_coor[l][m][n][:,:,None].T)**2, axis=1)
                                        for o, p in zip(*np.where(dist2_map < self.mol.cutoff**2)):
                                            q = self.box_id[i][j][k][o]
                                            r = self.box_id[l][m][n][p]
                                            if q != r:
                                                self.neighbor_list[q].append(r)
        for i in range(self.mol.n_atoms):
            self.neighbor_list[i].sort()
#             print(i, self.neighbor_list[i])
        
    def SASA_calc(self, env, force_recalc=False):
        if not self.isSASAcalculated or force_recalc:
            self.SASA_A2 = 0
            r = raster_unit_sphere(env.num_raster)
            if self.mol.n_atoms < 2500:
                self.neighbor_calc()
            else: 
                self.spatial_decomposition()
                self.neighbor_calc_sd()
            # Computes SASA for each atom and store in the mol.SASA (this will be a fraction from 0 to 1)
            for i in range(self.mol.n_atoms):
                if len(self.neighbor_list[i]) == 0:
                    SASA_this = 1.0
                else:
                    solvent_probe = self.xyz[i] + (self.mol.vdW[i]+env.r_sol) * r
    #                 print(i, np.sum(np.min(((solvent_probe[:,:,None] - self.xyz[self.neighbor_list[i]][:,:,None].T)**2).sum(1) - (self.mol.vdW[self.neighbor_list[i]]+env.r_sol)**2, axis=1)>0) / env.num_raster)
                    SASA_this = np.sum(np.min(((solvent_probe[:,:,None] - self.xyz[self.neighbor_list[i]][:,:,None].T)**2).sum(1) - (self.mol.vdW[self.neighbor_list[i]]+env.r_sol)**2, axis=1)>0) / env.num_raster
                self.SASA[i] = SASA_this
                self.SASA_A2 += SASA_this * (self.mol.vdW[i]+env.r_sol)**2 * 4 * np.pi
            self.isSASAcalculated = True # This avoids recalculation of SASA, which is time consuming
            
        else:
            return # Do nothing and return

    def SASA_calc_output_dots(self, env, fname='SASA.pdb', force_recalc=True):
        SASA_xyz = []
        if not self.isSASAcalculated or force_recalc:
            r = raster_unit_sphere(env.num_raster)
            if self.mol.n_atoms < 2500:
                self.neighbor_calc()
            else: 
                self.spatial_decomposition()
                self.neighbor_calc_sd()
            # Computes SASA for each atom and store in the mol.SASA (this will be a fraction from 0 to 1)
            for i in range(self.mol.n_atoms):
                if len(self.neighbor_list[i]) == 0:
                    good_solvent_probe = list(range(len(solvent_probe)))
                else:
                    solvent_probe = self.xyz[i] + (self.mol.vdW[i]+env.r_sol) * r
                    good_solvent_probe = np.where(np.min(((solvent_probe[:,:,None] - self.xyz[self.neighbor_list[i]][:,:,None].T)**2).sum(1) - (self.mol.vdW[self.neighbor_list[i]]+env.r_sol)**2, axis=1)>0)[0]
                for j in good_solvent_probe:
                    SASA_xyz.append(solvent_probe[j])
#                 print(i, len(good_solvent_probe) / env.num_raster, good_solvent_probe)
            with open(fname, 'w') as f:
                for idx, i in enumerate(SASA_xyz):
                    f.write(f'ATOM  {idx:5d}  XXX XXX P   1    {i[0]:8.3f}{i[1]:8.3f}{i[2]:8.3f}  0.00  0.00      P1\n')
                f.write('END\n')
        else:
            return # Do nothing and return
    
        
        
class Trajectory:
    def __init__(self, U, selection=None, use_CRYSOL=True, match_FoXS=False):
        # Take in the "Universe" object (just the molecule) and create these things 
        sel = U.select_atoms(selection)
        self.Molecule = Molecule(sel, use_CRYSOL=use_CRYSOL, match_FoXS=match_FoXS)
        self.Frames = []
        for ts in U.trajectory:
            self.Frames.append(Frame(sel.positions, self.Molecule))
            
    def SASA_calc_traj(self, env, force_recalc=False):
        if (env.r_sol > 1.8) and (env.r_sol > self.Molecule.r_sol):
            force_recalc = True
        self.Molecule.r_sol = env.r_sol
        for f in self.Frames:
            f.SASA_calc(env, force_recalc)


class Trajectory_slice:
    def __init__(self, U, selection=None, frame_min=0, frame_max=1, frame_step=0, use_CRYSOL=True, match_FoXS=False):
        # Take in the "Universe" object (just the molecule) and create these things 
        sel = U.select_atoms(selection)
        self.Molecule = Molecule(sel, use_CRYSOL=use_CRYSOL, match_FoXS=match_FoXS)
        self.Frames = []

        if frame_step == 0:
            # slicing from frame_min to frame_max
            if frame_max == 1:
                for ts in U.trajectory:
                    self.Frames.append(Frame(sel.positions, self.Molecule))
            else:
                for ts in U.trajectory[frame_min:frame_max]:
                    self.Frames.append(Frame(sel.positions, self.Molecule))
        else:
            # slicing with frame_step
            num_of_frames = U.trajectory.n_frames
            print('There are {:.1f} total frames'.format(num_of_frames))
            print('Taking the first in every {:.1f} frames'.format(frame_step))

            for ts in U.trajectory:
                if (ts.frame - frame_min) % frame_step == 0:
                    self.Frames.append(Frame(sel.positions, self.Molecule))
            
            # fiter = U.trajectory[frame_min:frame_max:frame_step]
            # for ts in fiter:
            #     self.Frames.append(Frame(sel.positions, self.Molecule))
            
            
    def SASA_calc_traj(self, env, force_recalc=False):
        if (env.r_sol > 1.8) and (env.r_sol > self.Molecule.r_sol):
            force_recalc = True
        self.Molecule.r_sol = env.r_sol
        for f in self.Frames:
            f.SASA_calc(env, force_recalc)


class Environment: # Sample environment related items
    def __init__(self, c1=1.0, 
                       c2=2.0, 
                       r_sol=1.8, 
                       r_m=1.62, 
                       rho=0.334,
                       num_raster=200):
        self.c1 = c1
        self.c2 = c2
        self.r_sol = r_sol
        self.r_m = r_m      # This is the term used in c1 term
        self.rho = rho      # This is the solvent electron density. 0.334 is that of water at 20 C.
        self.num_raster = num_raster

class Measurement: # Measurement related items, for now just the q vector
    def __init__(self, q):
        self.q = q

class Experiment: # Experimental data
    def __init__(self, q, S_exp, S_err):
        self.q = q
        self.S_exp = S_exp
        self.S_err = S_err
        self.XS = []
        self.c = []

    def fit_to_experiment(self, XS):
        XS_avg = np.mean(XS, axis=0)
        chi2_fun = lambda x: np.sqrt(np.mean( ((XS_avg - (self.S_exp * x[0] + x[1]))/self.S_err)**2 ))
        res = minimize(chi2_fun, (1,0), method='CG')
        self.c = res.x
        self.XS = (XS - res.x[1])/res.x[0]
        pass


def load_form_factors(flavor='WaasKirf'):
    # Return a dictionary containing the 11 coefficients from the WaasKirf table for each atom type
    # a1 a2 a3 a4 a5 c b1 b2 b3 b4 b5
    # 9 coefficients for CromerMann table
    # a1 a2 a3 a4 c b1 b2 b3 b4
    if flavor == 'WaasKirf':
        fname = r'form_factors/f0_WaasKirf.dat'
        # fname = r'/content/drive/My Drive/XS_calc/form_factors/f0_WaasKirf.dat' # for Google Colab)
    elif flavor == 'CromerMann':
        fname = r'form_factors/f0_CromerMann.dat'
        
    with open(fname) as f:
        content = f.readlines()
    
    f0s = {}
    for i, x in enumerate(content[:]):
        if x[0:2] == '#S':
            atom = x.split()[-1]
            coef = np.fromstring(content[i+3], sep='\t')
            f0s[atom] = coef
    
    return f0s

def load_vdW_radii(use_CRYSOL=True, match_FoXS=False):
    # vdW table in pm from mendeleev package (from W. M. Haynes, Handbook of Chemistry and Physics 95th Edition, CRC Press, New York, 2014, ISBN-10: 1482208679, ISBN-13: 978-1482208672.)
    # Crysol values are from CRYSOL paper and overrides the values in this table
    vdW_table = {'H':  110.0, 'He': 140.0, 'Li': 182.0, 'Be': 153.0,  'B': 192.0,
                 'C':  170.0, 'N':  155.0,  'O': 152.0,  'F': 147.0, 'Ne': 154.0,
                 'Na': 227.0, 'Mg': 173.0, 'Al': 184.0, 'Si': 210.0,  'P': 180.0,
                 'S':  180.0, 'Cl': 175.0, 'Ar': 188.0,  'K': 275.0, 'Ca': 231.0,
                 'Sc': 215.0, 'Ti': 211.0,  'V': 206.0, 'Cr': 206.0, 'Mn': 204.0,
                 'Fe': 204.0, 'Co': 200.0, 'Ni': 197.0, 'Cu': 196.0, 'Zn': 200.0,
                 'Ga': 187.0, 'Ge': 211.0, 'As': 185.0, 'Se': 190.0, 'Br': 185.0,
                 'Kr': 202.0, 'Rb': 303.0, 'Sr': 249.0,  'Y': 231.0, 'Zr': 223.0,
                 'Nb': 218.0, 'Mo': 217.0, 'Tc': 216.0, 'Ru': 213.0, 'Rh': 210.0,
                 'Pd': 210.0, 'Ag': 211.0, 'Cd': 218.0, 'In': 193.0, 'Sn': 217.0,
                 'Sb': 206.0, 'Te': 206.0,  'I': 198.0, 'Xe': 216.0, 'Cs': 343.0,
                 'Ba': 268.0, 'La': 243.0, 'Ce': 242.0, 'Pr': 240.0, 'Nd': 239.0,
                 'Pm': 238.0, 'Sm': 236.0, 'Eu': 235.0, 'Gd': 234.0, 'Tb': 233.0,
                 'Dy': 231.0, 'Ho': 229.0, 'Er': 229.0, 'Tm': 227.0, 'Yb': 225.0,
                 'Lu': 224.0, 'Hf': 223.0, 'Ta': 222.0,  'W': 218.0, 'Re': 216.0,
                 'Os': 216.0, 'Ir': 213.0, 'Pt': 213.0, 'Au': 214.0, 'Hg': 223.0,
                 'Tl': 196.0, 'Pb': 202.0, 'Bi': 206.0, 'Po': 197.0, 'At': 202.0,
                 'Rn': 220.0, 'Fr': 348.0, 'Ra': 283.0, 'Ac': 247.0, 'Th': 245.0,
                 'Pa': 243.0,  'U': 241.0, 'Np': 239.0, 'Pu': 243.0, 'Am': 244.0,
                 'Cm': 245.0, 'Bk': 244.0, 'Cf': 245.0, 'Es': 245.0, 'Fm': 245.0,
                 'Md': 246.0, 'No': 246.0, 'Lr': 246.0}
    vdW_table['H2O'] = 167.0
    if use_CRYSOL:
        vdW_table['H'] = 107.0
        vdW_table['C'] = 158.0
        vdW_table['N'] = 84.0
        vdW_table['O'] = 130.0
        vdW_table['S'] = 168.0
        vdW_table['Fe'] = 124.0
    if match_FoXS:
        vdW_table['H'] = 107.13 # 107.0
        vdW_table['C'] = 157.72 # 158.0
        vdW_table['N'] = 84.14  # 84.0
        vdW_table['O'] = 129.66 # 130.0
    # Return a dictionary containing the vdW radius for each atom type 
    return vdW_table

def FF_calc(frame, env, mea):
    # Calculate f(q, c1, c2) based on the q in measurement (mea), c1 and c2 in environment (env), and SASA in the frame

    # get s from q
    s = mea.q / (4 * np.pi)

    # anonymous function to calculate in vacuo form factors
    fv_func = lambda sval, a: np.sum(a[None, :5] * np.exp(-a[None, 6:] * sval[:, None] ** 2), axis=1) + a[5]

    # anonymous function to calculate C1, excluded volume adjustent coefficient
    #C1_func = lambda c1, q, rm: (c1 ** 3) * np.exp(-((4.0*math.pi/3.0)**(1.5)*(q**2)*(rm**2)*(c1**2-1.0)) / (4.0*math.pi))
    C1_func = lambda c1, q, rm: (c1 ** 3) * np.exp((-(4*np.pi/3)**(1.5)*(q**2)*(rm**2)*(c1**2-1))/(4*np.pi))

    # anonymous function to calculate excluded volue form factors
    # Fraser, R. D. B., T. P. MacRae and E. Suzuki. 1978. J. Appl. Cryst. 11:693-694
#     fs_func = lambda q, r0: math.pi**(1.5)*r0**3*env.rho*np.exp(-math.pi*(math.pi**(1.5)*r0**3)**(2/3)*q**2) 
#                               v_i               rho                      
    fs_func = lambda q, r0: (4.0/3.0*np.pi*(r0**3)) * env.rho * np.exp(-np.pi * ((4.0/3.0*np.pi*(r0**3))**(2.0/3.0)) * (q**2))  # Fraser's version
#     fs_func = lambda q, r0: (4.0/3.0*math.pi*(r0**3)) * env.rho * np.exp(-math.pi * ((4.0/3.0*math.pi*(r0**3))**(2.0/3.0)) * (q**2) / (4 * math.pi)) # FoXS's version
#     fs_func = lambda q, r0: (4.0/3.0*math.pi*(r0**3)) * env.rho * np.exp(-math.pi * (r0**2) * (q**2) )  # Darren's version

    # form factor of water with radius 1.67 A
    fw = fv_func(s,np.array([2.960427, 2.508818, 0.637853, 0.722838, 1.142756, 0.027014, 14.182259, 5.936858, 0.112726, 34.958481, 0.390240])) + 2 * fv_func(s,np.array([0.413048, 0.294953, 0.187491, 0.080701, 0.023736, 0.000049, 15.569946, 32.398468, 5.711404, 61.889874, 1.334118])) - C1_func(env.c1, s, env.r_m) * fs_func(s, 1.67) #1.6673)
    
    FF_q = [] # n_atoms by len(q) matrix
    for i in np.arange(len(frame.mol.elements)):
        FF_q.append(fv_func(s, frame.mol.FF[i,:]) - C1_func(env.c1,s,env.r_m)*fs_func(s,frame.mol.vdW[i]) + env.c2*frame.SASA[i]*fw)
        
    return np.array(FF_q)

def frame_XS_calc(frame, env, mea, ignoreSASA=False): # Calculate the X-ray scattering of a frame
    if not ignoreSASA:
        # Get the SASA calculated if not done
        frame.SASA_calc(env)

    # Calculate adjusted form factors as a table.
    FF_q = FF_calc(frame, env, mea)
    
    # an i by j matrix of distances between all atoms
    d_ij = np.sqrt(np.sum((frame.xyz[None,:,:]-frame.xyz[:,None,:])**2, axis=2))

    # Calculate scattering signal XS - currently this is quite slow. There has to be a way to make it faster
    XS = np.zeros(np.shape(mea.q))
    for i in np.arange(frame.mol.n_atoms):
        for j in np.arange(i+1, frame.mol.n_atoms):
            qd = mea.q * d_ij[i,j]
            XS += 2 * FF_q[i] * FF_q[j] * np.sinc(qd / np.pi)
        XS += FF_q[i] ** 2

    return XS

def frame_XS_calc_fast(frame, env, mea, ignoreSASA=False, timing=False): # Calculate the X-ray scattering of a frame
    t0 = time.time()
    if not ignoreSASA:
        # Get the SASA calculated if not done
        frame.SASA_calc(env)
    t1 = time.time()
    # Calculate adjusted form factors as a table.
    FF_q = FF_calc(frame, env, mea)
    t2 = time.time()
    # an i by j matrix of distances between all atoms
    d_ij = np.sqrt(np.sum((frame.xyz[None,:,:]-frame.xyz[:,None,:])**2, axis=2))
    d_ij += np.eye(len(d_ij)) * 1e-15
    t3 = time.time()

    # Calculate scattering signal XS - currently this is quite slow. There has to be a way to make it faster
    XS = np.zeros(np.shape(mea.q))
#     print((FF_q[:,None,:] * FF_q[:,:,None]).shape)

# This huge one-liner is slow
#     XS = np.sum((FF_q[:,None,:] * FF_q[:,:,None]) * np.sinc(mea.q[:, None, None] * d_ij[None,:,:] / np.pi), axis=(-1, -2))

# This q by q version is pretty good
    for idx, q in enumerate(mea.q):
        if q == 0.0:
            XS[idx] = np.sum((FF_q[:,idx][None,:] * FF_q[:, idx][:,None]))
#         XS[idx] = np.sum((FF_q[:,idx][None,:] * FF_q[:, idx][:,None]) * np.sinc(d_ij * q / np.pi))
        else:
            XS[idx] = np.sum((FF_q[:,idx][None,:] * FF_q[:, idx][:,None]) * np.sin(d_ij * q) / (d_ij * q))
    t4 = time.time()
    
# This element by element version is really slow
#     for i in np.arange(frame.mol.n_atoms):
#         for j in np.arange(i+1, frame.mol.n_atoms):
#             qd = mea.q * d_ij[i,j]
#             XS += 2 * FF_q[i] * FF_q[j] * np.sinc(qd / np.pi)
#         XS += FF_q[i] ** 2

    if timing:
        time_ms(t1, t0, 'SASA')
        time_ms(t2, t1, '  FF')
        time_ms(t3, t2, 'dist')
        time_ms(t4, t3, 'Xray')
    return XS

def frame_XS_calc_exp(frame, env, mea, ignoreSASA=False, timing=False):
    q_sphere = raster_unit_sphere(200)

    t0 = time.time()
    if not ignoreSASA:
        # Get the SASA calculated if not done
        frame.SASA_calc(env)
    t1 = time.time()
    # Calculate adjusted form factors as a table.
    FF_q = FF_calc(frame, env, mea)
    t2 = time.time()
    # Calculate scattering signal XS - currently this is quite slow. There has to be a way to make it faster
    XS = np.zeros(np.shape(mea.q))

    for idx, q in enumerate(mea.q):
        A = np.sum(FF_q[:,idx] * np.exp(-(1j) * q * np.dot(q_sphere, frame.xyz.T)), axis=1)
        XS[idx] = np.mean(np.abs(A)**2)
    t3 = time.time()
    if timing:
        time_ms(t1, t0, 'SASA')
        time_ms(t2, t1, '  FF')
        time_ms(t3, t2, 'Xray')
    return XS

# Calculate the X-ray scattering of an entire trajectory
def traj_calc(traj, env, mea, method="frame_XS_calc", n_processes=8, ignoreSASA=False): 
    tic = time.time()

    # Create an iterable list of tuples for multiprocessing
    frame_init_iter = []
    for i in np.arange(len(traj.Frames)):
        frame_init_iter.append((traj.Frames[i], env, mea))
    
    p = multiprocessing.Pool(processes=n_processes)
    if method == "frame_XS_calc_fast":
        XS = p.starmap(frame_XS_calc_fast, frame_init_iter)
    else:
        XS = p.starmap(frame_XS_calc, frame_init_iter)

    # pathos
    # pool = ProcessPool(4)
    # XS = pool.map(frame_XS_calc, frame_init_iter)
        

    # old loop for traj_calc without multiprocessing
    # XS = []
    # for frame in traj.Frames:
    #     # Calculate sturcture based on Debye formula and modified form factors
    #     XS.append(frame_XS_calc(frame, env, mea, ignoreSASA))

    XS = np.array(XS)

    toc = time.time()

    print('Done in {:.4f} seconds'.format(toc-tic))

    return XS

def fit_XS(XS, exp):
    XS_avg = np.mean(XS, axis=0)
    chi2_fun = lambda x: np.sqrt(np.mean( ((XS_avg - (exp.S_exp * x[0] + x[1]))/exp.S_err)**2 ))
    res = minimize(chi2_fun, (1,0), method='CG')
    return np.array([chi2_fun(res.x), res.x[0], res.x[1]])


def c_search(traj, mea, exp, c1_grid=np.arange(0.95,1.051,0.005), c2_grid=np.arange(-2.0,4.01,0.05), n_processes=8):
    chi2_grid = np.zeros((len(c1_grid), len(c2_grid), 3))
    for c1_i, c1_val in enumerate(c1_grid):
        for c2_i, c2_val in enumerate(c2_grid):
            c_grid = Environment(c1=c1_val, c2=c2_val)
            XS = traj_calc(traj, c_grid, mea, "frame_XS_calc_fast", n_processes)
            chi2_grid[c1_i, c2_i, :] = fit_XS(XS, exp)

    chi2_grid_mat = chi2_grid[:,:,0]
    c_best_idx = np.unravel_index(chi2_grid_mat.argmin(), chi2_grid_mat.shape)
    c1_best = c1_grid[c_best_idx[0]]
    c2_best = c2_grid[c_best_idx[1]]
    return c1_best, c2_best, chi2_grid 


def eom(XS_T, S_exp, S_err, N=20, K=50, M=5000, p_switch=0.2, bias=False):
    # Note: input is tranpose of XS
    # Make original K chromosomes of N genes
    chrom_or = np.zeros((N, K, M+1))
    # print(np.shape(chrom_0))
    for i in np.arange(K):
        chrom_or[:,i,0] = np.random.permutation(np.shape(XS)[1])[:N]


    chrom = np.zeros((N, (3*K), M))
    chrom_rm = np.zeros(np.shape(chrom_or))
    chrom_shuf = np.zeros(np.shape(chrom_or))
    chrom_cr = np.zeros(np.shape(chrom_or))

    sav_chrom = np.zeros((np.shape(XS_T)[0], (3*K), M))
    resnorm = np.zeros((np.shape(chrom)[1], M))
    resnorm_sort_get = np.zeros((K, M))
    if bias:
        sf = np.zeros((np.shape(chrom)[1], M, 2))
        sf_sort_get = np.zeros((K, M, 2))
    else:
        sf = np.zeros((np.shape(chrom)[1], M))
        sf_sort_get = np.zeros((K, M))


    for m in np.arange(M):
        
        # Random mutation
        chrom_rm[:,:,m] = chrom_or[:,:,m]
        for k in np.arange(K):
            i_pool = [];
            i_other = [];
            # number of genes to be swithced
            n_switch = int(np.round(np.random.uniform()*p_switch*N)) 
            i_switch = np.random.permutation(N)[:n_switch]
            if n_switch != 0:
                # indices to switch with pool
                i_pool[0:(int(np.floor(n_switch/2))+1)] = i_switch[0:(int(np.floor(n_switch/2))+1)] 
                # indices to switch with K-1 chromosome
                i_other[int(np.floor(n_switch/2)+1):(n_switch+1)] = i_switch[int(np.floor(n_switch/2)+1):(n_switch+1)]
            i_pool = np.array(i_pool)
            i_other = np.array(i_other)

            # Exchange with pool
            for pool_idx in i_pool:
                rand_pool = np.random.permutation(np.shape(XS_T)[1])[0]
                while rand_pool == chrom_rm[pool_idx, k, m]:
                    rand_pool = np.random.permutation(np.shape(XS_T)[1])[0]
                chrom_rm[pool_idx, k, m] = rand_pool

            # Exchange with K-1 chromosome
            for other_idx in i_other:
                chrom_rm[other_idx, k, m] = chrom_or[other_idx, k-1, m]

        # Crossing
        # Randomize the order of chromosomes
        chrom_shuf[:,:,m] = chrom_or[:,np.random.permutation(K),m]
        # Copy shuffled chromosomes to new matrix
        chrom_cr[:,:,m] = chrom_shuf[:,:,m]
        k = 1
        while k < K:
            n_cross = np.random.randint(1, N-2, 1)
            i_cross = np.random.permutation(N)[:n_cross[0]]
            chrom_cr[i_cross, k-1, m] = chrom_shuf[i_cross, k, m]
            chrom_cr[i_cross, k, m] = chrom_shuf[i_cross, k-1, m]
            k += 2

        # Compile the original and augmented chromosomes (3K)
        chrom[:,:,m] = np.concatenate((chrom_or[:,:,m], chrom_rm[:,:,m], chrom_cr[:,:,m]),1)

        # Calculate mean scattering patterns per chromosome
        for k in np.arange(np.shape(chrom)[1]):
            chrom_idx = chrom[:,k,m].astype(int)
            s_chrom = XS_T[:, chrom_idx]
            sav_chrom[:,k,m] = np.mean(s_chrom, axis=1)

        # "Elitism": fitting and sorting
        if bias:
            for k in np.arange(np.shape(chrom)[1]):
                chi2_fun = lambda x: np.mean( ((np.squeeze((sav_chrom[:,k,m])*x[0]+x[1]) - S_exp)/S_err)**2 )
                res = minimize(chi2_fun, (1,0), method='BFGS')
                sf[k, m, :] = res.x[:]
                resnorm[k, m] = chi2_fun(res.x)

            resnorm_sort = np.sort(resnorm[:,m])
            resnorm_sort_i = np.argsort(resnorm[:,m])
            sf_sort = sf[resnorm_sort_i, m, :]

            sf_get = []
            sf_chrom_i = []
            resnorm_get = []
            i = 1
            while len(sf_get) < K:
                if np.sum(sf_sort[i,:]) != 0:
                    sf_get.append(sf_sort[i-1])
                    resnorm_get.append(resnorm_sort[i-1])
                    sf_chrom_i.append(resnorm_sort_i[i-1])
                i += 1
            resnorm_sort_get[:,m] = resnorm_get
            sf_sort_get[:,m,:] = sf_get
            chrom_or[:,:,m+1] = chrom[:, sf_chrom_i, m]
                               
        else:
            for k in np.arange(np.shape(chrom)[1]):
                chi2_fun = lambda x: np.mean( ((np.squeeze(sav_chrom[:,k,m])*x[0] - S_exp)/S_err)**2 )
                res = minimize(chi2_fun, (1,), method='BFGS')
                sf[k, m] = res.x[0]
                resnorm[k, m] = chi2_fun(res.x)

            resnorm_sort = np.sort(resnorm[:,m])
            resnorm_sort_i = np.argsort(resnorm[:,m])
            sf_sort = sf[resnorm_sort_i, m]

            sf_get = []
            sf_chrom_i = []
            resnorm_get = []
            i = 1
            while len(sf_get) < K:
                if sf_sort[i] != 0:
                    sf_get.append(sf_sort[i-1])
                    resnorm_get.append(resnorm_sort[i-1])
                    sf_chrom_i.append(resnorm_sort_i[i-1])
                i += 1
            resnorm_sort_get[:,m] = resnorm_get
            sf_sort_get[:,m] = sf_get
            chrom_or[:,:,m+1] = chrom[:, sf_chrom_i, m]
        
    chrom_best = chrom[:,0,M-1].astype(int)
    return chrom_best, resnorm_sort_get, sf_sort_get, chrom


if __name__ == "__main__":
    U = mda.Universe('data/Ala10.pdb')
    