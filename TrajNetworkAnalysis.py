"""
A collection of functions to work with LAMMPS trajectories using MDAnalysis and SciKit-Network.
This started as a way to analyze N130 trajectories to get at the networked structure of the condensates generated
by N130 and rpL5, as a minimal system to look at the structure generated by NPM1 and Arginine-rich peptides inside
the nucleolus.
"""
__author__ = 'Furqan Dar'
__version__ = 0.1

import MDAnalysis
import numpy
# import sknetwork
import numpy as np
# import scipy as sp
import MDAnalysis as mdan
from MDAnalysis.analysis import distances as mAnDist


def Read_N130_Trajectory(top_file: str, trj_file: str, num_of_reps: int = 108, verbose: bool = False):
    """
    Specifically made to read in LAMMPS trajectories generated for N130 with peptides. The main reason for this
    is to give each molecule a residue name corresponding to either N130 or rpL5.
    :param top_file: Name of the topology file. Should be a lammps `write_data` output file.
    :param trj_file: Name of the trajectory file. Should be a .dcd file.
    :param num_of_reps: The number of repeats of the fundamental repeating unit. For us we have 1 N130 to 15 rpL5s.
    :param verbose: Prints out extra information about the trajectory.
    :return: [MDAnalysis.Universe, Molecule Lengths, Molecule Numbers]
    """
    dum_universe = mdan.Universe(top_file, trj_file, format="LAMMPS")
    if verbose:
        print("The simulation box is", dum_universe.dimensions[:3])
    # Adding the names of the molecules
    names_of_res = ['N130']
    [names_of_res.append('rpL5') for i in range(15)]
    names_of_res = names_of_res * num_of_reps
    dum_universe.add_TopologyAttr(topologyattr='resnames', values=names_of_res)

    # Getting the lengths of the different molecules (or bits) we care about
    n130_len = len(dum_universe.residues[0].atoms)
    pept_len = len(dum_universe.residues[1].atoms)
    n130_arm_len = int((n130_len - 1) / 5)
    mol_lens = [n130_len, n130_arm_len, pept_len]
    if verbose:
        print("N130 has {:} beads; each arm has {:} beads.".format(n130_len, n130_arm_len))
        print("The peptide has {:} beads.".format(pept_len))
    pept_num = int(len(dum_universe.select_atoms('resname rpL5')) / pept_len)
    n130_num = 108
    mol_nums = [n130_num, pept_num]
    return [dum_universe, mol_lens, mol_nums]


def get_distance_map(mdan_universe: numpy.ndarray, atom_sel: MDAnalysis.AtomGroup):
    """
    Given a frame from an MDAnalysis Universe object, we calculate the distance map for the given atom_selection.
    This is a self-map.
    """

    dist_map = mAnDist.distance_array(atom_sel.positions, atom_sel.positions,
                                      box=mdan_universe.dimensions)

    return dist_map


def filter_distance_map(this_map: numpy.ndarray, r_lo: float = 50., r_hi: float = 75.):
    """
    Given this Distance Map, we create a sort of adjacency matrix where only the distances between r_lo and r_hi
    define an unweighted edge.
    Filters the distance map to have 1's if distances are inside the range, 0's elsewhere.
    """

    map_copy = np.ones_like(this_map, dtype=int)
    map_copy[this_map <= r_lo] = 0
    map_copy[this_map >= r_hi] = 0

    return np.array(map_copy, int)


def get_only_FDs(mdan_universe):
    """
    Given this MDAnalysis Universe object for the N130 CG simulations, we extract only the Folded Domains (FD's)
    of the N130 molecules.
    """

    return mdan_universe.select_atoms("type 1")


def loop_trj_adjacency_mat_FDs(mdan_un: MDAnalysis.Universe, start: int = 1000,
                               stop: int = 2000, step: int = 100,
                               r_lo: float = 50.0, r_hi: float = 90.):
    """
    Loop over the trajectory for a given MDAnalysis universe. Looking at only the Folded-Domains, which have
    'type 1', we generate a series of adjacency matrices for the FD's.
    We return the adjacency matrices as a numpy array.
    """

    dum_list = []

    for a_frame in mdan_un.trajectory[start:stop:step]:
        dum_sel = get_only_FDs(mdan_un)

        dum_map = get_distance_map(mdan_un, dum_sel)
        dum_map = filter_distance_map(dum_map, r_lo=r_lo, r_hi=r_hi)

        dum_list.append(dum_map)

    return np.array(dum_list, int)


def gen_type_str_from_type_list(dum_list: list = None):
    """
    Given a list containing integers corresponding to types, we generate a string that combines all of them separated
    by a space ' '.
    """
    if dum_list is None:
        dum_list = [6, 7, 10, 12, 17]

    dum_str = list(map(str, dum_list))
    dum_str = " ".join(dum_str)
    return f'type {dum_str:}'


def gen_adjacency_inter(cont_map: numpy.ndarray):
    """
    Given a contact map, C. We build an adjacency matrix, A, using a block construction. A = [[0, C], [C^T, 0]]
    """
    x, y = cont_map.shape

    dum_adj_mat = np.block([[np.zeros((x, x)), cont_map],
                            [cont_map.T, np.zeros((y, y))]])
    return dum_adj_mat


def gen_adjacency_all(cmap_AA: numpy.ndarray, cmap_BB: numpy.ndarray, cmap_AB: numpy.ndarray):
    """
    Given the 3 contanct maps representing intra-domain and inter-domain interactions, we construct an adjacency matrix.
    We have a block matrix.
    """

    return np.block([[cmap_AA, cmap_AB], [cmap_AB.T, cmap_BB]])


def gen_adjacency_intra(cmap_AA: numpy.ndarray, cmap_BB: numpy.ndarray):
    """
    Given the two intra-domain contact maps, we generate an adjacency matrix, again using block matrices.
    """

    xA, yA = cmap_AA.shape
    xB, yB = cmap_BB.shape

    return np.block([[cmap_AA, np.zeros((xA, yB))], [np.zeros((xB, yA)), cmap_BB]])


def get_adjacency_eigvals(adj_mat: numpy.ndarray):
    """
    Given an adjacency matrix, we calculate all the eigenvalues.
    Return them sorted largest to smallest
    """
    dum_eigs = np.sort(np.linalg.eigvalsh(adj_mat))[::-1]

    return dum_eigs


def get_type_list_definitions():
    """
    Convenience function to return a dictionary containing the types for the charged beads in each of the four domains
    """
    _A0_LIST = [6, 7, 10, 12, 17]
    _A1_LIST = [20, 22, 24, 25, 27]
    _A2_LIST = [33, 34, 35, 37, 39, 40, 41, 42, 43]
    _PE_LIST = [44, 45, 46, 47, 48, 50, 52, 56, 57, 58]
    _TypeListDict = {'A0': _A0_LIST, 'A1': _A1_LIST, 'A2': _A2_LIST, 'Pep': _PE_LIST}

    return _TypeListDict


def get_atom_sels_of_list(mdan_univ: MDAnalysis.Universe, list_of_domains: list, type_list_dict: dict,
                          verbose: bool = False):
    """
    Convience function to generate a dictionary that has the atom_selections of the domains.
    """
    _SelDict = {}
    for a_dom in list_of_domains:
        _SelDict[a_dom] = mdan_univ.select_atoms(gen_type_str_from_type_list(type_list_dict[a_dom]))
        if verbose:
            print(f"Domain {a_dom} has {len(_SelDict[a_dom])} beads.")
    return _SelDict


def gen_pairs_of_domains(list_of_domains: list, mode: str ='all'):
    """
    Convenience function to get all pairs from the list of domains
    """

    assert mode in ['all', 'inter'], "Mode can only be 'all', 'inter', or 'intra' only!"

    if mode == 'all':
        return [m1 + '-' + m2 for mID1, m1 in enumerate(list_of_domains) for m2 in list_of_domains[mID1:]]
    elif mode == 'inter':
        return [m1 + '-' + m2 for mID1, m1 in enumerate(list_of_domains) for m2 in list_of_domains[mID1 + 1:]]
    elif mode == 'intra':
        return [m1 + '-' + m1 for mID1, m1 in enumerate(list_of_domains)]
    else:
        return None


def gen_contact_pairs(tot_univ: MDAnalysis.Universe, at_sel_1: MDAnalysis.AtomGroup, at_sel_2: MDAnalysis.AtomGroup,
                      r_cut: float = 16.):
    """
    Given the two atom selections, we use the NeighborSearch algorithms from MDAnalysis to return all pairs of contacts.
    Remember that the pairs are in ID-space of the AtomGroups, and not indices with respect to the Universe.
    """
    dum_neigh_search = mdan.lib.nsgrid.FastNS(r_cut, at_sel_1.positions, tot_univ.dimensions)
    dum_result = dum_neigh_search.search(at_sel_2.positions)
    return dum_result.get_pairs()


def _loop_over_trj_gen_contact_pairs(mdan_univ: mdan.Universe,
                                     trj_start: int = 1000, trj_step: int = 100, trj_stop: int = 2000,
                                     list_of_domains: list = None, verbose : bool = False,
                                     rLo: float = 0.1, rHi: float = 18.5,
                                     pair_mode='inter'):
    """
    For the supplied MDAnalysis Universe, we loop over the trajectory and generate the pair-contacts for every pair
    using the NeighborSearch stuff from MDAnalysis.lib
    """
    num_of_frames = len(mdan_univ.trajectory[trj_start:trj_stop:trj_step])
    assert num_of_frames > 0, "We should at least have one frame calculation!"

    possible_domains = ['A0', 'A1', 'A2', 'Pep']
    if list_of_domains is None:
        list_of_domains = ['A0', 'A1']

    for a_domain in list_of_domains:
        assert a_domain in possible_domains, f"Supplied domain {a_domain} is not one of A0 A1 A2 or Pep"

    _TypeListDict = get_type_list_definitions()

    _SelDict = get_atom_sels_of_list(mdan_univ, list_of_domains, _TypeListDict, verbose)

    AllPairs = gen_pairs_of_domains(list_of_domains, mode=pair_mode)
    # AllPairs = [a_dom+'-'+'Pep' for a_dom in ['A0', 'A1', 'A2']]

    if verbose:
        print(f"Processing {num_of_frames} frames for {len(AllPairs)} pairs.")

    AllMaps = {}
    for pairID, a_pair in enumerate(AllPairs):
        dom1, dom2 = a_pair.split("-")

        sel1 = mdan_univ.select_atoms(gen_type_str_from_type_list(_TypeListDict[dom1]))
        sel2 = mdan_univ.select_atoms(gen_type_str_from_type_list(_TypeListDict[dom2]))

        AllMaps[a_pair] = []

        for frameID, a_frame in enumerate(mdan_univ.trajectory[trj_start:trj_stop:trj_step]):
            AllMaps[a_pair].append(gen_contact_pairs(mdan_univ, sel1, sel2, rHi))

    return AllMaps


def gen_cont_map_from_pairs(pairs_list: np.ndarray, sizeX: int, sizeY: int):
    """
    Given a list of pair contacts, we generate a contact matrix from them. The pairs are in B-A form where A is
    the reference and B is the second set.
    """
    assert sizeX >= 1 and sizeY >= 1, "The matrix must have positive dimensions!"
    assert pairs_list.shape[
               0] <= sizeX * sizeY, f"List-size ({pairs_list.shape[0]}) is larger than MxN ({sizeX * sizeY})."

    dum_mat = np.zeros((sizeX, sizeY), np.uint8)

    for i, j in pairs_list:
        dum_mat[j, i] = 1
    return dum_mat


def block_to_block_reduction_all(tot_map: numpy.ndarray, b1_size: int = 5, b2_size: int = 5):
    """
    Block-Block reduction of the given Matrix (or map). This assumes that given matrix is asymmetric, or that map
    is for two different domains. Therefore we don't have to worry about treating the diagonal in a special way.
    Furthermore, the summation is carried over the entire matrix.
    """
    xSize = tot_map.shape[0]
    ySize = tot_map.shape[1]

    assert xSize % b1_size == 0, "Provided block1 size is not a divisor of the number of rows!"
    assert ySize % b2_size == 0, "Provided block1 size is not a divisor of the number of rows!"

    xSize = xSize // b1_size
    ySize = ySize // b2_size

    dum_bl_to_bl = np.zeros((xSize, ySize))

    for m1 in range(0, xSize):
        for m2 in range(0, ySize):
            dum_b1_to_b2 = tot_map[m1 * b1_size: (1 + m1) * b1_size,
                                   m2 * b2_size: (1 + m2) * b2_size]
            dum_bl_to_bl[m1, m2] = dum_b1_to_b2.sum()

    return dum_bl_to_bl


def block_to_block_reduction_self(tot_map: numpy.ndarray, b_size: int = 5, mode: str = 'all'):
    """
    Block-Block reduction of the given Matrix (or map). This assumes that given matrix is symmetric, or rather it
    represents the contact map for the same domains. Therefore, the diagonal has to be treated differently. Furthermore,
    only the upper-triangular half is looped over for efficiency. We then take the transpose and add that back.
    Then we add the diagonal.
    """

    assert mode in ['all', 'inter'], "Supported modes are 'all' or 'inter'. In inter we do not add the diagonal."
    xSize = tot_map.shape[0]
    ySize = tot_map.shape[1]

    assert xSize == ySize, "Provided matrix is not symmetric!"
    assert xSize % b_size == 0, "The block-size is not a divisor of the Matrix dimensions!"

    xSize = xSize // b_size
    ySize = ySize // b_size

    dum_bl_to_bl = np.zeros((xSize, ySize))

    for m1 in range(0, xSize):
        for m2 in range(m1 + 1, ySize):
            dum_b1_to_b2 = tot_map[m1 * b_size: (1 + m1) * b_size,
                                   m2 * b_size: (1 + m2) * b_size]
            dum_bl_to_bl[m1, m2] = dum_b1_to_b2.sum()

    dum_diag = np.zeros(xSize)
    if mode == 'all':
        for m1 in range(0, xSize):
            for m2 in range(m1, m1 + 1):
                dum_b1_to_b2 = tot_map[m1 * b_size: (1 + m1) * b_size,
                                       m2 * b_size: (1 + m2) * b_size]
                dum_diag[m1] = dum_b1_to_b2.sum()

    dum_bl_to_bl = dum_bl_to_bl + dum_bl_to_bl.T + np.diag(dum_diag)

    return dum_bl_to_bl
