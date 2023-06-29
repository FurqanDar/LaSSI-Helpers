__author__ = 'Furqan Dar'
__version__ = '0.0.3'

import numpy as np
import scipy as sp
import sknetwork
import sys
from typing import List, Dict, Any

_INT_SIZE = 4
_BYTE_ORDER = sys.byteorder

class TrjReader(object):
    """
    General purpose trajectory reader for binary LaSSI trajectories. The trajectories are formatted as such:
    
    N,X_1,Y_1,Z_1,F_1,...,X_N,Y_N,Z_N,F_N,
    
    where in each frame the first int corresponds to the total number of atoms, and then every four ints corresponds to
    a bead where X, Y, Z are coordinates, and F is the beadPartner.
    
    """
    
    def __init__(self, file_path: str = "", num_atoms: int = None, num_frames: int = None):
        """
        Can be used to instantiate the class. If we know both the number of
        atoms per frame, and the total number of frames, we have enough information
        to extract any frame.
        """
        self._f_path = file_path
        if (num_atoms is None) or (num_frames is None):
            self._n_atoms, self._n_frms = self._get_atoms_and_frames(file_path=self._f_path)
        else:
            self._n_atoms = num_atoms
            self._n_frms  = num_frames
    
    @staticmethod
    def _get_atoms_and_frames(file_path:str = None):
        """
        Get the total number of atoms per frame, and number of frames for this trajectory.
        :param file_path:
        :return:
        """
        with open(file_path, 'rb') as tF:
            num_atoms = tF.read(1 * _INT_SIZE)
            num_atoms = int.from_bytes(num_atoms, byteorder=_BYTE_ORDER)
            frm_size = num_atoms * 4 * _INT_SIZE
            tF.seek(0)
            num_frames = len(tF.read()) // (frm_size + _INT_SIZE)
            
        return num_atoms, num_frames
    
    def __str__(self):
        """
        String representation of this particular trajectory.
        :return:
        'Num of atoms  :
         Num of frames :
         File Path     :
         '
        """
        
        line1 = f'Num of atoms  : {self._n_atoms:<10}'
        line2 = f'Num of frames : {self._n_frms:<10}'
        line3 = f'File Path     : {self._f_path}'
        
        all_lines = [line1, line2, line3]
        return "\n".join(all_lines)
    
    def __eq__(self, other):
        """
        If the trajectories have the same paths,they are they same.
        :param other:
        :return:
        """
        
        return self._f_path == other._f_path
    
    def __repr__(self):
        """
        A way to instantiate this current trajectory
        :return:
        """
        
        return f'tp.TrjReader(file_path={self._f_path}, num_atoms={self._n_atoms}, num_frames={self._n_frms})'

class TrjExtractor(TrjReader):
    """
    General purpose class that has methods for extracting frames from a particular binary trajectory.
    """
    
    def __init__(self, file_path: str = "", num_atoms: int = None, num_frames: int = None, frames_list: list = None):
        """
        :param file_path:
        :param num_atoms:
        :param num_frames:
        :param frames_list:
        :return:
        """
        super().__init__(file_path=file_path, num_atoms=num_atoms, num_frames=num_frames)
        
        if frames_list is None:
            self._frm_list = np.arange(self._n_frms)
        else:
            self._frm_list = frames_list
            
    def __str__(self):
        """
        
        :return:
        """
        line1 = f"Frame List    : {str(list(self._frm_list))}"
        
        return f'{super().__str__()}\n{line1}'
    
    def __repr__(self):
        """
        
        :return:
        """
        return f'{super().__repr__()[:-1]}, frames_list={str(list(self._frm_list))}'
    
    @staticmethod
    def get_frm_idx(frame_id: int = 0, num_atoms: int = None):
        """
        For a given frame_id, and the total number of atoms in the system, we return a pair of indecies that
        span the frame.
        :param frame_id:
        :param num_atoms:
        :return:
        """
        
        frmSize = num_atoms * 4 * _INT_SIZE + 1 * _INT_SIZE
        
        start_idx = frame_id * (frmSize)
        end_idx   = (frame_id+1) * (frmSize)
        return (start_idx, end_idx)
    
    def extract_frames(self):
        """
        Given the frames_list, we extract each of those frames.
        :return:
        """
        all_frames = np.zeros((len(self._frm_list), self._n_atoms, 4), dtype=f'i{_INT_SIZE}')
        
        with open(self._f_path, 'rb') as tF:
            fullBuff = tF.read()
            for frmID in self._frm_list:
                frmStart, frmEnd = self.get_frm_idx(frame_id=frmID, num_atoms=self._n_atoms)
                thisFrm = np.frombuffer(fullBuff[frmStart + _INT_SIZE: frmEnd], dtype=f'i{_INT_SIZE}')
                all_frames[frmID] = thisFrm.reshape((self._n_atoms, 4))
        return all_frames

    def extract_coords(self):
        """
        Given the frames_list, we extract only the coordinates from each frame.
        :return:
        """
        return self.extract_frames()[:, :, :-1]
    
    def extract_bonds(self):
        """
        Given the frames_list, we extract only the bonds from each frame.
        :return:
        """
        return self.extract_frames()[:, :, -1]
    
class TrjClusterAnalysis(object):
    """
    General purpose class intended to be used with extracted frames from LaSSI simulations.
    """

    @staticmethod
    def from_frame_get_bonded_pairs_of_beads(a_bP_list: np.ndarray) -> np.ndarray:
        """
        Goes over the bondPartner list and finds _all_ pairs of bonded beads. Since bonds are symmetric, this produces
        a redundant list.
        
        This corresponds to anisotropic bonds in LaSSI.
        
        Gets the (i,j) pairs where there is a bond.
        :param a_bP_list a list of beadPartner or beadFaces
        :return firstBead, secondBead. This has shape (2, N), where N is the number of bonds.
        """
    
        dum_beads = np.argwhere(a_bP_list != -1).T[0]
        dum_parts = a_bP_list[dum_beads]
        num_beads = len(dum_beads)
    
        ret_val = np.zeros((2, num_beads), dtype=int)
        ret_val[0] = dum_beads
        ret_val[1] = dum_parts
    
        return ret_val
    
    @staticmethod
    def from_bonds_of_all_frames_get_pairs_of_bonded_beads(all_bond_frames: np.ndarray) -> List[np.ndarray]:
        """
        Given the bonds from multiple frames, we iterate over the frames and generate the pairs-of-bonded beads for each
        frame.
        :param all_bond_frames:
        :return:  [btp.TrjClusterAnalysis.from_frame_get_bond_pairs_of_beads(aFrame) for aFrame in all_bond_frames]
        """
        return [TrjClusterAnalysis.from_frame_get_bonded_pairs_of_beads(aFrame) for aFrame in all_bond_frames]

    @staticmethod
    def from_bonded_pairs_of_beads_get_bonded_pairs_of_mols(bonded_beads: np.ndarray,
                                                            chainIDs_for_beadIDs: np.ndarray) -> np.ndarray:
        """
        Given the pairs-of-bonded beads, and the molIDs for the beadIDs, we generate the pairs-of-bonded-mols.
        Since we go from beads to molecules, the list of bonded molecules can have redundant pairs if multiple beads were
        bonded from two of molecules to each other.
        :param bonded_beads:
        :param chainIDs_for_beadIDs:
        :return:
        """
        bonded_mols = np.zeros_like(bonded_beads, int)
        bonded_mols[0, :] = chainIDs_for_beadIDs[bonded_beads[0, :]]
        bonded_mols[1, :] = chainIDs_for_beadIDs[bonded_beads[1, :]]
        
        return bonded_mols

    @staticmethod
    def from_bonded_pairs_of_beads_of_all_frames_get_bonded_pairs_of_mols_for_trajectory(bonded_beads_all_frames: list,
                                                                                         mol_ids_for_each_bead_id: np.ndarray) -> \
            List[np.ndarray]:
        """
        Given the pairs-of-bonded-beads for multiple frames, we iterate over the frames and generate the pairs-of-bonded-mols
        for each frame.
        :param bonded_beads_all_frames:
        :param mol_ids_for_each_bead_id:
        :return: [from_bonded_pairs_of_beads_get_bonded_pairs_of_mols(aFrame, mol_ids_for_each_bead_id) for aFrame in bonded_beads_all_frames]
        """
        return [TrjClusterAnalysis.from_bonded_pairs_of_beads_get_bonded_pairs_of_mols(aFrame, mol_ids_for_each_bead_id) for aFrame in
                bonded_beads_all_frames]



class TrjClusterAnalysis_SameMolSizes(object):
    """
    Clustering analysis class which assumes that all the molecules in the trajectories have the same sizes.
    """
    
    @staticmethod
    def from_frame_get_bonded_pairs_of_mols_assuming_same_mol_size(a_bP_list: np.ndarray, mol_size: int = 55):
        """
        Given the (i, j) bead pairs that are bonded, we divide by the molecule size to produce a list of (I,J) pairs
        of moleculeIDs. Similarly to the beads version, this produces a redundant non-unique list of pairs.
        
        
        Handy division to generate which molID each beadID belongs to.
        This is a quick and dirty version which assumes that the molecules are all the same size. The name
        is intentionally laboriously long as a reminder.
        REMINDER: Assumes that all molecules have the same size in the system.
        
        :param a_bP_list a list of beadPartner or beadFaces
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :return firstMol, secondMol. This has shape (2, N), where N is number of mol-bonds
        """
        return TrjClusterAnalysis.from_frame_get_bonded_pairs_of_beads(a_bP_list) // mol_size
    
    @staticmethod
    def from_frame_get_unique_molecule_bond_pairs_assuming_same_mol_size(a_bP_list: np.ndarray, mol_size: int = 55):
        """
        Goes over the bondPartner list, which is the anisotropic interaction bond from LaSSI, and
        gets all the unique pairs of bonded molecules.
        :param a_bP_list a list of beadPartner or beadFaces
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :return firstMol, secondMol. This has shape (2, N), where N is the number of unique mol-bonds.
        """
        return np.unique(TrjClusterAnalysis_SameMolSizes.from_frame_get_bonded_pairs_of_mols_assuming_same_mol_size(
                a_bP_list=a_bP_list, mol_size=mol_size), axis=1)
    
    @staticmethod
    def from_frame_get_intra_and_inter_molecule_bonds_for_each_molecule_assuming_same_mol_size(a_bP_list: np.ndarray,
                                                                                               mol_num: int = 1000,
                                                                                               mol_size: int = 55):
        """
        Get the number of intra-molecular bonds and inter-molecular bonds for all chains.
         > Given the beadFaces, we first get the pairs of bonds converted to chainIDs
         > We then get the unique chains from the list, including the number of occurences and the indecies
          > The number of occurences is the total number of bonds each chain has.
         > Loop over every chain, get list of bond partners and see how many of the bonds are self-bonds: intra-bonds.
         > The difference gives the inter-bonds
        :param a_bP_list a list of beadPartner or beadFaces
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :param mol_num The number of molecules in the system.
        :return intraBonds, interBonds. This has shape (2, N), where N is the number of molecules
        """
        
        _dum_bond_pairs = TrjClusterAnalysis_SameMolSizes.from_frame_get_bonded_pairs_of_mols_assuming_same_mol_size(
                a_bP_list, mol_size=mol_size)
        bnd_hist = np.zeros((mol_num, 2), dtype=int)
        
        unique_elms, elm_ids, total_bonds = np.unique(_dum_bond_pairs[0], return_counts=True, return_index=True)
        
        for elmID, (molID, bndID, tot_bnds) in enumerate(zip(unique_elms, elm_ids, total_bonds)):
            _bond_list = _dum_bond_pairs[1][bndID: bndID + tot_bnds]
            intra_bonds = np.count_nonzero(_bond_list == molID)
            
            bnd_hist[molID, 0] = intra_bonds
            bnd_hist[molID, 1] = tot_bnds - intra_bonds
        return bnd_hist
    
    @staticmethod
    def from_frame_gen_molecule_adjacency_matrix_assuming_same_mol_size(a_bP_list: np.ndarray,
                                                                        mol_size: int = 55,
                                                                        mol_num: int = 1000):
        """
        Goes over the bondPartner list, which is the anisotropic interaction bond from LaSSI, and
        gets all the unique pairs of bonded molecules, and returns the pairs as an adjacency matrix.
        :param a_bP_list a list of beadPartner or beadFaces
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :param mol_num  Number of molecules in the system.
        :return AdjMat of shape (mol_size, mol_size)
        """
        
        dum_mat = np.zeros((mol_num, mol_num), dtype=int)
        
        dum_mol_bonds = TrjClusterAnalysis_SameMolSizes.from_frame_get_unique_molecule_bond_pairs_assuming_same_mol_size(
                a_bP_list=a_bP_list, mol_size=mol_size)
        
        dum_mat[dum_mol_bonds[0], dum_mol_bonds[1]] = 1
        
        return dum_mat
    
    @staticmethod
    def from_frame_get_connected_molecule_components_assuming_same_mol_size(a_bP_list: np.ndarray,
                                                                            mol_size: int = 55,
                                                                            mol_num: int = 1000):
        """
        Returns all the connected components where nodes are molecules, and not beads.
        :param a_bP_list a list of beadPartner or beadFaces
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :param mol_num  Number of molecules in the system.
        :return labelList of shape (mol_num) which assigns a unique cluster identity to each chain/molecule
        """
        
        dum_adj_mat = TrjClusterAnalysis_SameMolSizes.from_frame_gen_molecule_adjacency_matrix_assuming_same_mol_size(
                a_bP_list=a_bP_list, mol_size=mol_size, mol_num=mol_num)
        dum_adj_mat = sp.sparse.csr_matrix(dum_adj_mat)
        
        return sknetwork.topology.connected_components(dum_adj_mat)
    
    @staticmethod
    def from_trajectory_get_connected_molecule_components_assuming_same_mol_size(full_trj: np.ndarray,
                                                                                 mol_size: int = 55,
                                                                                 mol_num: int = 1000):
        """
        Returns all the connected components where nodes are molecules, and not beads. For the entire trajectory.
        :param full_trj We only care about the last coordinate from each frame.
        :param mol_size Size of the molecules in the system. Assumes all have the same size.
        :param mol_num  Number of molecules in the system.
        :return labelList of shape (mol_num) which assigns a unique cluster identity to each chain/molecule
        """
        trj_shape = full_trj.shape
        num_frms = trj_shape[0]
        num_beads = trj_shape[1]
        num_crds = trj_shape[2]
        
        trj_conn_comps = np.zeros((num_frms, mol_num), dtype=int)
        
        analysis_func = TrjClusterAnalysis_SameMolSizes.from_frame_get_connected_molecule_components_assuming_same_mol_size
        
        for frmID, aFrame in enumerate(full_trj[:, :, -1]):
            trj_conn_comps[frmID] = analysis_func(a_bP_list=aFrame, mol_size=mol_size, mol_num=mol_num)
        return trj_conn_comps
