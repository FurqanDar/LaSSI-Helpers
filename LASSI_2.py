"""
New LaSSI Helper modules. Contains two main modules:
    - SystemSetup
        Used for instantiating, defining, setting up, submitting jobs, and data collection.
    - Plotting
        Used to plot the various data that are generated.
"""
__author__ = 'Furqan Dar'
__version__ = 0.3

import gzip

import matplotlib.axes
import numpy as np
import scipy as sp
import os
import subprocess as sproc
import pickle
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
import time

from typing import Dict
from typing import Any

class _TopoUtils_Linear_Molecule_Back(object):
    """
    Helper function to generate linear molecules of arbitrary architectures
    Implemented features:
        - Add an arbitrary block of bead-types.
        - Add a linker between the blocks defined.
        - Repeat a particular block.
        - Given all the blocks and inter-block linkers, generate the linear structure.
    """
    
    def __init__(self):
        self.Blocks = []
        self.Linkers = []
        self.InterBlockLinkers = []
        self.MolStructure = []
    
    def add_block(self, bead_list, linker_list):
        """
        Add a block of bead-types (in bead_list) connected by the linkers (in linker_list)
            Example add_block([0,1,0], [2,2]) ==> 0--1--0
        :param bead_list:
        :param linker_list:
        :return: None
        """
        assert len(bead_list) == (len(linker_list) + 1), \
            "number of bead and linkers does not generate a correct molecule!"
        self.Blocks.append(bead_list)
        self.Linkers.append(linker_list)
        return None
    
    def add_inter_block_linker(self, linker_len):
        """
        Add a linker between the block that has been defined, and the block that you _will_ be defining.
        :param linker_len:
        :return:
        """
        self.InterBlockLinkers.append(linker_len)
    
    def repeat_block(self, bead_list, linker_list, inter_linker, repeat_num=5):
        """
        Similar to add_block, we repeat the supplied block of bead-types (in bead_list) where the beads are connected
        via linkers (in linker_list). We repeat the block repeat_num times, and each block has inter_linker linker
        between them.
            Example: repeat_block([0,1,2], [2,2], 3, 2) ==> 0--1--2---0--1--2
        :param bead_list:
        :param linker_list:
        :param inter_linker:
        :param repeat_num:
        :return: None
        """
        assert len(bead_list) == (len(linker_list) + 1), \
            "number of bead and linkers does not generate a correct molecule!"
        for a_block in range(repeat_num):
            self.Blocks.append(bead_list)
            self.Linkers.append(linker_list)
            if a_block < repeat_num - 1:
                self.add_inter_block_linker(inter_linker)
        return None
    
    def form_structure(self):
        """
        Stitches all the blocks and inter-block linkers to form a linear molecule. Since we have a linear molecule,
        the first and last beads only have 1 bond. Every other bead is bonded to a bead before and after.
        :return:
        """
        block_number = len(self.Blocks)
        inter_number = len(self.InterBlockLinkers)
        assert inter_number == (block_number - 1), \
            "number of blocks and inter-block-linkers does not generate a correct structure!"
        
        tot_linker_list = []
        for block_ID, a_block in enumerate(self.Linkers):
            if block_ID > 0:
                tot_linker_list.append(self.InterBlockLinkers[block_ID - 1])
            for lin_ID, a_lin in enumerate(a_block):
                tot_linker_list.append(a_lin)
        
        tot_bead_list = []
        for block_ID, a_block in enumerate(self.Blocks):
            for bead_ID, a_bead in enumerate(a_block):
                tot_bead_list.append(a_bead)
        num_linkers = len(tot_linker_list)
        num_beads = len(tot_bead_list)
        assert num_beads == (num_linkers + 1), 'the linker and bead numbers are not consistent!'
        tot_bead_id = 0
        tot_struc = []
        for bd_ID, a_bead in enumerate(tot_bead_list):
            if bd_ID == 0:
                if num_beads == block_number == 1:
                    tot_struc.append([tot_bead_id, a_bead, -1, -1])
                else:
                    tot_struc.append([tot_bead_id, a_bead, tot_linker_list[bd_ID], tot_bead_id + 1])
                    tot_bead_id += 1
            elif bd_ID == num_beads - 1:
                tot_struc.append([tot_bead_id, a_bead, tot_linker_list[bd_ID - 1], tot_bead_id - 1])
                tot_bead_id += 1
            else:
                tot_struc.append([tot_bead_id, a_bead, tot_linker_list[bd_ID - 1], tot_bead_id - 1])
                tot_struc.append([tot_bead_id, a_bead, tot_linker_list[bd_ID], tot_bead_id + 1])
                tot_bead_id += 1
        self.MolStructure = np.array(tot_struc, dtype=int)
    
    def get_structure(self):
        """
        Returns the linear molecular structure that _could_ be put into LaSSI. Returns the structure as a NumPy array.
        :return:
        """
        self.form_structure()
        return self.MolStructure


class TopoUtils_Gen_Linear_Molecule(object):
    """
    General helper class to generate certain archetypal linear molecules.
    Implemented Linear Molecules:
        - gen_implicit_linear => a homopolymer of a bead-type with a specific length, and linkers in between.
        - gen_monomer         => a singular bead of bead_type
        - gen_dimer           => a dimer of the two bead-types and a linker length
    You can also set the structure manually, but it _has_ to be a _TopoUtils_Linear_Molecule_Back object.
    """
    
    def __init__(self, num_of_mols=1000):
        """
        Initialize the molecule
        """
        self.MolNumber = num_of_mols
        self.MolStruc = []
    
    @property
    def Structure(self):
        assert len(self.MolStruc) > 0, "The structure has not been formed yet."
        return [self.MolNumber, self.MolStruc]
    
    def set_mol_struc(self, new_struc):
        assert isinstance(new_struc,
                          _TopoUtils_Linear_Molecule_Back), "The structure should be _TopoUtils_Linear_Molecule_Back Type object"
        self.MolStruc = new_struc.get_structure()
    
    def gen_implicit_linear(self, bead_type=0, bead_num=7, lin_len=2):
        """
        Given the bead_type and bead_num, we generate a simple linear molecule with implicit linkers
            Example gen_implicit_linear(0, 5, 2) ==> 0--0--0--0--0
        :param bead_type:
        :param bead_num:
        :param lin_len:
        :return: None
        """
        DumStruc = _TopoUtils_Linear_Molecule_Back()
        DumStruc.add_block([bead_type] * bead_num, [lin_len] * (bead_num - 1))
        self.MolStruc = DumStruc.get_structure()
        return None
    
    def gen_monomer(self, bead_type=0):
        """
        Generate a monomer of the given bead-type.
        
        :param bead_type:
        :return: None
        """
        DumStruc = _TopoUtils_Linear_Molecule_Back()
        DumStruc.add_block([bead_type] * 1, [])
        self.MolStruc = DumStruc.get_structure()
    
    def gen_dimer(self, bead_1=0, bead_2=1, lin_len=2):
        """
        Generate a dimer with the given beads
            Example gen_dimer(0, 1, 2) ==> 0--1
        
        :param bead_1:
        :param bead_2:
        :param lin_len:
        :return: None
        """
        DumStruc = _TopoUtils_Linear_Molecule_Back()
        DumStruc.add_block([bead_1, bead_2], [lin_len] * 1)
        self.MolStruc = DumStruc.get_structure()
        return None


class TopoUtils_Gen_Linear_System(object):
    """
    Helpers to generate multiple molecules which can then be used as structures for the SystemSetup
    class below. Leverages TopoUtils_Gen_Linear_Molecules to add different linear molecules to a make a system of
    molecules.
    
    Remember that we also have to define the number of copies for each molecule. This class can be directly used
    to generate files for LaSSI.
    
    Implemented Linear Molecules:
        - gen_implicit_linear => a homopolymer of a bead-type with a specific length, and linkers in between.
        - gen_monomer         => a singular bead of bead_type
        - gen_dimer           => a dimer of the two bead-types and a linker length
    You can also set the structure manually, but it _has_ to be a _TopoUtils_Linear_Molecule_Back object.
    """
    
    def __init__(self):
        self.MolStrucs = []
        self.MolNums = 0
        self.StickNum = 0
    
    def add_monomer(self, bead_type=0, mol_num=1000):
        self.MolNums += 1
        dum_struc = TopoUtils_Gen_Linear_Molecule(num_of_mols=mol_num)
        dum_struc.gen_monomer(bead_type)
        self.MolStrucs.append(dum_struc.Structure)
    
    def add_dimer(self, bead_1=0, bead_2=1, lin_len=2, mol_num=1000):
        self.MolNums += 1
        dum_struc = TopoUtils_Gen_Linear_Molecule(num_of_mols=mol_num)
        dum_struc.gen_dimer(bead_1, bead_2, lin_len=lin_len)
        self.MolStrucs.append(dum_struc.Structure)
    
    def add_implicit_linear(self, bead_type: int=0, bead_num: int=7, linker_len: int=2, mol_num: int=1000):
        self.MolNums += 1
        dum_struc = TopoUtils_Gen_Linear_Molecule(num_of_mols=mol_num)
        dum_struc.gen_implicit_linear(bead_type, bead_num, linker_len)
        self.MolStrucs.append(dum_struc.Structure)
    
    @property
    def SysStrucs(self):
        return self.MolStrucs
    
    @property
    def StickerNumber(self):
        """
        Given the structures in the system, we explicitly calculate how many unique bead-types (or stickers) we have.
        :return:
        """
        dum_list = []
        for a_struc in self.SysStrucs:
            sticker_types = a_struc[1].T[1]
            dum_list.append(sticker_types)
        dum_list = np.array([a_type for a_set in dum_list for a_type in a_set])
        dum_counts, dum_vals = np.unique(dum_list, return_counts=True)
        self.StickNum = len(dum_counts)
        return self.StickNum
    
    def add_molecule(self, new_molecule):
        assert isinstance(new_molecule,
                          TopoUtils_Gen_Linear_Molecule), "new_molecule must be TopoUtils_Gen_Linear_Molecule type"
        self.MolNums += 1
        self.MolStrucs.append(new_molecule.Structure)
    
    def __repr__(self):
        """
        Crude way to show the system. Gives each unique bead-type a letter. Linker lengths are represented as '-'
        :return: {str} that contains system representation
        """
        bead_to_letter = {a_number: a_letter for a_letter, a_number in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', range(26))}
        
        list_to_print = []
        for strucID, a_struc in enumerate(self.SysStrucs):
            dum_num = str(a_struc[0])
            dum_str = TopoUtils_Gen_Linear_System._gen_mol_in_letters(a_struc[1], bead_to_letter)
            list_to_print.append(r'' + dum_num + '\t' + dum_str)
        
        return "\n".join(list_to_print)
    
    @staticmethod
    def _gen_mol_in_letters(this_struc: np.ndarray, bead_to_letter: dict):
        """
        Crude way to represent a molecule structure as a string. Takes the NumPy array based structure, and the
        supplied bead-type to letter dictionary
        :param this_struc:
        :param bead_to_letter:
        :return:
        """
        dum_list = []
        bead_types = this_struc.T[1]
        linker_len = this_struc.T[2]
        start_bead, end_bead = bead_types[0], bead_types[-1]
        dum_list.append(bead_to_letter[start_bead])
        if len(this_struc) == 1:
            return "".join(dum_list)
        elif len(this_struc) == 2:
            dum_list.append('-' * linker_len[0])
            dum_list.append(bead_to_letter[end_bead])
            return "".join(dum_list)
        else:
            for beadID, (a_bead, a_lin) in enumerate(zip(bead_types[1:-1:2], linker_len[1:-1:2])):
                dum_list.append('-' * a_lin)
                dum_list.append(bead_to_letter[a_bead])
            dum_list.append('-' * linker_len[-1])
            dum_list.append(bead_to_letter[end_bead])
            return "".join(dum_list)


class EnergyUtils_Gen_File(object):
    def __init__(self, tot_stick_num):
        self.St_Num = tot_stick_num
        self.Ovlp_En = np.zeros((tot_stick_num, tot_stick_num))
        self.Cont_En = np.zeros((tot_stick_num, tot_stick_num))
        self.Sti_En = np.zeros((tot_stick_num, tot_stick_num))
        self.FSol_En = np.zeros((tot_stick_num, tot_stick_num))
        self.TInd_En = np.zeros((tot_stick_num, tot_stick_num))
        self.Tot_File = []
        self.FileQ = False
    
    def Add_Ovlp_Btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Ovlp_En[comp1, comp2] = energy
        self.Ovlp_En[comp2, comp1] = energy
        return None
    
    def Add_Cont_Btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Cont_En[comp1, comp2] = energy
        self.Cont_En[comp2, comp1] = energy
        return None
    
    def Add_Sti_Btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Sti_En[comp1, comp2] = energy
        self.Sti_En[comp2, comp1] = energy
        return None
    
    def Add_FSol_For(self, energy=-0.5, this_type=0):
        self.FSol_En[this_type, this_type] = energy
        return None
    
    def Add_TInd_For(self, energy=-0.5, this_type=0):
        self.TInd_En[this_type, this_type] = energy
        return None

    def add_ovlp_btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Ovlp_En[comp1, comp2] = energy
        self.Ovlp_En[comp2, comp1] = energy
        return None

    def add_cont_btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Cont_En[comp1, comp2] = energy
        self.Cont_En[comp2, comp1] = energy
        return None

    def add_sti_btwn(self, energy=-0.5, a_pair=(0, 0)):
        comp1, comp2 = a_pair
        self.Sti_En[comp1, comp2] = energy
        self.Sti_En[comp2, comp1] = energy
        return None

    def add_fsol_for(self, energy=-0.5, this_type=0):
        self.FSol_En[this_type, this_type] = energy
        return None

    def add_tind_for(self, energy=-0.5, this_type=0):
        self.TInd_En[this_type, this_type] = energy
        return None
    
    @staticmethod
    def Write_Matrix(this_mat):
        unique_ints = np.unique(this_mat)
        num_of_ints = len(unique_ints)
        write_mat = []
        if num_of_ints == 1:
            write_mat.append(f"{this_mat[0, 0]:.3f}")
        else:
            for aRow in this_mat:
                this_line = []
                for aNum in aRow:
                    this_line.append(f"{aNum:.3f}")
                this_line = " ".join(this_line)
                write_mat.append(this_line)
        return write_mat
    
    def Form_Energy_File(self):
        tot_file = []
        tot_file.append("#STICKERS")
        tot_file.append(str(self.St_Num))
        tot_file.append("")
        
        tot_file.append("#OVERLAP_POT")
        for aline in self.Write_Matrix(self.Ovlp_En):
            tot_file.append(aline)
        tot_file.append("")
        
        tot_file.append("#CONTACT_POT")
        for aline in self.Write_Matrix(self.Cont_En):
            tot_file.append(aline)
        tot_file.append("")
        
        tot_file.append("#CONTACT_RAD")
        tot_file.append(str(0.0))
        tot_file.append("")
        
        tot_file.append("#SC_SC_POT")
        for aline in self.Write_Matrix(self.Sti_En):
            tot_file.append(aline)
        tot_file.append("")
        
        tot_file.append("#FSOL_POT")
        for aline in self.Write_Matrix(self.FSol_En):
            tot_file.append(aline)
        tot_file.append("")
        
        tot_file.append("#T_IND_POT")
        for aline in self.Write_Matrix(self.TInd_En):
            tot_file.append(aline)
        tot_file.append("")
        
        tot_file.append("#LINKER_LENGTH")
        tot_file.append(str(1.0))
        tot_file.append("")
        
        tot_file.append("#LINKER_SPRCON")
        tot_file.append(str(0.0))
        tot_file.append("")
        
        tot_file.append("#LINKER_EQLEN")
        tot_file.append(str(1.0))
        tot_file.append("")
        self.Tot_File = tot_file
        self.FileQ = True
    
    def get_EnFile(self):
        """
        Getter for the energy method
        
        :return: Tot_File
        """
        assert self.FileQ, "File needs to have been formed using Form_Energy_File() method"
        return self.Tot_File
    
    def Print_File(self):
        """
        Prints the energy-file to the screen.
        
        :return:
        """
        assert self.FileQ, "File needs to have been formed using Form_Energy_File() method"
        for a_line in self.Tot_File:
            print(a_line)


class _NestedRunConditions(object):
    """
    Helper class that generates a deeply nested dictionary that contains all the possible run conditions given
    the list of box sizes, molecule numbers, and replicate numbers.
    """
    
    def __init__(self, prefix: str, box_list: list, mol_list: list, rep_num: int):
        self.DirPrefix = prefix
        self.Boxes = np.array(box_list)
        self.MolNums = np.array(mol_list, dtype=int)
        self.Reps = np.arange(1, rep_num + 1)
        self.NestedRunConditions = _NestedRunConditions.gen_nested_run_conditions_with_boxes_mols_reps(
            _boxes=self.Boxes,
            _mols=self.MolNums,
            _reps=self.Reps)
        self.NumberOfConditions = len(self.Boxes) * len(self.MolNums) * len(self.Reps)
        self.QSuccess = False
        return None
    
    def __str__(self) -> str:
        """
        Returns a string format for the following information:
        - Directory prefix:
        - Boxes
        - Molecule Numbers
        - Replicate Numbers
        :return:
        """
        l1 = f"Directory Prefix: {self.DirPrefix}/"
        l2 = f"Boxes:\n{self.Boxes}"
        l3 = f"Molecule numbers:\n{self.MolNums}"
        l4 = f"Replicates:\n{self.Reps}"
        l5 = f"Total number of conditions: {self.NumberOfConditions}"
        
        return "\n".join([l5, l1, l2, l3, l4])
    
    @staticmethod
    def _mol_num_to_underscrore_str(mol_num: np.ndarray) -> str:
        """
        Given a list of molecule numbers: [n1, n2, ..., nN], we produce a string where each molecule number is joined using
        an underscore '_'.
        :param mol_num: Array of molecule numbers [n1, n2, ..., nN]
        :return: str(n1)_str(n2)..._str(nN)
        """
        return "_".join([str(a_num) for a_num in mol_num])
    
    @staticmethod
    def gen_nested_run_conditions_with_boxes_mols_reps(_boxes: np.ndarray, _mols: np.ndarray, _reps: np.ndarray) -> \
    Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Given the list of boxes, the list of molecules, and the list of replicates, we generate a deeply nested
        dictionary. For every box-size, we loop over all the molecule numbers, and for every molecule, we loop over
        all the replicates. Takes all the lists and make string based keys for each run condition.
        :param _boxes:
        :param _mols:
        :param _reps:
        :return: Deeply nested dictionary that contains all the run conditions. The first level is box-sizes, the second
        is molecule numbers and the last is the replicates. Each value is set as None in this dictionary.
        """
        _totDict = {}
        for aBox in _boxes:
            boxKey = str(aBox)
            _totDict[boxKey] = {}
            for aMol in _mols:
                numKey = _NestedRunConditions._mol_num_to_underscrore_str(aMol)
                _totDict[boxKey][numKey] = {}
                for aRep in _reps:
                    repKey = str(aRep)
                    _totDict[boxKey][numKey][repKey] = None
        
        return _totDict
    
    def check_for_run_success(self):
        """
        Loops over the nested dictionary and reads the log files to see if they contain the keyword ENDING, which
        signifies that the run was successful.
        :return:
        """
        self.NestedRunConditions = IOUtils.loop_function_over_deeply_nested_run_conditions_dict(self,
                                                                                                passed_func=JobSubmission.read_log_for_ENDING,
                                                                                                file_name="log.txt")
        self.QSuccess = True
        return None
    
    def check_if_at_least_n_reps_successful(self, min_reps: int = 2):
        """
        Loops over the nested dictionary and checks if at least min_reps replicates were successful. Returns False after
        the first condition that fails.
        :param min_reps:
        :return: True if we have enough successful replicates, False if not.
        """
        assert(self.QSuccess), "Must have already checked if runs were successful first! `check_for_run_success`"
        
        for aBox, boxMols in self.NestedRunConditions.items():
            for aMol, molReps in boxMols.items():
                repIt = 0
                for aRep, repVal in molReps.items():
                    if repVal:
                        repIt += 1
                if repIt < min_reps:
                    return False
        return True
    
    @staticmethod
    def _convert_nested_dict_to_nested_list(nested_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> list:
        """
        Returns the NestedRunConditions dictionary as a nested list by looping.
        The list should have the following structure:
        [per_box][per_mol][per_rep]
        :return:
        """
        per_box = []
        for aBox, boxMols in nested_dict.items():
            per_mol = []
            for aMol, molReps in boxMols.items():
                per_rep = []
                for aRep, repVal in molReps.items():
                    per_rep.append(repVal)
                per_mol.append(per_rep)
            per_box.append(per_mol)
        return per_box[:]
    
    
class IOUtils(object):
    """
    A collection of IO Utilities.
    """
    
    @staticmethod
    def mdkir_catch(this_dir):
        try:
            os.mkdir(this_dir)
        except OSError:
            pass
    
    
    @staticmethod
    def gen_dir_str(boxSize: int = 100, molSet: list = None, repNum: int = 1, wInt: bool = True):
        """
        Given a boxSize, a set of molecule-numbers, the replicate number, and the interaction state,
        we generate the str that corresponds to this particular simulation instance.
        """
        
        if molSet is None:
            molSet = [1000, ]
        
        dum_dir = []
        # dum_dir.append('')
        dum_dir.append(str(boxSize))
        molStr = "_".join([str(a_mol) for a_mol in molSet])  # Create underscored string from mol-nums
        dum_dir.append(molStr)
        if wInt:
            dum_dir.append("WInt")
        else:
            dum_dir.append("NoInt")
        
        dum_dir.append(str(repNum))
        if os.name == 'nt':
            return "\\".join(dum_dir) + "\\"
        else:
            return "/".join(dum_dir) + "/"
    
    @staticmethod
    def gen_dir_for_box(boxSize=100,
                        molList=None,
                        repNums=1,
                        wInt=True):
        """
        For a given box-size, and a list of mol_nums, we generate the strings corresponding to all the mol_nums
        
        :param boxSize
        :param molList:
        :param repNums:
        :param wInt:
        :return:
        """
        
        if molList is None:
            molList = [[100, 100, ], [200, 200]]
        
        dum_dir_list = []
        for setID, a_set in enumerate(molList):
            for a_rep in range(1, repNums + 1):
                dum_dir_list.append(IOUtils.gen_dir_str(boxSize, a_set, a_rep, wInt))
        return dum_dir_list
    
    @staticmethod
    def gen_dir_for_box_nested(boxSize=100,
                               molList=None,
                               repNums=1,
                               wInt=True,
                               molCycles=1):
        """
        For a given box-size, and a list of mol_nums, we generate the strings corresponding to all the mol_nums
        
        :param boxSize
        :param molList:
        :param repNums:
        :param wInt:
        :param molCycles
        :return:
        """
        
        if molList is None:
            molList = [[100, 100, ], [200, 200]]
        
        per_dir_list = []
        for cycID, a_cycle in enumerate(range(molCycles)):
            per_mol_list = []
            for setID, a_set in enumerate(molList):
                this_set = np.roll(a_set, a_cycle)
                per_rep_list = []
                for a_rep in range(1, repNums + 1):
                    per_rep_list.append(IOUtils.gen_dir_str(boxSize, this_set, a_rep, wInt))
                per_mol_list.append(per_rep_list)
            per_dir_list.append(per_mol_list)
        return per_dir_list
    
    @staticmethod
    def gen_dir_list(boxList=None,
                     molList=None,
                     repNums=3,
                     wInt=True):
        """
        Given the list of box-sizes (boxList), and the list of molecule numbers (molList), we generate a list of all the
        possible directories for the given interaction state (wInt) and replicate numbers (repNums))
        We check to make sure that the lengths of boxList and molList are the same
        """
        
        if boxList is None:
            boxList = [100, 110, ]
        
        if molList is None:
            molList = [[1000, 1000], [900, 900], ]
        
        assert len(boxList) == len(molList), "Number of boxes and number of molecule-tuples should be the same!"
        
        dum_dir_list = []
        for a_box, molSet in zip(boxList, molList):
            dum_dir_list.append(IOUtils.gen_dir_for_box(a_box, molSet, repNums, wInt))
        
        return [a_dir for a_set in dum_dir_list for a_dir in a_set]
    
    @staticmethod
    def gen_dir_nested_list(boxList=None,
                            molList=None,
                            repList=None,
                            wInt=True,
                            dir_prefix='/'):
        """
        Generate a nested list given a list of box-sizes, a list_of_mol_nums, replicate numbers, interaction state.
        molCycle is used to use np.roll(a_mol) to generate the permutations for each molecule number set.
        The idea being that it can be used to generate the different directions in the case of orthogonal sampling
        The outermost index is the direction, then box, then molecule number, then replicate.
        Again, remember that the boxList and molList are assumed to be from SystemSetup.get_independent_conditions()
        
        :param boxList List of box-sizes
        :param molList: List of molecule numbers
        :param repList: Number of replicates
        :param wInt: Interaction state
        :return: nested_list_of_dirs = [a_dir, a_box, a_mol, a_rep]
        """
        
        if boxList is None:
            boxList = [100, 110, ]
        
        if molList is None:
            molList = [[1000, 1000], [900, 900], ]
        
        if repList is None:
            repList = [1, 2, 3]
        
        assert len(boxList) == len(molList), "Number of boxes and number of molecule-tuples should be the same!"
        per_box_list = []
        for totID, (a_box, a_mol_set) in enumerate(zip(boxList, molList)):
            per_mol_list = []
            for molID, a_mol in enumerate(a_mol_set):
                per_rep_list = []
                for repID, a_rep in enumerate(repList):
                    per_rep_list.append(dir_prefix + IOUtils.gen_dir_str(boxSize=a_box, molSet=a_mol, repNum=a_rep,
                                                                         wInt=wInt))
                per_mol_list.append(per_rep_list)
            per_box_list.append(per_mol_list)
        
        return per_box_list
    
    @staticmethod
    def gen_run_conditions(boxList=None,
                           molList=None,
                           repNums=3):
        """
        Creates flattened arrays of boxes and molList which are 1-1 with the gen_dir_list list_of_directories.
        This is to be used in writing key-files and such
        
        :param boxList:
        :param molList:
        :param repNums:
        :return:
        """
        
        if boxList is None:
            boxList = [100, 110, ]
        
        if molList is None:
            molList = [[1000, 1000], [900, 900], ]
        
        dum_boxes = []
        dum_mols = []
        
        for boxID, (a_box, a_mol) in enumerate(zip(boxList, molList)):
            for setID, a_set in enumerate(a_mol):
                for repID, a_rep in enumerate(range(repNums)):
                    dum_boxes.append(a_box)
                    dum_mols.append(a_set)
        
        return np.array(dum_boxes), np.array(dum_mols)
    
    @staticmethod
    def read_param_file(file_name: str, verbose: bool = True):
        """
        Read a LASSI key-file.
        For each line in the key-file, we store the key-word and the value as a key-value pair in a large dictionary.
        Ignore all lines that are empty, or start with '#'
        
        :return: [list of keys, the full_dictionary]
        """
        dum_dict = {}
        dum_list = []
        with open(file_name) as pFile:
            all_lines = pFile.readlines()
            for lineID, a_line in enumerate(all_lines):
                if a_line[0] == '#' or a_line[0] == '\n':
                    continue
                this_line = a_line.strip('\n').rstrip().lstrip().split(' ')
                this_key, this_val  = this_line[0], this_line[-1]
                dum_list.append(this_key)
                dum_dict[this_key] = this_val
        
        if verbose:
            print('-'*40)
            print('Keyfile')
            for keyID, a_key in enumerate(['N_STEPS', 'PREEQ_STEPS']):
                print(f'{a_key:<25} {int(dum_dict[a_key]):<8.1e} {"|":>5}')
            for keyID, a_key in enumerate(['MC_CYCLE_NUM', 'MC_TEMP', 'MC_DELTA_TEMP']):
                print(f'{a_key:<25} {dum_dict[a_key]:<8} {"|":>5}')
            print('-'*40)
        
        return [dum_list, dum_dict]
    
    @staticmethod
    def write_param_file(param_obj,
                         file_path: str,
                         box_size: int = 100,
                         run_name: str = 'DumR',
                         energy_file: str = 'DumE',
                         struc_file: str = 'DumS',
                         rng_seed: int = 0,
                         clus_mode: int = 0):
        """
        Given a template key-file, we write the key-file for this particular simulation.

        :param file_path: Absolute path to where the key-file should be written.
        :param box_size: Size of simulation box
        :param run_name: Name of the run
        :param energy_file: Absolute path to the energy-file
        :param struc_file: Absolute path to the structure-file
        :param rng_seed: Seed for the RNG in C
        :param clus_mode: 0 for aniso-based, 2 (or 1) for ovlp based

        :return: None
        """
        dum_keys = param_obj[0][:]
        dum_vals = copy.deepcopy(param_obj[1])
        
        dum_vals['BOX_SIZE'] = str(box_size)
        dum_vals['REPORT_PREFIX'] = run_name
        dum_vals['STRUCT_FILE'] = struc_file
        dum_vals['ENERGY_FILE'] = energy_file
        dum_vals['RANDOM_SEED'] = str(rng_seed)
        dum_vals['ANALYSIS_CLUSTER_MODE'] = clus_mode
        with open(file_path, "w+") as pFile:
            for a_key in dum_keys:
                line_w = f'{a_key:<25} {dum_vals[a_key]:}\n'
                pFile.write(line_w)
        return None
    
    @staticmethod
    def copy_file_with_postfix(file_path: str = 'param.key', postfix: str = 'old'):
        """
        A wrapper function to copy a file and add a postfix to the file-name. The following terminal command is run
         > cp {file_path} {file_path}.{postfix}
        :param file_path:
        :param postfix:
        :return:
        """
        dum_comm = f'cp {file_path} {file_path}.{postfix}'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def write_keyfile_with_new_seed_from_older_keyfile(old_file_path: str, new_file_path: str, new_seed: int = 0):
        """
        Reads in an existing param-file and writes a new param-file with a new RNG_SEED value.
        :param old_file_path:
        :param new_file_path:
        :param new_seed:
        :return:
        """
        dum_key = IOUtils.read_param_file(file_name=old_file_path, verbose=False)
        dum_key[1]['RANDOM_SEED'] = new_seed
    
        with open(new_file_path, "w") as keyFile:
            for key, val in dum_key[1].items():
                keyFile.write(f"{key:<25} {val}\n")
    
    @staticmethod
    def write_struc_file(struc_obj,
                         file_path):
        """
        Write the structure object in the format readable by LASSI to file_path
        
        :param file_path:
        :return:
        """
        with open(file_path, "w+") as strucFile:
            for a_Mol in struc_obj:
                mol_num = a_Mol[0]
                mol_struc = a_Mol[1]
                beads_per_mol = len(np.unique(mol_struc.T[0]))
                strucFile.write("#New Molecule Type:= {:} beads per molecule\n".format(beads_per_mol))
                strucFile.write("NEW{\n")
                strucFile.write(str(mol_num) + "\n")
                for a_line in mol_struc:
                    this_line = [str(a_num) + "\t" for a_num in a_line]
                    this_line.append("\n")
                    strucFile.write("".join(this_line))
                strucFile.write("}END\n")
        return None
    
    @staticmethod
    def write_energy_file(energy_file_obj,
                          file_path):
        """
        Writes the energy file to the specified path
        
        :param energy_file_obj: The output of
        :param file_path:
        :return: None
        """
        assert isinstance(energy_file_obj, EnergyUtils_Gen_File), "This method requires EnergyUtils_Gen_File objects."
        dum_file = energy_file_obj.get_EnFile()
        with open(file_path, "w") as myFile:
            for aline in dum_file:
                myFile.write(aline + "\n")
        return None
    
    @staticmethod
    def create_dirs_from_list(list_of_dirs):
        """
        Iterate over the given list to generate all the directories.
        
        :return:
        """
        for dirID, a_dir in enumerate(list_of_dirs):
            try:
                os.makedirs(a_dir)
            except OSError:
                pass
        
        return None
    
    @staticmethod
    def loop_over_dir_list(list_of_dirs, _passed_func, **kwargs):
        """
        Given the list of directories, we loop over each directory, change-directory into that directory,
        perform the function _passed_func(**kwargs) where **kwargs are assumed to be for _passed_func, store the
        return-values of that function, and continue.
        At the end, we change directory back to where we started evaluating this function.
        
        :param list_of_dirs
        :param _passed_func:
        :param kwargs:
        :return: list_of_return_vals
        """
        dum_start_dir = os.getcwd()
        ret_vals = []
        for dirID, a_dir in enumerate(list_of_dirs):
            os.chdir(a_dir)
            ret_vals.append(_passed_func(**kwargs))
        os.chdir(dum_start_dir)
        
        return ret_vals
    
    @staticmethod
    def loop_over_nested_dirs(nested_list, _passed_func, **kwargs):
        """
        Given a nested list of directories, we loop over each directory, and perform the function.
        The shape of the returned list will match that of the directory list.
        This is the nested version of loop_over_dirs_list()
        
        :param nested_list
        :param _passed_func:
        :param kwargs:
        :return:
        """
        dum_start_dir = os.getcwd()
        per_box_vals = []
        for boxID, a_box in enumerate(nested_list):
            per_mol_vals = []
            for molID, a_mol in enumerate(a_box):
                per_rep_vals = []
                for repID, a_dir in enumerate(a_mol):
                    os.chdir(a_dir)
                    per_rep_vals.append(_passed_func(**kwargs))
                per_mol_vals.append(per_rep_vals)
            per_box_vals.append(per_mol_vals)
        
        os.chdir(dum_start_dir)
        
        return per_box_vals
    
    @staticmethod
    def read_nploadtxt(file_name='__CLUS.dat'):
        """
        To be used with directory looping wrapper functions, reads in file_name using np.loadtxt and returns the array.
        
        :return:
        """
        return np.loadtxt(file_name)
    
    @staticmethod
    def loop_function_over_deeply_nested_run_conditions_dict(nested_dict: _NestedRunConditions,
                                                             passed_func, file_name: str) -> dict:
        """
        Given a _NestedRunConditions object, we loop over all run conditions and perform a function on a particular file.
        This assumes that the file_name is the same in every directory, but that each run condition is in its own directory.
        This is usually the case if LaSSI simulations are run using the functions in this module.
        Lastly, this also assumes that the function being passed takes a file_path as its argument.
        :param nested_dict:
        :param passed_func:
        :param file_name:
        :return:
        """
        
        run_conds = nested_dict.NestedRunConditions
        dir_pre   = nested_dict.DirPrefix
        _thisDict = {}
        for aBox, boxMols in run_conds.items():
            _thisDict[aBox] = {}
            for aMol, molReps in boxMols.items():
                _thisDict[aBox][aMol] = {}
                for aRep, repVal in molReps.items():
                    full_name = f"{dir_pre}/{aBox}/{aMol}/WInt/{aRep}/{file_name}"
                    _thisDict[aBox][aMol][aRep] = passed_func(full_name)
                    
        return _thisDict

    @staticmethod
    def loop_function_over_deeply_nested_run_conditions_dict_only_successful(nested_dict: _NestedRunConditions,
                                                             passed_func, file_name: str) -> dict:
        """
        Given a _NestedRunConditions object, we loop over all run conditions and perform a function on a particular file.
        This assumes that the file_name is the same in every directory, but that each run condition is in its own directory.
        This is usually the case if LaSSI simulations are run using the functions in this module.
        Lastly, this also assumes that the function being passed takes a file_path as its argument.
        This function only performs the function if the run was successful, which assumes that the last value is a boolean.
        :param nested_dict:
        :param passed_func:
        :param file_name:
        :return:
        """
        assert nested_dict.QSuccess, "Can only be run on nested dicts that have checked for run success!"
        run_conds = nested_dict.NestedRunConditions
        dir_pre = nested_dict.DirPrefix
        _thisDict = {}
        for aBox, boxMols in run_conds.items():
            _thisDict[aBox] = {}
            for aMol, molReps in boxMols.items():
                _thisDict[aBox][aMol] = {}
                for aRep, repVal in molReps.items():
                    if repVal:
                        full_name = f"{dir_pre}/{aBox}/{aMol}/WInt/{aRep}/{file_name}"
                        _thisDict[aBox][aMol][aRep] = passed_func(full_name)
                    else:
                        _thisDict[aBox][aMol][aRep] = None
        return _thisDict
    
    @staticmethod
    def loop_function_over_deeply_nested_run_conditions_dict_only_failed(nested_dict: _NestedRunConditions,
                                                             passed_func, file_name: str) -> dict:
        """
        Given a _NestedRunConditions object, we loop over all run conditions and perform a function on a particular file.
        This assumes that the file_name is the same in every directory, but that each run condition is in its own directory.
        This is usually the case if LaSSI simulations are run using the functions in this module.
        Lastly, this also assumes that the function being passed takes a file_path as its argument.
        This function only performs the function if the run failed.
        :param nested_dict:
        :param passed_func:
        :param file_name:
        :return:
        """
        assert nested_dict.QSuccess, "Can only be run on nested dicts that have checked for run success!"
        run_conds = nested_dict.NestedRunConditions
        dir_pre = nested_dict.DirPrefix
        _thisDict = {}
        for aBox, boxMols in run_conds.items():
            _thisDict[aBox] = {}
            for aMol, molReps in boxMols.items():
                _thisDict[aBox][aMol] = {}
                for aRep, repVal in molReps.items():
                    if not repVal:
                        full_name = f"{dir_pre}/{aBox}/{aMol}/WInt/{aRep}/{file_name}"
                        _thisDict[aBox][aMol][aRep] = passed_func(full_name)
                    else:
                        _thisDict[aBox][aMol][aRep] = None
        return _thisDict

    @staticmethod
    def loop_function_over_deeply_nested_run_conditions_dict_only_successful_with_min_reps(
            nested_dict: _NestedRunConditions,
            passed_func, file_name: str, min_reps: int) -> dict:
        """
        Given a _NestedRunConditions object, we loop over all run conditions and perform a function on a particular file.
        This assumes that the file_name is the same in every directory, but that each run condition is in its own directory.
        This is usually the case if LaSSI simulations are run using the functions in this module.
        Lastly, this also assumes that the function being passed takes a file_name as its argument.
        This function only performs the function if the run was successful, which assumes that the last value is a boolean.
        Here we only perform the function for min_reps. The use case being if repID=0 somehow failed, by the other reps
        finished successfully. So we can pass over that replicate, for that condition. This will preserve the overall
        nested structure of the dict, which should be converted to a list later.
        :param nested_dict:
        :param passed_func:
        :param file_name:
        :return:
        """
        assert nested_dict.QSuccess, "Can only be run on nested dicts that have checked for run success!"
        assert nested_dict.check_if_at_least_n_reps_successful(min_reps), f"This system does not have at least {min_reps}."
        run_conds = nested_dict.NestedRunConditions
        dir_pre = nested_dict.DirPrefix
        _thisDict = {}
        for aBox, boxMols in run_conds.items():
            _thisDict[aBox] = {}
            for aMol, molReps in boxMols.items():
                _thisDict[aBox][aMol] = {}
                rep_it = 0
                for aRep, repVal in molReps.items():
                    if repVal and rep_it < min_reps:
                        full_name = f"{dir_pre}/{aBox}/{aMol}/WInt/{aRep}/{file_name}"
                        _thisDict[aBox][aMol][aRep] = passed_func(full_name)
                        rep_it += 1
        return _thisDict
    

class SamplingUtils(object):
    """
    Collection of functions that help in sampling different concentrations for multi-component systems.
    In order to generate the different concentrations, we not only need to sample different number combinations,
    or stoichiometries, but also different box sizes.

    There are two main ways to sample.
    - Regular grid-like sampling in N-dimensions where a hybrid (log & linear) sampling of the molecule numbers
    generates a hypersurface of constant concentration for a single box-size. Then we can sample more boxes to generate
    more hypersurfaces.
    - Orthogonal-like sampling in N-dimensions where we start the sampling at a pre-determined total concentration.
    Then we generate a large array of molecule numbers and boxes, and filter to find the combinations of box-size
    and molecule-numbers that ensure for a given component being varied, all other component concentrations are constant.
    """
    
    @staticmethod
    def mol_nums_from_linear(mol_min: int = 100, mol_max: int = 2000, num_of_bins: int = 15):
        """
        Given the minimum and maximum molecule numbers, we generate a linear sampling.
        This is just a convenient wrapper for np.linspace
        """
        lin_mols = np.linspace(mol_min, mol_max - mol_min, num_of_bins + 1)
        lin_mols = np.array(lin_mols, dtype=int)
        return lin_mols
    
    @staticmethod
    def mol_nums_from_log(mol_min: int = 100, mol_max: int = 2000, num_of_bins: int = 15):
        """
        Given the minimum and maximum molecule numbers, we generate a log sampling.
        We log-sample until mol_max/2, then reverse the sampling to generate an even sampling around
        mol_max/2
        """
        actual_bins = int(np.floor((num_of_bins) / 2 + 1))
        log_mols = np.ceil(10.0 ** np.linspace(np.log10(mol_min), np.log10(mol_max / 2.), actual_bins))
        log_mols = np.array(log_mols, dtype=int)
        log_mols = np.sort(np.append(log_mols, mol_max - log_mols[-2::-1]))
        
        return log_mols
    
    @staticmethod
    def mol_nums_from_log_and_linear(mol_min: int = 100, mol_max: int = 2000, num_of_bins: int = 15):
        """
        Combining the two sampling techniques to generate the hybrid sampling.
        """
        lin_sam = SamplingUtils.mol_nums_from_linear(mol_min, mol_max, num_of_bins)
        log_sam = SamplingUtils.mol_nums_from_log(mol_min, mol_max, num_of_bins)
        
        dum_sam = np.append(lin_sam, log_sam)
        dum_sam = np.unique(dum_sam)
        
        return dum_sam
    
    @staticmethod
    def mol_nums_for_N_comps(num_of_comps: int = 2, mol_min: int = 100, mol_max: int = 2000, num_of_bins: int = 15):
        """
        Used to generate a regular grid of hybrid sampled N-components.
        Suppose M is the mol_max.
        For 1-component, we just return back the hybrid sampling.
        For 2-components, we generate the hybrid sampling for 1-component. Then, since M=m1+m2, m2=M-m1.
        For 3-components and more, we have an iterative process:
            - We generate the hybrid sampling for 1-component, call it R.
            - Using R_{i=0}, we have now M' = M - R_{i=0}, which is used as a new max_mols for N-1 components.
                - This keeps iterating downwards till we reach N=2.
            - We then iterate over all R_i values
        """
        assert num_of_bins % 2 == 1, "Bin number has to be odd."
        hybrid_sam = SamplingUtils.mol_nums_from_log_and_linear(mol_min, mol_max, num_of_bins)[::2]
        tot_sam = []
        if num_of_comps == 1:
            tot_sam.append(hybrid_sam)
            return np.array(tot_sam).T
        if num_of_comps == 2:
            tot_sam.append(hybrid_sam)
            tot_sam.append(mol_max - hybrid_sam)
            return np.array(tot_sam).T
        else:
            sup_dum_sam = []
            for numID, a_num in enumerate(hybrid_sam[:]):
                if mol_max - a_num < 2 * mol_min:
                    continue
                dum_sam = SamplingUtils.mol_nums_for_N_comps(num_of_comps=num_of_comps - 1,
                                                             mol_min=mol_min,
                                                             mol_max=mol_max - a_num,
                                                             num_of_bins=num_of_bins).T
                temp_sam = np.zeros((num_of_comps, dum_sam.shape[-1]), int)
                temp_sam[0] = a_num
                temp_sam[1:] = dum_sam
                sup_dum_sam.append(temp_sam.T)
            tot_sam = [a_trip for a_set in sup_dum_sam for a_trip in a_set]
            return np.array(tot_sam)
    
    @staticmethod
    def mol_nums_for_fixed_conc_for_boxes(list_of_boxes : np.ndarray = np.array([100, 110, 120, ]), comp_conc: float = 1e-3):
        r"""
        Given the list of boxes, and the intended concentration, we calculate the molecule_numbers that would produce
        the given concentration. We have that
        $$\rho = N / V = N / L^3 \rightarrow N=\rho L^{3},$$
        where $N$ has to be an integer.
        
        :param list_of_boxes:
        :param comp_conc:
        :return:
        """
        
        assert comp_conc <= 1, "Density cannot be greater than 1."
        
        dum_mols = comp_conc * (list_of_boxes**3.)
        dum_mols = np.array(dum_mols, dtype=int)
        return dum_mols
        
    @staticmethod
    def box_lens_from_log(low_con: float = 1e-6, high_con: float = 1e-2, tot_beads: int = 2e5, tot_boxes: int = 11):
        """
        Given the high and low concentrations, we generate box-sizes that are linearly spaced in log-space
        
        :param low_con:
        :param high_con:
        :param tot_beads:
        :param tot_boxes:
        :return: box-sizes
        """
        assert 0. < low_con < high_con, "Low conc should be lower than high conc!"
        dum_li = np.linspace(np.log10(low_con), np.log10(high_con), tot_boxes)
        dum_ar = 10. ** dum_li
        dum_ar = tot_beads / dum_ar
        dum_ar = np.array(dum_ar ** (1. / 3.), dtype=int)
        dum_ar_s = np.sort(dum_ar)
        return dum_ar_s
    
    @staticmethod
    def box_lens_from_linear(low_con: float = 1e-6, high_con: float = 1e-2, tot_beads: int = 2e5, tot_boxes: int = 11):
        """
        Given the high and low concentrations, we generate box-sizes that are linearly spaced.
        
        :param low_con:
        :param high_con:
        :param tot_beads:
        :param tot_boxes:
        :return: box-sizes
        """
        assert 0. < low_con < high_con, "Low conc should be lower than high conc!"
        box_lo = int((tot_beads / low_con) ** (1. / 3.))
        box_hi = int((tot_beads / high_con) ** (1. / 3.))
        dum_li = np.linspace(box_lo, box_hi, tot_boxes, dtype=int)
        dum_ar = np.sort(dum_li)
        return dum_ar
    
    @staticmethod
    def box_lens_from_log_and_linear(low_con: float = 1e-6, high_con: float = 1e-2, tot_beads: int = 2e5, tot_boxes: int = 11):
        """
        Use both box-sampling techniques to generate a hybrid box-size array
       
        :param low_con:
        :param high_con:
        :param tot_beads:
        :param tot_boxes:
        :return:
        """
        lin_box = SamplingUtils.box_lens_from_linear(low_con, high_con, tot_beads, tot_boxes)
        log_box = SamplingUtils.box_lens_from_log(low_con, high_con, tot_beads, tot_boxes)
        dum_box = np.append(lin_box, log_box)
        return np.unique(dum_box)[::2]
    
    @staticmethod
    def box_lens_for_target_conc_of_comp_from_mol_list(list_of_mol_nums=np.array([[100, 200, 300], [200, 100, 300]]),
                                                       this_comp: int = 0,
                                                       target_con=1e-2):
        """
        Given a list of molecule numbers, we generate box sizes such that the concentration
        of 1 of the K-components is at the target_conc
        
        :param list_of_mol_nums:
        :param this_comp: Which component to look at in particular to set the target concentration
        :param target_con:
        :return: list_of_boxes
        """
        
        dum_boxes = list_of_mol_nums.T[this_comp] / target_con
        dum_boxes = dum_boxes ** (1. / 3.)
        dum_boxes = np.array(dum_boxes, int)
        
        return dum_boxes
    
    @staticmethod
    def box_lens_for_target_conc_of_all_from_mol_list(list_of_mol_nums=np.array([[100, 200, 300], [200, 100, 300], ]),
                                                      target_con=1e-2):
        """
        Given the target concentration and the list of mol_nums, we calculate what the box-size should be if each of the
        individual components' concentration was at target_con
        
        :param list_of_mol_nums:
        :param target_con:
        :return:
        """
        return np.array([np.array((a_comp / target_con) ** (1. / 3), dtype=int) for a_comp in list_of_mol_nums.T]).T
    
    @staticmethod
    def ortho_filter_boxes(list_of_boxes=np.array([[100, 100], [110, 110]]),
                           this_comp=-1):
        """
        Given a generated list of box-sizes for a target concentration, we select box-sizes where only 1-component,
        *this_comp*, varies while all others are more-or-less constant. The slight differences are due to lattice-sizes
        being integer values. We go over the list_of_boxes, which have the same shape as the molecule_numbers provided
        to box_lens_for_target_conc_of_all_from_mol_list. Thus, we ignore *this_comp* from the *list_of_boxes*. Then, we
        pick the rows where the box-sizes are all the same. We output both the ids and the list. The ids are returned so
        that molecule_numbers can also be filtered later, if needed.
        
        
        :param list_of_boxes: Box-list to filter.
        :param this_comp: This component is allowed to vary, and thus that component's boxes are not considered.
        :return: list_of_ids, list_of_boxes
        """
        
        #Deleting the boxes which can vary.
        dum_boxes = np.delete(list_of_boxes.T, this_comp, axis=0).T
        list_of_ids = []
        list_of_boxes = []
        for setID, a_set in enumerate(dum_boxes):
            dum_un = np.unique(a_set)
            if len(dum_un) == 1:
                list_of_ids.append(setID)
                list_of_boxes.append(a_set[0])
        return np.array(list_of_ids), np.array(list_of_boxes)
    
    @staticmethod
    def ortho_mol_roll_set(list_of_mol_nums: np.ndarray = np.array([[100, 200, 300], [200, 100, 300],]),
                           roll_nums: int = 3):
        """
        Given the molecule numbers, we produce a set of permutations of each of the molecule numbers, using
        numpy.roll *roll_nums* number of times per set. This should produce a set of molecule numbers that with the
        correct box-sizes produce orthogonal lines.
    
        :param list_of_mol_nums:
        :param roll_nums:
        :return: numpy.ndarray containing permutations of molecule numbers.
        """
        dum_list_of_mols = []
        for setID, a_set in enumerate(list_of_mol_nums):
            dum_mol = [np.roll(a_set, a_comp) for a_comp in range(roll_nums)]
            dum_list_of_mols.append(dum_mol)
            
        return np.array(dum_list_of_mols)
        
    @staticmethod
    def ortho_mol_roll_with_fixed_comp(list_of_mol_nums: np.ndarray = np.array([[100, 200, 300], [200, 100, 300],]),
                                       this_comp: int = 2, roll_nums: int = 3):
        """
        Produce combinations of mol-nums where *this_comp* is not sampled over. The implementation is inefficient and
        dumb. I delete *this_comp* from every combination of molecule numbers, where *roll_nums* number of rolls are
        performed. This is similar to ortho_mol_roll_set. Note that roll_nums can be larger than the number of molecules
        per-set (or SystemSetup.CompNum) so an incorrectly rolled set of molecule numbers could be produced.
        
        :param list_of_mol_nums:
        :param this_comp:
        :param roll_nums:
        :return:
        """

        dum_list_of_mols = []
        for molID, a_mol in enumerate(list_of_mol_nums):
            this_set = np.delete(a_mol, this_comp)
            rolled_set = [np.insert(np.roll(this_set, a_comp), this_comp, a_mol[this_comp]) for a_comp in range(roll_nums)]
            dum_list_of_mols.append(np.array(rolled_set))

        return np.array(dum_list_of_mols)


class SystemSetup(object):
    """
    General class that is used to store a particular system. A system is defined as having a unique interaction
    or topology set.
    """
    
    def __init__(self, comp_num: int = 2, mol_min: int = 100, mol_max: int = 2000):
        self.CompNum = comp_num
        assert self.CompNum > 0, "Number of components must be more than 1!"
        
        self.MinMax = tuple([mol_min, mol_max])
        assert self.MinMax[0] <= self.MinMax[1], "MinMax should be (min, max), in that order where max>=min."
        
        self.Structure = []
        self.StrucQ = False
        self.WIntEnergyFile = ''
        self.WIntEnQ = False
        self.NoIntEnergyFile = ''
        self.NoIntEnQ = False
        self.Boxes = []
        self.BoxesQ = False
        self.MolNums = []
        self.MolNumsQ = False
        self.OrthoQ = False
        self.OrthoQ_fComp = None
    
    def reset_struc(self):
        """
        For the given SysName, we reinitialize the array that holds the topology/structure
        
        :return:
        """
        self.Structure = []
        self.StrucQ = False
        return None
    
    def add_struc(self, sys_struc_ar):
        """
        Given a full structure array, like the ones produced by TopoUtils_Gen_Linear_System,
        we add the structure to this system.
        
        :param sys_struc_ar:
        :return:
        """
        assert len(
                sys_struc_ar) == self.CompNum, "Added structure does not match the number of components defined for this system."
        self.Structure = sys_struc_ar
        self.StrucQ = True
        return None
    
    def gen_struc_from_mols(self, mol_list):
        """
        Given a set of molecule numbers, we renumber the molecules.
        
        :param mol_list:
        :return:
        """
        
        for compID, a_comp in enumerate(mol_list):
            self.Structure[compID][0] = a_comp
        return None
    
    def set_MinMax(self, mol_min: int = 100, mol_max: int = 2000):
        self.MinMax = tuple([mol_min, mol_max])
        return None
    
    def set_WInt_energy_file(self, file_name: str):
        """
        Set the absolute path for the WInt_energy_file for this system.
        
        :param file_name:
        :return: None
        """
        self.WIntEnergyFile = file_name
        self.WIntEnQ = True
    
    def set_NoInt_energy_file(self, file_name: str):
        """
        Set the absolute path for the NoInt_energy_file for this system.
        
        :param file_name:
        :return: None
        """
        self.NoIntEnergyFile = file_name
        self.NoIntEnQ = True
    
    def set_hybrid_molecule_numbers(self, num_of_bins: int = 15):
        """
        Generate and set self.MolNums using mol_nums_for_N_comps()
        :param num_of_bins: Estimated number of bins per component.
        :return: None
        """
        mol_min, mol_max = self.MinMax
        self.MolNums = SamplingUtils.mol_nums_for_N_comps(num_of_comps=self.CompNum,
                                                          mol_min=mol_min,
                                                          mol_max=mol_max,
                                                          num_of_bins=num_of_bins)
        self.MolNumsQ = True
        return None
    
    def set_MolNums(self, mol_nums: np.ndarray, verbose: bool = True):
        """
        Manually set the list of list of molecule numbers. Remember that this ignores the preset molMin and molMax
        
        :param mol_nums: np.ndarray(n , self.CompNum)
        :return:
        """
        assert isinstance(mol_nums, np.ndarray), "Mol Nums should be a NumPy array of NumPy arrays!"
        assert isinstance(mol_nums[0], np.ndarray), "Mol Nums should be a NumPy array of NumPy arrays!"
        assert mol_nums.shape[-1] == self.CompNum, "Mol Nums does not have the correct number of components"
        self.MolNums = []
        
        for molID, a_mol in enumerate(mol_nums):
            self.MolNums.append(a_mol)
        
        self.MolNums = np.array(self.MolNums)
        self.MolNumsQ = True
        if verbose:
            print("Remember that this method ignores mol_min and mol_max.")
        return None
    
    @staticmethod
    def calc_struc_beads_in_molecule(mol_struc):
        """
        Given a molecule structure, we calculate the total number of beads
        
        :param mol_struc:
        :return: number of beads in molecule (int)
        """
        this_struc = mol_struc[1][:]  # 0th index is for molecule number, so we pick 1
        bead_num = len(np.unique(this_struc.T[0]))
        return bead_num
    
    def calc_struc_beads_per_molecule(self):
        """
        Loop over each component in the structure to calculate the number of beads.
        
        :return: nd.array where each component is the number of beads for that component.
        """
        
        bead_nums = np.zeros(self.CompNum, int)
        for compID, a_struc in enumerate(self.Structure):
            bead_nums[compID] = self.calc_struc_beads_in_molecule(a_struc)
        return bead_nums
    
    def calc_struc_beads_max(self):
        """
        We loop over the structure for this system and calculate the total number of beads per molecule.
        
        :return: The maximum number of beads in this system.
        """
        assert self.StrucQ, "The structure has not been defined"
        mol_bead_nums = self.calc_struc_beads_per_molecule()
        bead_nums = np.zeros(len(self.MolNums), int)
        for setID, a_set in enumerate(self.MolNums):
            bead_nums[setID] = np.dot(a_set, mol_bead_nums)
        max_arg = bead_nums.argmax()
        if self.OrthoQ:
            return [max_arg, bead_nums[max_arg], self.MolNums[max_arg], self.Boxes[max_arg]]
        else:
            return [max_arg, bead_nums[max_arg], self.MolNums[max_arg]]
    
    def set_regular_boxes(self, low_con=1e-6, high_con=1e-2, tot_boxes: int = 11, verbose: bool = False):
        """
        We calculate the maximum number of beads for this system, and generate the hybrid_sampled box-sizes.
        
        :param low_con:
        :param high_con:
        :param tot_boxes:
        :param verbose:
        :return:
        """
        assert self.MolNumsQ, "The molecule numbers need to be defined already!"
        dum_beads = self.calc_struc_beads_max()
        tot_beads = dum_beads[1]
        self.Boxes = SamplingUtils.box_lens_from_log_and_linear(low_con, high_con, tot_beads, tot_boxes)
        self.BoxesQ = True
        if verbose:
            print(rf"Smallest box is {self.Boxes.min():}, and largest box is {self.Boxes.max():}")
        return None
    
    def set_boxes_to(self, list_of_boxes=np.array([100, 110, 120]), verbose=False):
        self.Boxes = list_of_boxes[:]
        self.BoxesQ = True
        if verbose:
            print(r"Smallest box is {:}, and largest box is {:}".format(self.Boxes.min(), self.Boxes.max()))
        return None
    
    def set_ortho_boxes(self, mol_num_boxes=11, target_con=1e-2, comp_ig: int = -1):
        """
        Given the target concentration for all-components other than comp_ig, we generate the set of boxes
        and molecule numbers such that only 1 out of K-components varies. Simple cyclic
        permutations of the molecule-numbers per box-size produces different components to vary.
        
        :param mol_num_boxes: estimated number of bins -- usually a severe underestimate
        :param target_con: estimated target concentrations
        :param comp_ig: this component is allowed to vary
        :return:
        """
        self.OrthoQ = True
        self.set_hybrid_molecule_numbers(num_of_bins=mol_num_boxes)
        
        raw_molecule_ar = self.MolNums.copy()
        raw_box_ar = SamplingUtils.box_lens_for_target_conc_of_all_from_mol_list(raw_molecule_ar, target_con)
        
        pruned_ids, new_box_ar = SamplingUtils.ortho_filter_boxes(raw_box_ar, comp_ig)
        
        self.Boxes = new_box_ar[:]
        self.BoxesQ = True
        self.MolNums = raw_molecule_ar[pruned_ids]
        print("You have to make sure that the box-size is large enough manually!")
        
        return None
    
    @staticmethod
    def check_if_conds_can_be_ortho(mol_nums : np.ndarray, box_list : np.ndarray) -> bool:
        """
        Given the list of molecule numbers and box-sizes, we check if the number of boxes and molecules is the same.
        If so, we return a True.
        
        :param mol_nums:
        :param box_list:
        :return: bool: If the mol_nums and box_list _could_ represent an ortho-compatible system.
        """
        
        if mol_nums.shape[0] == box_list.shape[0]:
            return True
        else:
            return False
    
    def check_ortho_compatibility(self, verbose: bool = True):
        """
        For this system, we check if the defined molecule-numbers and box-sizes _can_ be represented as ortho-slices
        
        :return: None
        """
        
        assert self.MolNumsQ, "Molecule numbers have not been set!"
        assert self.BoxesQ, "Box-sizes have not been set!"
        
        self.OrthoQ = self.check_if_conds_can_be_ortho(self.MolNums, self.Boxes)
        assert self.OrthoQ, "System is not Ortho-compatible!"
        if verbose:
            print("This system is Ortho compatible.")
        return None
    
    def set_ortho_fixed_component(self, this_comp: int = 2):
        """
        Given a system that has ortho-sampling for K-components. *this_comp* is not looped over when sampling.
        This is intended for systems where one of the components is to be held at a fixed concentration. Therefore,
        the molecule-numbers associated with *this_comp* are not to change, and are unique and thus cannot be
        used in np.roll() when generating to-be-sampled molecule numbers.
        
        :param this_comp:
        :return: None.
        """
        assert self.OrthoQ, "System should be ortho-compatible before setting a fixed component."
        assert this_comp <= self.CompNum, f"this_comp cannot be larger than {self.CompNum}."
        assert this_comp >= 0, "this_comp needs to be positive."
        self.OrthoQ_fComp = this_comp
    
    # TODO: Add a function that checks if the ortho boxes are such that there are no conditions with _bead_ density
    # higher than 1.0.
    
    def get_number_of_conditions(self):
        """
        Calculate the total number of runs independent concentrations for this system. Let R be the number of
        independent concentrations
        If a regular grid, then we have N x M where N is the number of molecules sampled, and M is the number of
        boxes sampled
        If an orthogonal grid, then R = K x N where we have K components.
        
        :return:
        """
        assert self.BoxesQ, "The boxes for this system have not been defined!"
        assert self.MolNumsQ, "The molecule numbers for this system have not been calculated!"
        
        if not self.OrthoQ:
            return len(self.MolNums) * len(self.Boxes)
        else:
            if self.OrthoQ_fComp:
                return (self.CompNum - 1) * len(self.MolNums)
            else:
                return self.CompNum * len(self.MolNums)
    
    def get_independent_conditions(self):
        """
        For this system, we return the box-sizes and their corresponding molecule number sets.
        If a regular grid, then we have the same mol_num list for each box-size.
        If an ortho-grid, then for each box-size, we have the cycles of the molecule-numbers.
        
        :return: [box_list, list_of_list_of_mol_nums]
        """
        assert self.BoxesQ, "The boxes for this system have not been defined!"
        assert self.MolNumsQ, "The molecule numbers for this system have not been calculated!"
        
        list_of_mol_nums = []
        
        if self.OrthoQ:
            if self.OrthoQ_fComp is not None:
                list_of_mol_nums = SamplingUtils.ortho_mol_roll_with_fixed_comp(self.MolNums,
                                                                                self.OrthoQ_fComp,
                                                                                self.CompNum-1)
            else: #No fixed component.
                list_of_mol_nums = SamplingUtils.ortho_mol_roll_set(self.MolNums,
                                                                    self.CompNum)
        else: #Normal grid.
            list_of_mol_nums = np.array([self.MolNums] * len(self.Boxes))
            
        return np.copy(self.Boxes), list_of_mol_nums


class SimulationSetup(object):
    """
    Main class used to instantiate systems that will be simulated.
    """
    
    def __init__(self, key_file, sys_name_list=None, simulations_path=None, num_of_reps: int=2, param_verbose: bool = True):
        """

        :param key_file: Absolute path to the key-file for this instance
        :param sys_name_list: The list of the unique system names.
        :param simulations_path: The absolute path where the simulations will take place.
        """
        
        if sys_name_list is None:
            sys_name_list = ["SysA"]
        
        self.KeyFileTemplate = IOUtils.read_param_file(file_name=key_file, verbose=param_verbose)
        
        self.CurrentDirectory = self._gen_curDir()
        self.Sims_Path = self._gen_simsPath(sims_path=simulations_path)
        self.Data_Path = self._gen_dataPath(sims_path=simulations_path)
        for a_dir in [self.Sims_Path, self.Data_Path]:
            try:
                os.makedirs(a_dir)
            except OSError:
                pass
        self.PathToLASSI = '/project/fava/packages/bin/lassi'
        self.SysNames = sys_name_list[:]
        self.Num_Reps = num_of_reps
        self.Num_Temps = int(self.KeyFileTemplate[1]['MC_CYCLE_NUM'])
        self.SysInfo = {}
        for a_sys in sys_name_list:
            self.SysInfo[a_sys] = SystemSetup()
            for a_dir in [self._gen_struc_dir_prefix_for(a_sys),
                          self._gen_data_dir_prefix_for(a_sys)]:
                try:
                    os.makedirs(a_dir)
                except OSError:
                    pass
        self.RunInfo: Dict[str, _NestedRunConditions] = {}
        return None

    def _gen_run_conditions_dict_for_sys(self, sysName: str = "SysA"):
        """
        For this system, we generate a deeply nested run conditions dictionary, of type _NestedRunConditions
        :param sysName:
        :return:
        """
        thisSys = self.SysInfo[sysName]
        
        assert thisSys.BoxesQ, f"{sysName} does not have set boxes!"
        assert thisSys.MolNumsQ, f"{sysName} does not have set molecule numbers!"
        
        boxList = thisSys.Boxes
        molList = thisSys.MolNums
        dirPre = self._gen_sims_dir_prefix_for(sysName)
    
        _tmpDict = _NestedRunConditions(prefix=dirPre,
                                        box_list=boxList,
                                        mol_list=molList,
                                        rep_num=self.Num_Reps)
    
        self.RunInfo[sysName] = _tmpDict
        return None
    
    def gen_run_conditions_dict_forAll(self):
        """
        Loops over all systems and generates deeply nested run conditions dictionary --  of type _NestedRunConditions
        :return:
        """
        for aSys in tqdm(self.SysInfo):
            self._gen_run_conditions_dict_for_sys(aSys)
        return None
    
    def _check_for_run_success_for_sys(self, sysName: str = "SysA"):
        """
        Loops over all the run conditions for this system and checks if the simulations ran successfully
        Applies _NestedRunConditions().check_for_run_success().
        :param sysName:
        :return:
        """
        self.RunInfo[sysName].check_for_run_success()
        return None
    
    def check_for_run_success_forAll(self):
        """
        For every system, we loop over all the run conditions and check if the simulations ran successfully
        Applies _NestedRunConditions().check_for_run_success() to each system.
        :return:
        """
        for aSys in tqdm(self.SysInfo):
            self._check_for_run_success_for_sys(aSys)
        return None
    
    def print_run_success_forAll(self):
        """
        Assuming that we have checked for the success state of every simulation, we print out a table of if n-number of
        replicates have finished for each run condition for each system.
        Runs _NestedRunConditions.check_if_at_least_n_reps_successful on every system, and prints out the information
        in a nicer format.
        :return:
        """
        reps = self.Num_Reps
        title_str = [f"{'SysName':<20}"]
        for aRep in range(1, reps + 1):
            title_str.append(f"{str(aRep) + ' Reps.':<10}")
        title_str = "| " + "".join(title_str) + "|"
        print("-" * len(title_str))
        print(title_str)
        print("-" * len(title_str))
        for sysName, aSys in self.RunInfo.items():
            _repStr = "".join(
                    [f"{str(aSys.check_if_at_least_n_reps_successful(aRep)):<10}" for aRep in range(1, reps + 1)])
            print(f"| {sysName:<20}{_repStr}|")
        print("-" * len(title_str))
        
        return None
    
    def _collect_timings_for_nReps_for_sys(self, sysName: str, file_name: str, min_reps: int) -> list:
        """
        We collect the data for the given file_name, where we should have at least nReps successful
        runs, and return a deeply nested list of that data. The returned list should have the format
        [perBox][perMol][perRep]
        :param sysName:
        :param file_name:
        :param min_reps:
        :return:
        """
        thisSys = self.RunInfo[sysName]
        gathering_func = IOUtils.loop_function_over_deeply_nested_run_conditions_dict_only_successful_with_min_reps
        timings_func   = JobSubmission.read_log_for_timing
        _tmpDict = gathering_func(nested_dict=thisSys,
                                  passed_func=JobSubmission.read_log_for_timing,
                                  file_name=file_name,
                                  min_reps=min_reps)
        convert_func = _NestedRunConditions._convert_nested_dict_to_nested_list
        return convert_func(_tmpDict)
        
    def _collect_raw_data_for_fileName_for_nReps_for_sys(self, sysName: str, file_name: str, min_reps: int) -> list:
        """
        We collect the data for the given file_name, where we should have at least nReps successful
        runs, and return a deeply nested list of that data. The returned list should have the format
        [perBox][perMol][perRep]
        :param sysName:
        :param file_name:
        :param min_reps:
        :return:
        """
        thisSys = self.RunInfo[sysName]
        gathering_func = IOUtils.loop_function_over_deeply_nested_run_conditions_dict_only_successful_with_min_reps
        _tmpDict = gathering_func(nested_dict=thisSys,
                                  passed_func=np.loadtxt,
                                  file_name=file_name,
                                  min_reps=min_reps)
        convert_func = _NestedRunConditions._convert_nested_dict_to_nested_list
        return convert_func(_tmpDict)
        
    def write_raw_data_for_fileName_for_nReps_for_sys_compressed(self, sysName: str, file_name: str, min_reps: int):
        """
        We collects the data for the given file_name, where we should have at least nReps successful
        runs, and saves to a  gzip compressed file in the appropriate data directory for this system.
        :param sysName:
        :param file_name:
        :param min_reps:
        :return:
        """
        _save_file = self._gen_data_dir_prefix_for(sysName) + file_name + '.b'
        _tmpDat = self._collect_raw_data_for_fileName_for_nReps_for_sys(sysName=sysName,
                                                                        file_name=file_name,
                                                                        min_reps=min_reps)
        
        with gzip.open(_save_file+".gz", "wb") as sFile:
            pickle.dump(_tmpDat, sFile)
        
        return None

    def read_raw_data_for_fileName_for_sys_compressed(self, sysName: str, file_name: str) -> list:
        """
        Reads the given file_name from the appropriate data directory for this system. Assumes that gzip was used to
        generate the compressed data file. Returns a deeply nested list of the data.
        The usual format is [perBox][perMol][perRep]
        :param sysName:
        :param file_name:
        :return:
        """
        _save_file = self._gen_data_dir_prefix_for(sysName) + file_name + '.b'
        _tmpDat = []
        with gzip.open(_save_file + ".gz", "rb") as sFile:
            _tmpDat = pickle.load( sFile)
    
        return _tmpDat
    
    def write_raw_data_for_fileName_for_nReps_forAll_compressed(self, file_name: str, min_reps: int):
        """
        For each system, we collects the data for the given file_name, where we should have at least nReps successful
        runs, and saves to a  gzip compressed file in the appropriate data directory for that system.
        :param file_name:
        :param min_reps:
        :return:
        """
        for aSys in tqdm(self.SysInfo):
            self.write_raw_data_for_fileName_for_nReps_for_sys_compressed(sysName=aSys,
                                                                          file_name=file_name,
                                                                          min_reps=min_reps)
        return None
    
    def read_raw_data_for_fileName_forAll_compressed(self, file_name: str) -> dict:
        """
        Returns a dictionary of loaded data, for the given file_name, for each system. Assumes that the data exists in
        the Data directory for each system, and that the data are compressed using gzip.
        :param sysName:
        :param file_name:
        :param min_reps:
        :return: A dictionary where the system names are the keys and the data are the values.
        """
        _tmpDat = {}
        for aSys in tqdm(self.SysInfo):
            _tmpDat[aSys] = self.read_raw_data_for_fileName_for_sys_compressed(sysName=aSys,
                                                                               file_name=file_name)
        
        return _tmpDat
    
    @staticmethod
    def _gen_curDir():
        """
        Generates the string for the current path. Wrapper for OS.
        
        :return: str for file path.
        """
        
        if os.name == 'nt':
            return os.getcwd() + '\\'
        else:
            return os.getcwd() + '/'

    def _gen_simsPath(self, sims_path=None):
        """
        Return string of directory where Runs shall be
        
        :return:
        """
        if os.name == 'nt':
            if sims_path is None:
                return self.CurrentDirectory + 'Runs\\'
            else:
                return sims_path + 'Runs\\'
        else:
            if sims_path is None:
                return self.CurrentDirectory + 'Runs/'
            else:
                return sims_path + 'Runs/'
    
    def _gen_dataPath(self, sims_path=None):
        """
        Return string of directory where Runs shall be
        
        :return:
        """
        if os.name == 'nt':
            if sims_path is None:
                return self.CurrentDirectory + 'Data\\'
            else:
                return sims_path + 'Data\\'
        else:
            if sims_path is None:
                return self.CurrentDirectory + 'Data/'
            else:
                return sims_path + 'Data/'
    
    def _gen_sims_dir_prefix_for(self, sys_name="SysA"):
        """
        Convenient wrapper to generate the directory prefix for the Runs directory
        
        :param sys_name:
        :return: {str}
        """
        if os.name == 'nt':
            return self.Sims_Path + sys_name + '\\'
        else:
            return self.Sims_Path + sys_name + '/'
    
    def _gen_data_dir_prefix_for(self, sys_name="SysA"):
        """
        Convenient wrapper to generate the directory prefix for the Data directory
        
        :param sys_name:
        :return: {str}
        """
        if os.name == 'nt':
            return self.Data_Path + sys_name + '\\'
        else:
            return self.Data_Path + sys_name + '/'
    
    def _gen_struc_dir_prefix_for(self, sys_name="SysA"):
        """
        Convenient wrapper to generate the directory prefix for the Data directory
        
        :param sys_name:
        :return: {str}
        """
        if os.name == 'nt':
            return self.CurrentDirectory + 'Structures\\' + sys_name + '\\'
        else:
            return self.CurrentDirectory + 'Structures/' + sys_name + '/'
    
    def _get_dirs_list_for(self, sys_name="SysA", wInt: bool=True):
        """
        For this system, return all the directories for this interaction state.
        
        :param sys_name:
        :param wInt:
        :return:
        """
        dum_boxes, dum_mols = self.SysInfo[sys_name].get_independent_conditions()
        dum_dir_list = IOUtils.gen_dir_list(boxList=dum_boxes,
                                            molList=dum_mols,
                                            repNums=self.Num_Reps,
                                            wInt=wInt)
        dum_prefix = self._gen_sims_dir_prefix_for(sys_name=sys_name)
        
        dum_dir_list = [dum_prefix + a_dir for a_dir in dum_dir_list]
        
        return dum_dir_list
    
    def _get_dirs_list_ofAll_for_condition(self, boxSize: int = 100, molNum: list = None, repNum: int = 1, wInt: bool = True):
        """
        Generates a list of the directories for every system, given this specific condition.
        Assumes that all the systems share this condition (better would be that all systems have the same conditions)
        :param boxSize:
        :param molNum:
        :param repNum:
        :param wInt:
        :return:
        """
        
        dum_dir_list = []
        
        dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
        
        for sysID, aSys in enumerate(self.SysInfo):
            thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
            dum_dir_list.append(thisDir+dum_postfix)
        
        return dum_dir_list
    
    def _get_dirs_ofAll_for_boxSize_perRep_perMol_perSys(self, boxSize:int = 100, wInt: bool = True):
        """
        Generates a list of all the directories for the given boxSize where the outer-most looping is per rep.
        Assumes that all the systems share this box-size, at least.
        Inner most loop is over the systems.
        :param boxSize:
        :return:
        """
        
        dum_dir_list = []

        dum_reps = np.arange(self.Num_Reps)+1
        dum_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()
        
        
        for repID, repNum in enumerate(dum_reps):
            for molID, molSet in enumerate(dum_mols):
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molSet, repNum=repNum, wInt=wInt)
                for sysID, aSys in enumerate(self.SysInfo):
                    thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                    dum_dir_list.append(thisDir+dum_postfix)
        
        return dum_dir_list

    def _get_dirs_ofAll_for_molSet_perRep_perBox_perSys(self, molNum:list = None, wInt: bool = True):
        """
        Generates a list of all the directories for the given molNum where the outer-most looping is per rep.
        Assumes that all the systems share this MolNum, at least.
        Inner most loop is over the systems.
        :param boxSize:
        :return:
        """
    
        dum_dir_list = []

        dum_reps = np.arange(self.Num_Reps)+1
        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
    
    
        for repID, repNum in enumerate(dum_reps):
            for boxID, boxSize in enumerate(dum_boxes):
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                for sysID, aSys in enumerate(self.SysInfo):
                    thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                    dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_orhto_for_repNum_perBox_perMol_perSys(self, repNum:int = 1, wInt: bool = True):
        """
        Generates a list of all the directories for the given repNum where the outer-most looping is per Box.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """
        
        assert self.SysInfo[self.SysNames[0]].OrthoQ, "System needs to be Ortho Compatible!"
        
        num_comps = self.SysInfo[self.SysNames[0]].CompNum
        fixed_cmp = self.SysInfo[self.SysNames[0]].OrthoQ_fComp
        
        dum_dir_list = []
    
        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        tmp_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()
        
        if fixed_cmp:
            dum_mols = SamplingUtils.ortho_mol_roll_with_fixed_comp(list_of_mol_nums=tmp_mols,
                                                                    this_comp=fixed_cmp,
                                                                    roll_nums=num_comps-1)
        else:
            dum_mols = SamplingUtils.ortho_mol_roll_set(list_of_mol_nums=tmp_mols, roll_nums=num_comps)
        
        for setID, (boxSize, molSet) in enumerate(zip(dum_boxes, dum_mols)):
            for permID, molNum in enumerate(molSet):
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                for sysID, aSys in enumerate(self.SysInfo):
                    thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                    dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_for_repNum_perBox_perMol_perSys(self, repNum:int = 1, wInt: bool = True):
        """
        Generates a list of all the directories for the given repNum where the outer-most looping is per Box.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """
    
        dum_dir_list = []
    
        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        dum_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()
        
    
        for boxID, boxSize in enumerate(dum_boxes):
            for molID, molNum in enumerate(dum_mols):
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                for sysID, aSys in enumerate(self.SysInfo):
                    thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                    dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_for_repNum_perMol_perBox_perSys(self, repNum:int = 1, wInt: bool = True):
        """
        Generates a list of all the directories for the given repNum where the outer-most looping is per Mol.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """

        dum_dir_list = []

        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        dum_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()


        for molID, molNum in enumerate(dum_mols):
            for boxID, boxSize in enumerate(dum_boxes):
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                for sysID, aSys in enumerate(self.SysInfo):
                    thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                    dum_dir_list.append(thisDir+dum_postfix)

        return dum_dir_list

    def _get_dirs_ofAll_ortho_perRep_perMol_perBox_perSys(self, wInt: bool = True):
        """
        Generates a list of all the directories.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """

        assert self.SysInfo[self.SysNames[0]].OrthoQ, "System needs to be Ortho Compatible!"

        num_comps = self.SysInfo[self.SysNames[0]].CompNum
        fixed_cmp = self.SysInfo[self.SysNames[0]].OrthoQ_fComp
    
        dum_reps = np.arange(self.Num_Reps)+1

        dum_dir_list = []

        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        tmp_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()

        if fixed_cmp:
            dum_mols = SamplingUtils.ortho_mol_roll_with_fixed_comp(list_of_mol_nums=tmp_mols,
                                                                    this_comp=fixed_cmp,
                                                                    roll_nums=num_comps-1)
        else:
            dum_mols = SamplingUtils.ortho_mol_roll_set(list_of_mol_nums=tmp_mols, roll_nums=num_comps)
    
        for repID, repNum in enumerate(dum_reps):
            for setID, (boxSize, molSet) in enumerate(zip(dum_boxes, dum_mols)):
                for permID, molNum in enumerate(molSet):
                    dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                    for sysID, aSys in enumerate(self.SysInfo):
                        thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                        dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_perRep_perMol_perBox_perSys(self, wInt: bool = True):
        """
        Generates a list of all the directories.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """
    
        dum_dir_list = []
        
        dum_reps = np.arange(self.Num_Reps)+1
        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        dum_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()
    
        for repID, repNum in enumerate(dum_reps):
            for molID, molNum in enumerate(dum_mols):
                for boxID, boxSize in enumerate(dum_boxes):
                    dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                    for sysID, aSys in enumerate(self.SysInfo):
                        thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                        dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_perRep_perBox_perMol_perSys(self, wInt: bool = True):
        """
        Generates a list of all the directories.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """
    
        dum_dir_list = []

        dum_reps = np.arange(self.Num_Reps)+1
        dum_boxes = self.SysInfo[self.SysNames[0]].Boxes.copy()
        dum_mols = self.SysInfo[self.SysNames[0]].MolNums.copy()

        for repID, repNum in enumerate(dum_reps):
            for boxID, boxSize in enumerate(dum_boxes):
                for molID, molNum in enumerate(dum_mols):
                    dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum , wInt=wInt)
                    for sysID, aSys in enumerate(self.SysInfo):
                        thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                        dum_dir_list.append(thisDir+dum_postfix)
    
        return dum_dir_list

    def _get_dirs_ofAll_perRep_ofBoxID_ofMolID_perSys(self, wInt: bool = True, boxID: int = 0, molID: int = 0):
        """
        Generates a list of all the directories.
        Assumes that the systems have the same boxes and mol-nums.
        Inner most loop is over the systems.
        :param repNum:
        :return:
        """
    
        dum_dir_list = []
    
        dum_reps = np.arange(self.Num_Reps) + 1
    
        for repID, repNum in enumerate(dum_reps):
            for sysID, aSys in enumerate(self.SysInfo):
                molNum = self.SysInfo[aSys].MolNums[molID].copy()
                boxSize= self.SysInfo[aSys].Boxes[boxID].copy()
                dum_postfix = IOUtils.gen_dir_str(boxSize=boxSize, molSet=molNum, repNum=repNum, wInt=wInt)
                thisDir = self._gen_sims_dir_prefix_for(sys_name=aSys)
                dum_dir_list.append(thisDir + dum_postfix)
    
        return dum_dir_list
        
    def _get_nested_dirs_for(self, sys_name: str="SysA", wInt: bool=True, repList: list = None):
        """
        Generate the fully nested list of directories for every possible run condition for this system.
        
        :param sys_name:
        :param wInt:
        :return:
        """
        if repList is None:
            repList = np.arange(1,self.Num_Reps+1)
        
        dum_boxes, dum_mols = self.SysInfo[sys_name].get_independent_conditions()
        dum_prefix = self._gen_sims_dir_prefix_for(sys_name)
        dum_dirs_nested_list = IOUtils.gen_dir_nested_list(boxList=dum_boxes,
                                                           molList=dum_mols,
                                                           repList=repList,
                                                           wInt=wInt,
                                                           dir_prefix=dum_prefix)
        
        return dum_dirs_nested_list
    
    def create_dirs_for(self, sys_name: str="SysA", wInt: bool=True):
        """
        For this system, we create all the directories for this system with the given interaction state
        
        :param sys_name:
        :return:
        """
        
        dum_dir_list = self._get_dirs_list_for(sys_name=sys_name, wInt=wInt)
        IOUtils.create_dirs_from_list(dum_dir_list)
        
        return None
    
    def write_strucs_for(self, sys_name):
        """
        We write all possible structures for this system to $CURDIR/Structures/sys_name/
        
        :param sys_name:
        :return: None
        """
        this_sys = self.SysInfo[sys_name]
        assert this_sys.StrucQ, "This system does not have a defined structure"
        assert this_sys.MolNumsQ, "This system does not have a calculated MolNums"
        
        tmp_mols = this_sys.MolNums[:].copy()
        dum_prefix = self._gen_struc_dir_prefix_for(sys_name)
        if not this_sys.OrthoQ:
            dum_mols = tmp_mols
            for setID, a_mol in enumerate(dum_mols):
                    this_sys.gen_struc_from_mols(a_mol)
                    dum_name = dum_prefix + "_".join([str(a_num) for a_num in a_mol]) + ".prm"
                    IOUtils.write_struc_file(this_sys.Structure, dum_name)
        else:
            fxd_cmp = this_sys.OrthoQ_fComp
            if fxd_cmp:
                dum_mols = SamplingUtils.ortho_mol_roll_with_fixed_comp(list_of_mol_nums=tmp_mols,
                                                                        this_comp=fxd_cmp,
                                                                        roll_nums=this_sys.CompNum-1)
            else:
                dum_mols = SamplingUtils.ortho_mol_roll_set(list_of_mol_nums=tmp_mols,
                                                            roll_nums=this_sys.CompNum)
            for setID, a_set in enumerate(dum_mols):
                for permID, a_mol in enumerate(a_set):
                    this_sys.gen_struc_from_mols(a_mol)
                    dum_name = dum_prefix + "_".join([str(a_num) for a_num in a_mol]) + ".prm"
                    IOUtils.write_struc_file(this_sys.Structure, dum_name)
        
        return None
    
    def write_params_for(self, sys_name: str = "SysA", wInt: bool = True, fullRand: bool = False, cls_mode: int = 0,
                         repList: list =None):
        """
        Write key-files for LASSI in the appropriate directories
        
        :param repList:
        :param sys_name:
        :param wInt:
        :return:
        """
        this_sys = self.SysInfo[sys_name]
        
        if wInt:
            assert this_sys.WIntEnQ, "Energy file for WInt has not been set yet!"
            dum_en_file = this_sys.WIntEnergyFile
        else:
            assert this_sys.NoIntEnQ, "Energy file for NoInt has not been set yet!"
            dum_en_file = this_sys.NoIntEnergyFile
        
        if repList is None:
            repList = np.arange(1,self.Num_Reps+1)
        
        dum_boxes, dum_mols = this_sys.get_independent_conditions()
        dum_nested_dirs = self._get_nested_dirs_for(sys_name, wInt)
        
        dum_struc_dir_prefix = self._gen_struc_dir_prefix_for(sys_name)
        for boxID, (a_box_dir, a_box, a_mol_set) in enumerate(zip(dum_nested_dirs, dum_boxes, dum_mols)):
            for molID, (a_mol_dir, a_mol) in enumerate(zip(a_box_dir, a_mol_set)):
                for repID, (a_rep_dir, a_rep) in enumerate(zip(a_mol_dir, repList)):
                    dum_file_name = a_rep_dir + 'param.key'
                    dum_struc_name = dum_struc_dir_prefix + "_".join([str(a_num) for a_num in a_mol]) + ".prm"
                    if not fullRand:
                        IOUtils.write_param_file(param_obj=self.KeyFileTemplate,
                                                 file_path=dum_file_name,
                                                 box_size=a_box,
                                                 run_name='_',
                                                 energy_file=dum_en_file,
                                                 struc_file=dum_struc_name,
                                                 rng_seed=a_rep,
                                                 clus_mode=cls_mode)
                    else:
                        IOUtils.write_param_file(param_obj=self.KeyFileTemplate,
                                                 file_path=dum_file_name,
                                                 box_size=a_box,
                                                 run_name='_',
                                                 energy_file=dum_en_file,
                                                 struc_file=dum_struc_name,
                                                 rng_seed=0,
                                                 clus_mode=cls_mode)
        
        return None
    
    def clean_dirs_for(self, sys_name: str="SysA", b_txt: bool=True, b_trajectory=True, b_data=True, wInt=True):
        """
        Loops over every directory and deletes the files using 'rm *.txt *.lammpstrj *.dat"
        :param sys_name:
        :param b_txt: {Boolean} Delete .txt files or not.
        :param b_trajectory: {Boolean} Delete .lammpstrj files or not.
        :param b_data: {Boolean} Delete .dat files or not.
        :param wInt: {Boolean} wInt or noInt folders to loop over.
        :return: None
        """
        
        assert b_trajectory or b_txt or b_data, "At least one of the types of files have to be deleted."
        
        base_comm = ['rm']
        if b_txt:
            base_comm.append('*.txt')
        if b_data:
            base_comm.append('*.dat')
        if b_trajectory:
            base_comm.append('*lammpstrj')
        
        full_comm = ' '.join(base_comm)
        
        dum_dir_list = self._get_dirs_list_for(sys_name=sys_name, wInt=wInt)
        IOUtils.loop_over_dir_list(list_of_dirs=dum_dir_list,
                                   _passed_func=sproc.run,
                                   args=full_comm, shell=True, capture_output=True, text=True)
        
        
        return None

    def _read_raw_data_files_for(self, sys_name: str = "SysA", wInt: bool = True, file_name='__CLUS.dat',
                                 repList: list = None):
        """
        For this system, we generate the full nested_directory list, and then read the file_name.
        Remember that to generate the nested_list of directories, we do NOT use SystemSetup.get_independent_conditions()
        
        :param sys_name:
        :param wInt:
        :param file_name:
        :param repList
        :return:
        """
        
        if repList is None:
            repList = np.arange(1,self.Num_Reps+1)
        
        dum_nest_list = self._get_nested_dirs_for(sys_name=sys_name, wInt=wInt, repList=repList)
        
        dum_ret_vals = IOUtils.loop_over_nested_dirs(nested_list=dum_nest_list,
                                                     _passed_func=IOUtils.read_nploadtxt,
                                                     file_name=file_name)
        
        return dum_ret_vals
    
    def _save_raw_npy_data_for(self, sys_name: str = "SysA", wInt:bool = True, file_name: str ='__CLUS.dat', repList: list = None):
        """
        Convenient wrapper to use np.save the particular data-file for this system.
        
        :param sys_name:
        :param wInt:
        :param file_name:
        :param repList:
        :return:
        """
        
        if repList is None:
            repList = np.arange(1,self.Num_Reps+1)
        
        dum_data = np.array(self._read_raw_data_files_for(sys_name, wInt, file_name, repList=repList))
        dum_file_name = self._gen_data_dir_prefix_for(sys_name) + file_name
        np.save(dum_file_name, dum_data)
    
    def _save_raw_pickle_data_for(self, sys_name, wInt=True, file_name='__COMDen.dat', repList: list = None):
        """
       Convenient wrapper to use pickle.dump the particular data-file for this system.
       This is for the types of data that cannot be converted to a numpy-array like the RDFs
       
       :param sys_name:
       :param wInt:
       :param file_name:
       :param repList:
       :return:
       """
        
        if repList is None:
            repList = np.arange(1,self.Num_Reps+1)
        
        dum_data = self._read_raw_data_files_for(sys_name, wInt, file_name, repList=repList)
        dum_file_name = self._gen_data_dir_prefix_for(sys_name) + file_name + '.b'
        
        with open(dum_file_name, 'wb') as dFile:
            pickle.dump(dum_data, dFile)
    
    def save_CorrDen_for(self, sys_name, path_to_norm_data='', norm_naming_func=None, COMDen_file_name='__COMDen.dat'):
        """
        Reads COMDen.dat for this system, and generate the CorrDen.dat and saves it to the data directory.
        We save the [xAr_list, corr_den] using pickle
        
        :param sys_name:
        :return:
        """
        
        this_sys = self.SysInfo[sys_name]
        
        dum_comden_name = self._gen_data_dir_prefix_for(sys_name) + COMDen_file_name + '.b'
        
        dum_comden_data = []
        with open(dum_comden_name, 'rb') as dFile:
            dum_comden_data = pickle.load(dFile)
        
        dum_comden_analysis = COMDenUtils(total_comden_data=dum_comden_data,
                                          temp_nums=self.Num_Temps,
                                          comp_nums=this_sys.CompNum,
                                          box_list=this_sys.Boxes)
        
        dum_comden_analysis.reshape_raw_data(wMode=1)
        dum_comden_analysis.gen_corr_den(path_to_norm_data=path_to_norm_data,
                                         naming_func=norm_naming_func)
        
        dum_corrden_name = self._gen_data_dir_prefix_for(sys_name) + '__CorrDen.dat.b'
        with open(dum_corrden_name, 'wb') as dFile:
            pickle.dump([dum_comden_analysis.xArr_list[:], dum_comden_analysis.corr_data[:]], dFile)
        
        return None
    
    def save_CorrDen_MolTypeCOM_for(self, sys_name, path_to_norm_data='', norm_naming_func=None,
                                    COMDen_file_name='__COMDen.dat'):
        """
        Reads COMDen.dat for this system, and generate the CorrDen.dat and saves it to the data directory.
        We save the [xAr_list, corr_den] using pickle
        
        :param sys_name:
        :return:
        """
        
        this_sys = self.SysInfo[sys_name]
        
        dum_comden_name = self._gen_data_dir_prefix_for(sys_name) + COMDen_file_name + '.b'
        
        dum_comden_data = []
        with open(dum_comden_name, 'rb') as dFile:
            dum_comden_data = pickle.load(dFile)
        
        dum_comden_analysis = COMDenUtils(total_comden_data=dum_comden_data,
                                          temp_nums=self.Num_Temps,
                                          comp_nums=this_sys.CompNum,
                                          box_list=this_sys.Boxes)
        
        dum_comden_analysis.reshape_raw_data(wMode=0)
        dum_comden_analysis.gen_corr_den(path_to_norm_data=path_to_norm_data,
                                         naming_func=norm_naming_func)
        
        dum_corrden_name = self._gen_data_dir_prefix_for(sys_name) + '__CorrDen_MolType.dat.b'
        with open(dum_corrden_name, 'wb') as dFile:
            pickle.dump([dum_comden_analysis.xArr_list[:], dum_comden_analysis.corr_data[:]], dFile)
        
        return None


class COMDenUtils(object):
    """
    Collection of funcitons to analyze and manipulate COMDen data files. It is assumed that the data are generated by
    SimulationSetup._save_raw_pickle_data_for() for the COMDen data.
    Therefore, the indexing structure of COMDen_data should be COMDen_Data[boxID][numID][repID][compID & tempIDs mixed].
    Since the last index is a convolution, the first thing we do is reshape the data into a more tractable form.

    reshaped_data[boxID][tempID][numID][compID][repID][:] so then each box-size can be stored as a numpy array
    """
    
    def __init__(self, total_comden_data, temp_nums=3, comp_nums=3, box_list=np.array([100, 110, 120])):
        """
        Given the total_comden_data, we initialize this object by storing the relavent dimensions.
        
        :param total_comden_data:
        :param temp_nums: The total number of different temperatures probed in this data.
        :param comp_nums: The total number of different components in the system.
        """
        self.raw_data = total_comden_data[:]  # Store a copy to be careful
        self.ReshapeQ = False
        self.CorrDenQ = False
        self.num_boxes = len(self.raw_data)
        self.num_mols = len(self.raw_data[0])
        self.num_reps = len(self.raw_data[0][0])
        self.num_temps = temp_nums
        self.num_comps = comp_nums
        # Total number of data lines per temperature per mode
        self.tot_comden_comps = self.num_comps * (self.num_comps + 1)
        self.Boxes = box_list[:]
        self.corr_data = []
        self.xArr_list = []
    
    def reshape_raw_data(self, wMode=1):
        """
        
        :param wMode = 0 for COM from MolType largest cluster, 1 for all molecules of MolType
        This reshapes the data to have the following structure
        [boxID, tempID, numID, compID, repID, r_i]
        :return:
        """
        dum_comps_list = self.gen_pairs_list(totMols=self.num_comps)
        per_box_data = []
        for boxID, com_den_of_box in enumerate(self.raw_data[:]):
            dum_box_data = np.array(com_den_of_box)
            per_temp_data = []
            for tempID in range(self.num_temps):
                per_num_data = []
                for numID in range(self.num_mols):
                    per_comp_data = []
                    for compID, (compA, compB) in enumerate(dum_comps_list):
                        per_rep_data = []
                        for repID in range(self.num_reps):
                            this_idx = self.index_comp_to_comp_wMode_of_temp(compA=compA, compB=compB, temp_id=tempID,
                                                                             mode=wMode)
                            # print(this_idx)
                            per_rep_data.append(dum_box_data[numID, repID, this_idx])
                        per_comp_data.append(per_rep_data)
                    per_num_data.append(per_comp_data)
                per_temp_data.append(per_num_data)
            per_box_data.append(np.array(per_temp_data))
        self.raw_data = per_box_data[:]
        self.ReshapeQ = True
        
        return None
    
    def gen_pairs_list(self, totMols=None):
        """
        Generates the comp-to-comp list given this many total_molecules.
        The list is like [[-1,0], [-1,1], ...,
                          [0,0],  [0,1], ...,
                          ]
        Every possible pair, and in order. With the addition of the -1,n pairs which correspond to the system's COM
        
        :param totMols:
        :return:
        """
        if totMols is not None:
            totMols = self.num_comps
        return COMDenUtils._gen_pairs_list(totMols)
    
    def index_comp_to_comp(self, compA=0, compB=0, tot_mols=None):
        """
        From the unshaped raw data, we return the index that corresponds the the density profile of compB with respect
        to the COM of compA
        
        :param compA:
        :param compB:
        :param tot_mols:
        :return:
        """
        
        if tot_mols is None:
            tot_mols = self.num_comps
        
        if compA < 0:
            return compB
        else:
            return tot_mols + compB + tot_mols * compA
    
    def index_comp_to_comp_wMode(self, compA=0, compB=0, tot_comden_comps=None, mode=0):
        """
        The raw index adjusted for the internal shift in LaSSI where if mode=0, we only look at the beads in the largest
        cluster for compA, and with mode=1 we look at all beads.
        
        :param compA:
        :param compB:
        :param tot_comden_comps:
        :param mode:
        :return:
        """
        assert 1 >= mode >= 0, "Mode can only be 0 or 1"
        
        if tot_comden_comps is None:
            tot_comden_comps = self.tot_comden_comps
        
        return tot_comden_comps * mode + self.index_comp_to_comp(compA, compB)
    
    def index_comp_to_comp_wMode_of_temp(self, compA=0, compB=0, tot_comden_comps=None, mode=0, temp_id=0):
        """
        Convenient wrapper that includes the shift due to tempID
        
        :param compA:
        :param compB:
        :param tot_comden_comps:
        :param mode:
        :param temp_id
        :return:
        """
        if tot_comden_comps is None:
            tot_comden_comps = self.tot_comden_comps
        return temp_id * 2 * tot_comden_comps + self.index_comp_to_comp_wMode(compA, compB, tot_comden_comps, mode)
    
    def gen_corr_den(self, path_to_norm_data, naming_func=None):
        """
        Given the total path to the normalization data, we loop over every box and normalize the number distributions
        to generate the density distributions. Particularly, we have N(r_i) / N_0(r_i) and we only perform the divisions
        where N_0(r_i) != 0.
        One can provide the naming function for the normalization files where the naming_func only takes box_size
        as the argument.
        
        :param path_to_norm_data:
        :return:
        """
        assert self.ReshapeQ, "The raw data needs to be reshaped first. Use the reshape_raw_data() method!"
        
        self.xArr_list = []
        
        per_box_data = []
        for boxID, a_box in enumerate(self.Boxes):
            dum_norm_data = self._get_norm_data(path_to_norm_data=path_to_norm_data,
                                                box_size=a_box,
                                                naming_func=naming_func)
            good_pts = np.where(dum_norm_data != 0)
            norm_den_cor = dum_norm_data[good_pts]
            dum_xAr = np.arange(0, a_box, 0.25)
            xAr_cor = dum_xAr[good_pts]
            self.xArr_list.append(xAr_cor)
            
            per_temp_data = []
            for tempID in range(self.num_temps):
                per_num_data = []
                for numID in range(self.num_mols):
                    per_comp_data = []
                    for compID in range(self.tot_comden_comps):
                        per_rep_data = []
                        for repID in range(self.num_reps):
                            dum_yAr = self.raw_data[boxID][tempID, numID, compID, repID]
                            per_rep_data.append(dum_yAr[good_pts] / norm_den_cor)
                        per_comp_data.append(per_rep_data)
                    per_num_data.append(per_comp_data)
                per_temp_data.append(per_num_data)
            per_box_data.append(np.array(per_temp_data))
        
        self.corr_data = per_box_data[:]
        self.CorrDenQ = True
        
        return None
    
    def _get_norm_data(self, path_to_norm_data, box_size, naming_func=None):
        """
        Fetch the normalization data given the box-size.
        
        :param path_to_norm_data:
        :param box_size:
        :param naming_func:
        :return:
        """
        assert box_size > 1, "Box-size should be bigger than 1!"
        
        if naming_func is None:
            naming_func = self._gen_norm_filename
        
        dum_file_name = path_to_norm_data + naming_func(box_size)
        
        return np.loadtxt(dum_file_name)
    
    @staticmethod
    def _gen_norm_filename(box_size):
        """
        Given the box-size, we generate the file-name for the normalization. This internal function assumes the file-names
        are P0_S_{box_size}.dat
        
        :param box_size:
        :return:
        """
        return f"P0_S_{box_size}.dat"
    
    @staticmethod
    def _gen_pairs_list(totMols:int = 1):
        """
        Generates the comp-to-comp list given this many total_molecules.
        The list is like [[-1,0], [-1,1], ...,
                          [0,0],  [0,1], ...,
                          ]
        Every possible pair, and in order. With the addition of the -1,n pairs which correspond to the system's COM
        
        :param totMols:
        :return:
        """
        assert totMols > 0, "Positive integers only."
        
        dum_pairs_list = []
        for i in range(totMols):
            dum_pairs_list.append([-1, i])
        
        for i in range(totMols):
            for j in range(totMols):
                dum_pairs_list.append([i, j])
        return np.array(dum_pairs_list)


class COMDenAnalysis(object):
    """
    Collection of functions to help with the analysis of COMDen data to calculate the densities of coexisting phases.
    It is assumed that initializer is given the [xAr_list, corr_data] generated from _COMDen_Utils() either directly from
    the class, or read from a previously saved file.
    """
    
    def __init__(self, tot_corr_den_obj, num_of_comps: int = 3):
        """
        Just initializing the lists required. Just remember thqat xArList contains xAr's for each box-size.
        Similarly, _CorrData contains the corrected_den_data for each box-size. That is why both of those lists
        can not be turned into true NumPy arrays. Each box-size is it's own NumPy array.
        
        :param tot_corr_den_obj: Output of _COMDen_Utils.gen_corr_den()
        :param num_of_comps: The number of unique molecule types.
        """
        self.xArList = tot_corr_den_obj[0][:]
        self._CorrData = tot_corr_den_obj[1][:]
        self.Num_Comps = num_of_comps
        self.Num_Boxes = len(self.xArList)
        self.PairsList = COMDenUtils._gen_pairs_list(self.Num_Comps)
        self.RepAvg = self.per_rep_avg(self._CorrData)
        self.RepErr = self.per_rep_err(self._CorrData)
        # del self._CorrData
    
    @staticmethod
    def per_rep_avg(TotData):
        """
        For each box-size, we generate the per-replicate average of the total arrays.
        This corresponds to a axis=-2 averaging.
        
        :return:
        """
        
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.mean(a_box, axis=-2))
        
        return dum_arr
    
    @staticmethod
    def per_rep_err(TotData):
        """
        For each box-size, we generate the per-replicate std of the total arrays.
        This corresponds to a axis=-2 std'ing.
        
        :param TotData
        :return:
        """
        
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.std(a_box, axis=-2, ddof=1))
        
        return dum_arr
    
    @staticmethod
    def calc_hi_and_lo_avgs(data_ave: np.ndarray, data_err: np.ndarray,
                            radHi_start: int = 0,
                            radHi_cut: int = 8,
                            radLo_start: int = -25,
                            radLo_cut: int = -10):
        """
        Given data with error, we calculate the average of the data over the given shell
        
        :param data_ave: Data corresponding to the means per rep.
        :param data_err: Data corresponding to the errors.
        :param radHi_start: Radius we start averaging over. This is supposed to be the high concentration.
        :param radHi_cut: Radius uptil which we average over. This is supposed to be the high concentration.
        :param radLo_start: Radius from where we start averaging. This is supposed to be the low concentration.
        :param radLo_cut: Radius from where we end averaging This is supposed to be the low concentration.
        :return: [denHi, denHi_err, denLo, denLo_err]
        """
        
        denHi, denHi_err = RadialFuncUtils.calc_avg_over_shell(data_ave, data_err, radHi_start, radHi_cut)
        denLo, denLo_err = RadialFuncUtils.calc_avg_over_shell(data_ave, data_err, radLo_start, radLo_cut)
        
        return np.array([denHi, denHi_err, denLo, denLo_err])
    
    @staticmethod
    def calc_coexistence_assuming_uniform_drops(tot_ave_data, tot_err_data,
                                                radHi_start: int = 0,
                                                radHi_cut: int = 4,
                                                radLo_start: int = -25,
                                                radLo_cut: int = -5):
        """
        We calculate the coexisting concentrations by simpling averaging over the density profile near r=0, or the
        COM, and far away, or outside in the dilute region. Note that this is _only_ correct for nice systems that
        produce non-layered droplets, and only only droplet. We use both the values, and their corresponding errors.
        It is assumed that the AveData and ErrData are the COMDenAnalysis.RegAvg and COMDenAnalysis.RepErr, so AveData
        is used to determine the shape of the output.
        
        :param tot_ave_data: Data corresponding to the means per rep.
        :param tot_err_data: Data corresponding to the errors.
        :param radHi_start: Radius we start averaging over. This is supposed to be the high concentration.
        :param radHi_cut: Radius up to which we average over. This is supposed to be the high concentration.
        :param radLo_start: Radius from where we start averaging. This is supposed to be the low concentration.
        :param radLo_cut: Radius from where we end averaging This is supposed to be the low concentration.
        :return: [denHi, denHi_err, denLo, denLo_err] where each of the den* correspond to the same shape
        as [boxNums, tempNums, molNums, compNums].
        """
        assert len(tot_ave_data) == len(tot_err_data) and tot_ave_data[0].shape == tot_err_data[0].shape, "AveData & ErrData need to have the same shapes!"
        
        box_nums = len(tot_ave_data)
        temp_nums, mol_nums, comp_nums = tot_ave_data[0].shape[:-1]
        
        denHi = np.zeros((box_nums, temp_nums, mol_nums, comp_nums))
        denHi_err = np.zeros_like(denHi)
        denLo = np.zeros_like(denHi)
        denLo_err = np.zeros_like(denHi)
        
        for boxID, (box_avg, box_err) in enumerate(zip(tot_ave_data, tot_err_data)):
            for tempID, (temp_avg, temp_err) in enumerate(zip(box_avg, box_err)):
                for numID, (num_avg, num_err) in enumerate(zip(temp_avg, temp_err)):
                    for compID, (comp_avg, comp_err) in enumerate(zip(num_avg, num_err)):
                        
                        c_hi, c_hi_err, c_lo, c_lo_err = COMDenAnalysis.calc_hi_and_lo_avgs(comp_avg, comp_err,
                                                                                            radHi_start, radHi_cut,
                                                                                            radLo_start, radLo_cut)
                        
                        denHi[boxID, tempID, numID, compID] = c_hi
                        denHi_err[boxID, tempID, numID, compID] = c_hi_err
                        
                        denLo[boxID, tempID, numID, compID] = c_lo
                        denLo_err[boxID, tempID, numID, compID] = c_lo_err
        
        return np.array([denHi, denHi_err, denLo, denLo_err])
    
    @staticmethod
    def per_rep_sum_of_components_avg(TotData, comp_list: list = None):
        """
        We compute the sum of the given components, and then take the per-rep average. This is for better error
        since taking the sum later will amplify the errors.
        Remember the organization of TotData. TotData[boxID][tempID, numID, compID, xID]
        
        :param TotData:
        :param comp_list: Indecies of the components to be summed over.
        :return: ProdData[boxNums][tempNums, molNums, xID]
        """
        if comp_list is None:
            comp_list = [0, 1]
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.mean(np.sum(a_box[:, :, comp_list], axis=-3), axis=-2))
        return dum_arr
    
    @staticmethod
    def per_rep_sum_of_components_err(TotData, comp_list: list = None):
        """
        We compute the sum of the given components, and then take the per-rep std. This is for better error
        since taking the sum later will amplify the errors.
        Remember the organization of TotData. TotData[boxID][tempID, numID, compID, xID]
        
        :param TotData:
        :param comp_list: Indecies of the components to be summed over.
        :return:
        """
        if comp_list is None:
            comp_list = [0, 1]
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.std(np.sum(a_box[:, :, comp_list], axis=-3), axis=-2, ddof=1))
        return dum_arr
    
    @staticmethod
    def per_rep_product_of_components_avg(TotData, comp_list: list = None):
        """
        We compute the sum of the given components, and then take the per-rep average. This is for better error
        since taking the sum later will amplify the errors.
        Remember the organization of TotData. TotData[boxID][tempID, numID, compID, xID]
        
        :param TotData:
        :param comp_list: Indicies of the components to be summed over.
        :return: ProdData[boxNums][tempNums, molNums, xID]
        """
        if comp_list is None:
            comp_list = [0, 1]
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.mean(np.prod(a_box[:, :, comp_list], axis=-3), axis=-2))
        return dum_arr
    
    @staticmethod
    def per_rep_product_of_components_err(TotData, comp_list: list = None):
        """
        We compute the sum of the given components, and then take the per-rep std. This is for better error
        since taking the sum later will amplify the errors.
        Remember the organization of TotData. TotData[boxID][tempID, numID, compID, xID]
        
        :param TotData:
        :param comp_list: Indecies of the components to be summed over.
        :return:
        """
        if comp_list is None:
            comp_list = [0, 1]
        dum_arr = []
        for boxID, a_box in enumerate(TotData):
            dum_arr.append(np.std(np.prod(a_box[:, :, comp_list], axis=-3), axis=-2, ddof=1))
        return dum_arr
    
    def per_rep_prod_of_components(self, comp_list: list = None):
        """
        Convenient way to get both the per_rep_product_of_components_avg and per_rep_product_of_components_err
        of the total data stored. This is inefficient since I take the products twice.
        
        :param comp_list: List of components we wish to take the product of.
        :return: per_rep_product_of_components_avg, per_rep_product_of_components_err
        """
        dum_avg = self.per_rep_product_of_components_avg(self._CorrData, comp_list)
        dum_err = self.per_rep_product_of_components_err(self._CorrData, comp_list)
        return dum_avg, dum_err

    def per_rep_sum_of_components(self, comp_list: list = None):
        """
        Convenient way to get both the per_rep_sum_of_components_avg and per_rep_sum_of_components_err
        of the total data stored. This is inefficient since I take the sums twice.
        
        :param comp_list: List of components we wish to take the product of.
        :return: per_rep_sum_of_components_avg, per_rep_sum_of_components_err
        """
        dum_avg = self.per_rep_sum_of_components_avg(self._CorrData, comp_list)
        dum_err = self.per_rep_sum_of_components_err(self._CorrData, comp_list)
        return dum_avg, dum_err
    
    def get_data_for(self, boxID: int = 0, numID: int = 0, tempID: int = 0, compID: int = 0):
        """
        Conveniece function to return a _DataWithError object for the given set of conditions.
        :param boxID:
        :param numID:
        :param tempID:
        :param compID:
        :return:
        """
        dum_x = self.xArList[boxID]
        dum_y = self.RepAvg[boxID][tempID, numID, compID]
        dum_e = self.RepErr[boxID][tempID, numID, compID]
        return _DataWithError(x_vals=dum_x, y_vals=dum_y, e_vals=dum_e, debug_mode=False)
    
    def get_all_data_for_comp_with_COM(self, compID:int =0, COM:int = -1):
        """
        Returns a list with all the data for this component, with the given COM.
        If COM = -1, that means that the system COM is the chosen COM.
        :param compID:
        :param COM:
        :return:
        """
        assert (compID < self.Num_Comps) and (
                    compID >= 0), "compID has to be positive, and smaller than the number of components in the system!"
        assert (COM == -1) or (COM < self.Num_Comps), "COM Has to be -1 for system COM, or compID for that component!"
        
        _compID = self.Num_Comps * (COM + 1) + compID
        
        full_data = []
        
        for boxID, (xAr, yAr_tot, eAr_tot) in enumerate(zip(self.xArList, self.RepAvg, self.RepErr)):
            per_temp = []
            for tempID, (yAr_temp, eAr_temp) in enumerate(zip(yAr_tot, eAr_tot)):
                per_num = []
                for numID, (yAr_num, eAr_num) in enumerate(zip(yAr_temp, eAr_temp)):
                    yAr = yAr_num[_compID]
                    eAr = eAr_num[_compID]
                    
                    per_num.append(_DataWithError(xAr, yAr, eAr))
                per_temp.append(per_num)
            full_data.append(per_temp)
        
        return full_data


class RadialFuncUtils(object):
    """
    Collection of functions to manipulate data that is sampled in radial distance like density distributions and
    pair-correlations.
    """
    
    @staticmethod
    def calc_avg_over_shell(data_avg, data_err,
                            rad_start: int = 0,
                            rad_cut: int = 8):
        """
        Given data with error, we calculate the average of the data over the given shell.
        
        :param data_avg: Data corresponding to the means per rep.
        :param data_err: Data corresponding to the errors.
        :param rad_start: Radius we start averaging over. This is supposed to be the high concentration.
        :param rad_cut: Radius uptil which we average over. This is supposed to be the high concentration.
        :return: [avg, err]
        """
        
        dum_avg = np.mean(data_avg[rad_start:rad_cut])
        dum_err = np.sqrt(np.mean(data_err[rad_start:rad_cut]) ** 2.)
        
        return dum_avg, dum_err
    
    @staticmethod
    def smooth_median_filter_sliding(y: np.ndarray, w: int = 4):
        """
        Using a sliding window of width w, we pick the median value of y.
        The resulting array will have shape that is w-1 less than y.
        """
        if w < 3:
            print("A median doesn't _really_ make sense with less than 3 elements")
        _dum_y = np.zeros(y.shape[0]-w+1)
        for wID in range(_dum_y.shape[0]):
            _dum_y[wID] = np.median(y[wID: wID + w])
        return _dum_y
    
    @staticmethod
    def smooth_mean_filter_sliding(y: np.ndarray, w: int = 4):
        """
        Using a sliding window of width w, we pick the mean value of y.
        The resulting array will have shape that is w-1 less than y.
        """
        _dum_y = np.zeros(y.shape[0]-w+1)
        for wID in range(_dum_y.shape[0]):
            _dum_y[wID] = np.mean(y[wID: wID + w])
        return _dum_y
    
    @staticmethod
    def smooth_max_filter_sliding(y: np.ndarray, w: int = 4):
        """
        Using a sliding window of width w, we pick the max value of y.
        The resulting array will have shape that is w-1 less than y.
        """
        _dum_y = np.zeros(y.shape[0]-w+1)
        for wID in range(_dum_y.shape[0]):
            _dum_y[wID] = np.max(y[wID: wID + w])
        return _dum_y
    
    @staticmethod
    def smooth_min_filter_sliding(y: np.ndarray, w: int = 4):
        """
        Using a sliding window of width w, we pick the min value of y.
        The resulting array will have shape that is w-1 less than y.
        """
        _dum_y = np.zeros(y.shape[0]-w+1)
        for wID in range(_dum_y.shape[0]):
            _dum_y[wID] = np.min(y[wID: wID + w])
        return _dum_y
    
    @staticmethod
    def smooth_general_filter_sliding(y: np.ndarray, fil_func=np.median, w: int = 4):
        """
        Using a sliding window of width w, we apply the provided function.
        The resulting array will have shape that is w-1 less than y.
        """
        _dum_y = np.zeros(y.shape[0]-w+1)
        for wID in range(_dum_y.shape[0]):
            _dum_y[wID] = fil_func(y[wID: wID + w])
        return _dum_y


class _DataWithError(object):
    """
    Since I am tired of implementing error propoagation for data that are like $(x, f(x), \delta f(x))$, I
    have decided to write a general-ish object that can be used to manipulate such data.
    In particular, simple operations like adding, subtracting, multiplying, dividing, and taking the logarithm will
    be implemented.

    - For addition and subtraction, the error is quadrature summed: $\sqrt{ \delta a^2 + \delta b^2 + ...}$
    - For multiplication, if $g = a \times b$, we have $ g \times \sqrt{ (\delta a / a)^2 + (\delta b / b)^2 }$.
    - For division, if $ g = a / b$, then we have to make sure that we only calculate the ratio where $b \neq 0$.
      The error is the same as multiplication. Furthermore, we filter out all the non-calculated values from x as well.
    - For logarithms, again we make sure to only calculate for non-zero quantities, and filter. If $ g = \log(a)$, then
      $\delta g = \delta a / a$, as the relative error.
    """
    
    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray, e_vals: np.ndarray, debug_mode: bool = False):
        """
        Initializer must get x-coordinates, y-values and their corresponding error-values.
        """
        if debug_mode:
            assert len(x_vals) == len(y_vals) == len(e_vals), "All arrays must be the same length!"
        self._x = x_vals.copy()
        self._y = y_vals.copy()
        self._e = e_vals.copy()
    
    def get_data(self):
        """
        Returns the stored values as a tuple.
        :return: tuple containing (x, y, z)
        """
        return (self.x.copy(), self.y.copy(), self.e.copy())

    @property
    def x(self):
        return self._x.copy()
    
    @x.setter
    def x(self, this_val: np.ndarray):
        self._x = this_val.copy()
    
    @x.deleter
    def x(self):
        del self._x
    
    @property
    def y(self):
        return self._y.copy()
    
    @y.setter
    def y(self, this_val: np.ndarray):
        self._y = this_val.copy()
    
    @y.deleter
    def y(self):
        del self._y
    
    @property
    def e(self):
        return self._e.copy()
    
    @e.setter
    def e(self, this_val: np.ndarray):
        self._e = this_val.copy()
    
    @e.deleter
    def e(self):
        del self._e
    
    def __eq__(self, other):
        """
        
        :param other:
        :return:
        """
        return np.array_equal(self.x, other.x) and np.array_equal(self.y, other.y) and np.array_equal(self.e, other.e)
        
    def __len__(self):
        return len(self.x)
    
    def __add__(self, other):
        """
        Addition of two data with error. The y-values are added, while the errors are added
        in quadrature.
        """
        # assert isinstance(other, _DataWithError), "Both objects need to be DataWithError objects"
        # assert len(self) == len(other), "Both objects have to be the same length."
        
        new_vals = self.y + other.y
        new_err = np.sqrt(self.e ** 2. + other.e ** 2.)
        
        return _DataWithError(self.x, new_vals, new_err)
    
    def __sub__(self, other):
        """
        Subtraction of two data with error. The y-values are subtracted, while the errors are added
        in quadrature.
        """
        # assert isinstance(other, _DataWithError), "Both objects need to be DataWithError objects"
        # assert len(self) == len(other), "Both objects have to be the same length."
        
        new_vals = self.y - other.y
        new_err = np.sqrt(self.e ** 2. + other.e ** 2.)
        
        return _DataWithError(self.x, new_vals, new_err)
    
    def __mul__(self, other):
        """
        Multiplication of two data with error. Firstly, we find indecies where both are non-zero.
        The y-values are multiplied, while the relative errors are added in quadrature. Only the non-zero elements
        shared between both are used.
        """
        # assert isinstance(other, _DataWithError), "Both objects need to be DataWithError objects"
        # assert len(self) == len(other), "Both objects have to be the same length."

        gd_pts_self = self.y  != 0
        gd_pts_othr = other.y != 0

        gd_pts = gd_pts_othr * gd_pts_self

        new_vals = self.y[gd_pts] * other.y[gd_pts]
        new_err = new_vals * np.sqrt(
            (self.e[gd_pts] / self.y[gd_pts]) ** 2. + (other.e[gd_pts] / other.y[gd_pts]) ** 2.)
        
        return _DataWithError(self.x[gd_pts], new_vals, new_err)
    
    def __truediv__(self, other):
        """
        Division of two data with error. Firstly, we find indecies where other.y is non-zero.
        We then also find where self.y is non-zero. The division is only performed for indecies where
        both are non-zero.
        The y-values are multiplied, while the relative errors are added
        in quadrature.
        """
        # assert isinstance(other, _DataWithError), "Both objects need to be of DataWithError type"
        # assert len(self) == len(other), "Both objects have to be the same length."
        
        gd_pts_self = self.y != 0
        gd_pts_othr = other.y !=0
        
        gd_pts = gd_pts_othr * gd_pts_self
        
        new_vals = self.y[gd_pts] / other.y[gd_pts]
        new_err = new_vals * np.sqrt(
                (self.e[gd_pts] / self.y[gd_pts]) ** 2. + (other.e[gd_pts] / other.y[gd_pts]) ** 2.)
        
        return _DataWithError(self.x[gd_pts], new_vals, new_err)

    def __div__(self, other):
        """
        Division of two data with error. Firstly, we find indecies where other.y is non-zero.
        We then also find where self.y is non-zero. The division is only performed for indecies where
        both are non-zero.
        The y-values are multiplied, while the relative errors are added
        in quadrature.
        """
        # assert isinstance(other, _DataWithError), "Both objects need to be of DataWithError type"
        # assert len(self) == len(other), "Both objects have to be the same length."
    
        gd_pts_self = self.y != 0
        gd_pts_othr = other.y != 0
    
        gd_pts = gd_pts_othr * gd_pts_self
    
        new_vals = self.y[gd_pts] / other.y[gd_pts]
        new_err = new_vals * np.sqrt(
                (self.e[gd_pts] / self.y[gd_pts]) ** 2. + (other.e[gd_pts] / other.y[gd_pts]) ** 2.)
    
        return _DataWithError(self.x[gd_pts], new_vals, new_err)
    
    def __pow__(self, power: float = 2.0):
        """
        Raise the data to the power. For error, if $f = a^n$, then $\delta f = n * ( a ^ {n-1}) \delta a$
        """
        if power == 1:
            return _DataWithError(self.x, self.y, self.e)
        elif power > 1:
            new_x = self.x
            new_y = self.y ** power
            new_e = power * (self.y ** (power - 1)) * self.e
        else:
            gd_pts = (self.y != 0)
            
            new_x = self.x[gd_pts]
            new_y = self.y[gd_pts] ** power
            new_e = power * (self.y[gd_pts] ** (power - 1.)) * self.e[gd_pts]
        
        return _DataWithError(new_x, new_y, np.abs(new_e))
    
    def __abs__(self):
        """
        Return the absolute of the y-values. The error is left unchanged.
        """
        return _DataWithError(self.x, np.abs(self.y), self.e)
    
    def __floordiv__(self, other):
        """
        Cheeky FloorDivision of two data with error. Since __div__ and __truediv__ have the possibility of
        returning a smaller version of the arrays, we want one division implementation where we do not do that.
        Instead, we only divide where both y-values are non-zero, but set the rest to np.nan.
        
        Firstly, we find indecies where other.y is non-zero.
        We then also find where self.y is non-zero. The division is only performed for indecies where
        both are non-zero.
        The y-values are multiplied, while the relative errors are added
        in quadrature.
        """

        gd_pts_self = self.y != 0
        gd_pts_othr = other.y != 0

        gd_pts = gd_pts_othr * gd_pts_self
        
        new_vals = np.zeros_like(self.y)*np.nan
        new_err  = np.zeros_like(self.y)*np.nan
        
        new_vals[gd_pts] = self.y[gd_pts] / other.y[gd_pts]
        new_err[gd_pts] = new_vals[gd_pts] * np.sqrt(
                (self.e[gd_pts] / self.y[gd_pts]) ** 2. + (other.e[gd_pts] / other.y[gd_pts]) ** 2.)

        return _DataWithError(self.x, new_vals, new_err)
    
    def _non_negative(self):
        """
        Look for indecies where self.y is non-zero. Then we return a new object that has only the non-zero values.
        """
        gd_pts = self.y != 0
        
        return _DataWithError(self.x[gd_pts], self.y[gd_pts], self.e[gd_pts])

    def _non_zero_error(self):
        """
        Look for indecies where self.e is non-zero. Then we return a new object that has only the non-zero values.
        """
        gd_pts = self.e != 0
    
        return _DataWithError(self.x[gd_pts], self.y[gd_pts], self.e[gd_pts])
    
    def _log10(self):
        """
        Calculate the log10 of the data and propagate the error through, which is relative error.
        x -> x
        y -> log(y)
        e -> e/y
        """
        
        pos_data = self._non_negative()
        
        new_y = np.log10(pos_data.y)
        new_e = pos_data.e / pos_data.y
        
        return _DataWithError(pos_data.x, new_y, new_e)
    
    def calc_sum(self):
        """
        Return the sum of the data, and the uncertainty. The uncertainty is the quadrature sum of the error.
            SUM, SQRT(SUM(ERR**2))
        :return: sum_value, sum_err
        """

    
        dum_sum = np.sum(self.y)
    
        dum_err = np.sqrt(np.sum((self.e) ** 2.))
    
        return dum_sum, dum_err
    
    def calc_distribution_moment_k(self, k: int = 1, normalized: bool = False):
        """
        Treats the data as a distribution, and calculates the moment. Let that moment be k.
        To calculate the value
            - F = x ** k
            - A = ( F * e )
            - B = SUM(A)
            - Mean = B / SUM(y)
        To calculate the error, we add all the errors in quadrature
            - F = x ** k
            - A = ( F * e )
            - B = A**2
            - C = SUM(B)
            - D = SQRT(D)
            - Err = D / SUM(y)
            
        :param k: Which moment to calculate
        :return: moment_val, moment_err
        """
        
        assert k > 0, "Moments should be positive!"

        if normalized:
            norm_fac = 1.
        else:
            norm_fac = 1./np.sum(self.y)
        
        dum_dom  = self.x ** k

        dum_mean = np.sum(dum_dom * self.y)

        dum_err  = np.sqrt(np.sum((dum_dom * self.e)**2.))

        return dum_mean*norm_fac, dum_err*norm_fac
        
    def calc_distribution_mean(self, normalized: bool = False):
        """
        Treats the data as a distribution, and calculates the mean-value as SUM(x * y) / SUM (y).
        To calculate the mean
            - A = ( x * e )
            - B = SUM(A)
            - Mean = B / SUM(y)
        To calculate the error, we add all the errors in quadrature
            - A = ( x * e )
            - B = A**2
            - C = SUM(B)
            - D = SQRT(D)
            - Err = D / SUM(y)
            
        :return: mean_val, mean_err
        """
        if normalized:
            norm_fac = 1.
        else:
            norm_fac = 1./np.sum(self.y)
        
        dum_mean = np.sum(self.x * self.y)
        
        dum_err  = np.sqrt(np.sum((self.x * self.e)**2.))
        
        return dum_mean*norm_fac, dum_err*norm_fac
    
    def calc_distribution_var(self, normalized: bool = False):
        """
        Treating the data as a distribution, we calculate the variance.
        Var = E(X^2) - E(X)^2
        The implementation calculates the first and second moments, and creates dummy _DataWithError() instances
        for easier propagation of error.
        :return: Var_val, Var_err
        """
        
        first_moment_m, first_moment_e = self.calc_distribution_moment_k(k=1, normalized=normalized)
        first_moment = _DataWithError(np.array([1]), np.array([first_moment_m]), np.array([first_moment_e]))

        second_moment_m, second_moment_e = self.calc_distribution_moment_k(k=2, normalized=normalized)
        second_moment = _DataWithError(np.array([1]), np.array([second_moment_m]), np.array([second_moment_e]))
        
        dum_var = second_moment - first_moment**2
        
        return dum_var.y, dum_var.e
        

class PlottingUtils(object):
    """
    A collection of functions that make plotting the data generated by the simulations more convenient.
    """
    
    @staticmethod
    def sampling_plot_ortho3D_sampled(mol_nums: np.ndarray,
                                      box_lens: np.ndarray,
                                      ax_obj=None,
                                      comp_list: list = None,
                                      init_comp: int = 0,
                                      plkwargs: dict = None):
        """
        Given the list of molecule numbers, and a list of box-sizes, we plot the starting concentrations we have
        sampled (presumably) for simulations. In particular, we plot the components given in [comp_list].
        
        :param ax_obj: Axes object to draw on.
        :param mol_nums: List of molecule numbers.
        :param box_lens: List of box-sizes.
        :param comp_list: index list corresponding to which components to plot.
        :param init_comp: The component that was allowed to vary when using set_ortho_boxes(..., comp_ig=init_comp)
        :return:
        """
        
        assert mol_nums.shape[0] == box_lens.shape[0], "Number of boxes and molecule-numbers is not the same."
        if ax_obj is None:
            ax_obj = plt.gca(projection='3d')
        if comp_list is None:
            comp_list = [0, 1, 2]
        assert len(comp_list) == 3, "Can only project 3 components in 3D."
        
        if plkwargs is None:
            plkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        
        conc_list = np.log10(np.array([a_comp / (box_lens ** 3.) for a_comp in mol_nums.T]))
        
        for pID in range(3):
            x, y, z = np.roll(conc_list, pID - init_comp, axis=0)[comp_list]
            ax_obj.plot(x, y, z, f"C{pID}o-", **plkwargs)
        
        return None
    
    @staticmethod
    def _embed3D_varying_planes(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                ax_obj=None,
                                proj_onto: int = 0,
                                xLo: float = -5., xHi: float = -1.,
                                plkwargs: dict = None,
                                plColor: str = 'C0'):
        """
        Given a full set of x, y, z values where (x_i, y_i, z_i) define coordinates in 3D-space, we project the data
        onto the two-planes. The two planes correspond to the planes where both components vary, while component-proj_
        onto is held constant.
        
        :param x: Values corresponding to the x-components.
        :param y: Values corresponding to the y-components.
        :param z: Values corresponding to the z-components.
        :param ax_obj: The matplotlib.Axes3D object to plot these on.
        :param proj_onto: Which plane to project onto. 0: yz, 1:xz, 2:xy.
        :param xLo: Shifting variable for yz and xy planes. _Should_ correspond to the 3D-plot lo-limit.
        :param xHi: Shifting variable for xz plane. _Should_ correspond to 3D-plot hi-limit.
        :param plkwargs: plt.plot() kwargs.
        :param plColor: Color for the plot.
        :return:
        """
        if plkwargs is None:
            plkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        if ax_obj is None:
            ax_obj = plt.gca(projection='3d')
        
        if proj_onto != 0:  # yz
            ax_obj.plot(y, z, "o-", zdir='x', zs=xLo, color=plColor, **plkwargs)
        if proj_onto != 1:  # xz
            ax_obj.plot(x, z, "o-", zdir='y', zs=xHi, color=plColor, **plkwargs)
        if proj_onto != 2:  # xy
            ax_obj.plot(x, y, "o-", zdir='z', zs=xLo, color=plColor, **plkwargs)
        
        return None

    @staticmethod
    def _embed3D_this_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            ax_obj=None,
                            proj_onto: str = 'xy',
                            xLo: float = -5., xHi: float = -1.,
                            plkwargs: dict = None,
                            plColor: str = 'C0'):
        """
        Given a full set of x, y, z values where (x_i, y_i, z_i) define coordinates in 3D-space, we project the data
        onto the desired plane.
        
        :param x: Values corresponding to the x-components.
        :param y: Values corresponding to the y-components.
        :param z: Values corresponding to the z-components.
        :param ax_obj: The matplotlib.Axes3D object to plot these on.
        :param proj_onto: Which plane to project onto. yz, xz, xy.
        :param xLo: Shifting variable for yz and xy planes. _Should_ correspond to the 3D-plot lo-limit.
        :param xHi: Shifting variable for xz plane. _Should_ correspond to 3D-plot hi-limit.
        :param plkwargs: plt.plot() kwargs.
        :param plColor: Color for the plot.
        :return:
        """
        assert proj_onto in ['yz', 'xz', 'xy'], "Invalid projection. Should be 'xy' | 'yz' | 'xz'"
        if plkwargs is None:
            plkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        if ax_obj is None:
            ax_obj = plt.gca(projection='3d')
    
        if proj_onto == 'yz':
            ax_obj.plot(y, z, "o-", zdir='x', zs=xLo, color=plColor, **plkwargs)
        if proj_onto == 'xz':
            ax_obj.plot(x, z, "o-", zdir='y', zs=xHi, color=plColor, **plkwargs)
        if proj_onto == 'xy':
            ax_obj.plot(x, y, "o-", zdir='z', zs=xLo, color=plColor, **plkwargs)
    
        return None
    
    
    
    @staticmethod
    def sampling_plot_ortho3D_embed_this_plane(mol_nums: np.ndarray,
                                               box_lens: np.ndarray,
                                               ax_obj=None,
                                               comp_list: list = None,
                                               init_comp: int = 0,
                                               proj_onto: str = 'yz',
                                               xLo: float = -5,
                                               xHi: float = -1,
                                               plkwargs: dict = None):
        """
        Given the list of molecule numbers, and a list of box-sizes, we project the sampled concentrations
        onto the given plane, and then shifted by xLo (or xHi depending on the projection). The plane is embedded in
        the 3D-plot.
        
        :param ax_obj: Axes object to draw on.
        :param mol_nums: List of molecule numbers.
        :param box_lens: List of box-sizes.
        :param comp_list: index list corresponding to which components to plot.
        :param init_comp: The component that was allowed to vary when using set_ortho_boxes(..., comp_ig=init_comp)
        :param proj_onto: Which axes to project onto. 'xy' | 'yz' | 'xz'
        :param xLo: Lower bound of xlim()
        :param xHi: Upper bound of xlim()
        :return:
        """
        
        assert mol_nums.shape[0] == box_lens.shape[0], "Number of boxes and molecule-numbers is not the same."
        if ax_obj is None:
            ax_obj = plt.gca(projection='3d')
        if comp_list is None:
            comp_list = [0, 1, 2]
        assert len(comp_list) == 3, "Can only project 3 components in 3D."
        conc_list = np.log10(np.array([a_comp / (box_lens ** 3.) for a_comp in mol_nums.T]))
        
        if plkwargs is None:
            plkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        
        assert proj_onto in ['yz', 'xz', 'xy'], "Invalid projection. Should be 'xy' | 'yz' | 'xz'"
        
        proj_dict = {'yz': 0, 'xz': 1, 'xy': 2}
        
        projID = proj_dict[proj_onto]
        x, y, z = np.roll(conc_list, projID - init_comp, axis=0)[comp_list]
        
        PlottingUtils._embed3D_this_plane(x=x, y=y, z=z,
                                          ax_obj=ax_obj, proj_onto=proj_onto,
                                          xLo=xLo, xHi=xHi,
                                          plkwargs=plkwargs, plColor=f'C{projID}')
        
        return None
    
    @staticmethod
    def sampling_plot_ortho3D_all(mol_nums: np.ndarray,
                                  box_lens: np.ndarray,
                                  ax_obj=None,
                                  comp_list: list = None,
                                  init_comp: int = 0,
                                  xLo: float = -5., xHi: float = -1.,
                                  samkwargs: dict = None,
                                  prjkwargs: dict = None,
                                  plot_sam: bool = True,
                                  plot_prj: bool = True):
        """
        Given the list of molecule numbers, and a list of box-sizes, we plot the starting concentrations we have
        sampled (presumably) for simulations, and all the projections.
        
        :param ax_obj: Axes object to draw on.
        :param mol_nums: List of molecule numbers.
        :param box_lens: List of box-sizes.
        :param comp_list: index list corresponding to which components to plot.
        :param init_comp: The component that was allowed to vary when using set_ortho_boxes(..., comp_ig=init_comp)
        :param xLo, xHi : Lower and upper bounds for the plots
        :param prjkwargs: Keyword args for the sampled plots.
        :param samkwargs: Keywords args for the projection plots.
        :return:
        """
        
        assert mol_nums.shape[0] == box_lens.shape[0], "Number of boxes and molecule-numbers is not the same."
        if ax_obj is None:
            ax_obj = plt.gca(projection='3d')
        if comp_list is None:
            comp_list = [0, 1, 2]
        assert len(comp_list) == 3, "Can only project 3 components in 3D."
        
        conc_list = np.log10(np.array([a_comp / (box_lens ** 3.) for a_comp in mol_nums.T]))
        
        if samkwargs is None:
            samkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        
        for pID in range(3):
            x, y, z = np.roll(conc_list, pID - init_comp, axis=0)[comp_list]
            if plot_sam:
                ax_obj.plot(x, y, z, f"C{pID}o-", **samkwargs)
            if plot_prj:
                PlottingUtils._embed3D_varying_planes(ax_obj=ax_obj, x=x, y=y, z=z,
                                                      proj_onto=pID, xLo=xLo, xHi=xHi,
                                                      plkwargs=prjkwargs,
                                                      plColor=f"C{pID}")
        
        return None

    @staticmethod
    def _project3D_onto_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              ax_obj=None,
                              proj_onto: str = 'xy',
                              plkwargs: dict = None,
                              plColor: str = 'C0'):
        """
        Given a full set of x, y, z values where (x_i, y_i, z_i) define coordinates in 3D-space, we plot the desired
        plane. This assumes that the plot itself is 2D.
        
        :param x: Values corresponding to the x-components.
        :param y: Values corresponding to the y-components.
        :param z: Values corresponding to the z-components.
        :param ax_obj: The matplotlib.Axes object to plot these on.
        :param proj_onto: Which plane to project onto. yz, xz, xy.
        :param plkwargs: plt.plot() kwargs.
        :param plColor: Color for the plot.
        :return:
        """
        assert proj_onto in ['yz', 'xz', 'xy'], "Invalid projection. Should be 'xy' | 'yz' | 'xz'"
        if plkwargs is None:
            plkwargs = {'alpha': 0.5, 'lw': 2, 'markersize': 5}
        if ax_obj is None:
            ax_obj = plt.gca()
    
        if proj_onto == 'yz':
            ax_obj.plot(y, z, "o-", color=plColor, **plkwargs)
        if proj_onto == 'xz':
            ax_obj.plot(x, z, "o-", color=plColor, **plkwargs)
        if proj_onto == 'xy':
            ax_obj.plot(x, y, "o-", color=plColor, **plkwargs)
    
        return None
    
    
    
    
    @staticmethod
    def calc_molar_conc(conc_ar: np.ndarray, lat_cons: float = 1e-9):
        """
        Given a NumPy array of concentrations in beads/voxels, we convert the concentrations into molar units. This
        also, then, requires a length-scale that is used to calculate the volumes. Therefore, we also supply a the lattice-constant.
        
        $(1/L^3)(10^{-3})(1/A)\rho$, where $L$ is the lattice-constant, $A$ is Avagadro's constant, and $\rho$ is the density.
        
        :param conc_ar: Array of concentrations in beads/voxels
        :param lat_cons: Lattice constant. Lattice separation in meters.
        :return: Concentrations rescaled to molar units.
        """
        
        assert lat_cons > 0., "Lattice constant must be positive!"
        
        AVAG_CONST = 6.02214e23
        DEN_CONST  = (1./AVAG_CONST)*(1.0e-3)*(1./(lat_cons**3.))
        
        return DEN_CONST * conc_ar
    

class _KeyFileStepsUtils(object):
    """
    A collection of methods to help with calculating the number of MC steps for different key files.
    """
    
    def __init__(self, eq_steps: int = 0, mc_steps: int = int(5e7), mc_cycles: int = 1):
        assert eq_steps >= 0, "Thermalization steps has to be a positive number."
        assert mc_steps >= 0, "Monte-Carlo steps has to be a positive number."
        assert mc_cycles >= 0, "Number of cycyles has to be a positive number."

        self.eq_steps = eq_steps
        self.mc_steps = mc_steps
        self.mc_cycles = mc_cycles
        
    def get_full_number_of_steps(self):
        """
        Calculates the _total_ number of MC steps, including thermalization.
        :return: t_eq + n_cycles * t_steps
        """
        return self.eq_steps + self.mc_cycles * self.mc_steps
    
    def get_total_run_steps(self):
        """
        Calculates the number of MC steps, excluding thermalization.
        :return: n_cycles * t_steps
        """
        return self.mc_cycles * self.mc_steps
    
    def get_number_of_data_points(self, data_freq: int = int(2.5e6)):
        """
        Returns the total number of data acquisitions given the supplied frequency. Only the last half of each run-cycle
        corresponds to steps where data are acquired.
        :param data_freq: Frequency, in MCSteps, of how often data are acquired.
        :return: (t_steps / 2 / data_freq)
        """
        assert data_freq > 0, "Data analysis frequency must be a non-zero number."
        return self.mc_steps // 2 // data_freq
    
    def get_freq_for_number_of_data(self, data_num: int = 1000):
        """
        Returns what the frequency _should_ be to get the number of data points supplied. Only the last half of each
        run-cycle corresponds to steps where data are acquired.
        :param data_num: Number of data points we wish to have.
        :return: (t_steps / 2 / data_num)
        """
        assert data_num >= 1, "Number of data points has to be at least 1"
        return self.get_number_of_data_points(data_freq=data_num)


class BeadsOfPolymers(object):
    """
    At the beginning of most Jupyter notebooks, I hard code a loop that basically generates a dictionary where the
    keys are normal strings, or letters like A B Wa Wb etc, and the value is the beadID that that bead will eventually
    have.
    This is an attempt to remove some of that grunt work, and maybe open a door to more internal consistency.
    """
    
    def __init__(self, list_of_beads: [str] = None):
        """
        Takes the list of strings and maps them to an incrementing index starting at 0.
        Just iterate of the list of strings.
        :param list_of_beads:
        """
        if list_of_beads is None:
            list_of_beads = ["A", "B"]
        assert isinstance(list_of_beads[0], str), "The input is a list of strings. E.G ['A', 'B', 'Wa', 'Wb'] for a" \
                                                  "system with 4 bead types."
        
        self.beads = {}
        self.num   = len(list_of_beads)
        for beadID, aBead in enumerate(list_of_beads):
            self.beads[str(aBead)] = beadID
    
    def __str__(self):
        """
        Returns a table style string for the mapping between the beads list and their typeIDs or beadIDs.
        :return: $beadStr  $beadID
        """
        dum_str = [f"|{aStr:<5} | {i:<5} |" for i, aStr in self.beads.items()]
        brace_line = "-" * 16 + "\n"
        return brace_line + "\n".join(dum_str) + "\n" + brace_line


class JobSubmission(object):
    """
    Bunch of helper functions to submit job for the EIT cluster. The majority of these functions simply run
    shell commands to delete, rename and read files. All the functions assume that we are in the directory
    where the commands will be run.
    """
    
    @staticmethod
    def submit_job(job_name: str = "lJob", job_queue: str = "pappu-compute",
                   lassi_exec: str = '/project/fava/packages/bin/lassi', only_show: bool = True):
        bsub_comm = f'bsub -n "1" -q "{job_queue}" -R "rusage[mem=4]" -o "log.txt" -J "{job_name}" '
        bsub_comm += f'"/.{lassi_exec}" '
        bsub_comm += f'-L "/bin/bash" '
        if only_show:
            return bsub_comm
        else:
            dum_run = sproc.run(bsub_comm, shell=True, capture_output=True, text=True)
            return dum_run.stdout[:-1], dum_run.stderr[:-1]
    
    @staticmethod
    def del_log():
        """
        Deletes the log file for LaSSI runs. The usual name is `log.txt`.
            > rm log.txt
        :return:
        """
        dum_comm = 'rm log.txt'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1], dum_run.stderr[:-1]
    
    @staticmethod
    def del_lammps_style_trjs():
        """
        Deletes all LAMMPS style trajectories in the current directory.
            > rm *.lammpstrj
        :return:
        """
        dum_comm = 'rm *.lammpstrj'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1], dum_run.stderr[:-1]
    
    @staticmethod
    def del_lassi_style_trjs():
        """
        Deletes all LAMMPS style trajectories in the current directory.
            > rm *.lassi
        :return:
        """
        dum_comm = 'rm *.lassi'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1], dum_run.stderr[:-1]
    
    @staticmethod
    def del_data():
        """
        Deletes all data files in the current directory.
            > rm *.dat
        :return:
        """
        dum_comm = 'rm *.dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1], dum_run.stderr[:-1]
    
    @staticmethod
    def check_data():
        """
        Checks if there are any data files in the current directory.
            > ls | grep -c .dat
        :return:
        """
        dum_comm = 'ls | grep -c .dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def read_log_for_ENDING(log_file: str = 'log.txt') -> bool:
        """
        Reads the log file for the ENDING keyword in the current directory. If the log has
        the keyword, that means that the LaSSI simulations successfully finished.
            > tail -n60 $log_file | grep -c ENDING
        :param log_file: Name of the log-file. Usually just 'log.txt'
        :return:
        """
        dum_comm = f'tail -n60 {log_file} | grep -c ENDING'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return bool(int(dum_run.stdout[:-1]))
    
    @staticmethod
    def read_log_for_segfault_failure(log_file='log.txt'):
        """
        Reads the log-file in the current directory and looks for `(core dumped)` which signifies that something
        went wrong -- usually with the job submission itself -- and that the run crash.
        > 'tail -n40 $log_file | grep "(core dumped)"
        :param log_file:
        :return:
        """
        dum_comm = f'tail -n40 {log_file} | grep "(core dumped)" '
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def read_log_for_lassi_crash(log_file='log.txt'):
        """
        Reads the log-file in the current directory and looks for `Crashed` which signifies that something
        went wrong and that the simulation failed a sanity check -- and that the run crashed.
        > 'tail -n40 $log_file | grep "Crashing."
        :param log_file:
        :return:
        """
        dum_comm = f'tail -n40 {log_file} | grep "Crashing." '
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def read_log_for_timing(file_name: str = 'log.txt') -> float:
        """
        Reads the log file for the mins keyword. If the log has the keyword, that means that the LaSSI simulations
        successfully finished in that many minutes.
            > tail -n 60 $file_name | grep mins | awk '{print $1}'
        :param file_name: Name of the log-file. Usually just 'log.txt'
        :return:
        """
        dum_comm = f'tail -n 60 {file_name} | grep  "mins"' + " | awk '{print $1}'"
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return float(dum_run.stdout[:-1])
    
    @staticmethod
    def rename_log(run_it: int = 0, log_name: str = 'log.txt'):
        """
        Renames the log-file and adds an iterator to count which number of renaming this is. This function is usually
        used to set up simulations that are using a previous simulation as a restart point.
            > mv log.txt R{$run_it}_log.txt
        :param run_it:
        :return:
        """
        dum_comm = f'mv {log_name} R{run_it}_log.txt'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_param(run_it: int = 0):
        """
        Renames the log-file and adds an iterator to count which number of renaming this is. This function is usually
        used to set up simulations that are using a previous simulation as a restart point. This way we can keep track
        of all the key-files used to run the systems.
            > mv param.key R{$run_it}_param.txt
        :param run_it:
        :return:
        """
        dum_comm = f'mv param.key R{run_it}_param.key'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_lammps_style_trj(run_it: int = 0):
        """
        Renames the LAMMPS style trajectory and adds an iterator to count which number of renaming this is.
        This function is usuallyused to set up simulations that are using a previous simulation as a restart point.
            > mv __trj.lammpstrj R{$run_it}_trj.lammpstrj
        :param run_it:
        :return:
        """
        dum_comm = f'mv __trj.lammpstrj R{run_it}_trj.lammpstrj'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_lassi_style_trj(run_it: int = 0):
        """
        Renames the LaSSI style trajectory and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __trj.lassi R{$run_it}_trj.lassi
        :param run_it:
        :return:
        """
        dum_comm = f'mv __trj.lassi R{run_it}_trj.lassi'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_top(run_it: int = 0):
        """
        Renames the LAMMPS style topology file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __topo.lammpstrj R{$run_it}_topo.lammpstrj
        :param run_it:
        :return:
        """
        dum_comm = f'mv __topo.lammpstrj R{run_it}_topo.lammpstrj'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_comden(run_it: int = 0):
        """
        Renames the COMDen file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __COMDen.dat R{$run_it}_COMDen.dat
        :param run_it:
        :return:
        """
        dum_comm = f'mv __COMDen.dat R{run_it}_COMDen.dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_clus(run_it: int = 0):
        """
        Renames the CLUS file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __CLUS.dat R{$run_it}_CLUS.dat
        :param run_it:
        :return:
        """
        dum_comm = f'mv __CLUS.dat R{run_it}_CLUS.dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_gr(run_it: int = 0):
        """
        Renames the GR file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __GR.dat R{$run_it}_GR.dat
        :param run_it:
        :return:
        """
        dum_comm = f'mv __GR.dat R{run_it}_GR.dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_MolClus(run_it: int = 0):
        """
        Renames the MolClus file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __GR.dat R{$run_it}_MolClus.dat
        :param run_it:
        :return:
        """
        dum_comm = f'mv __MolClus.dat R{run_it}_MolClus.dat'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_file(run_it: int = 0, file_name='COMDen.dat'):
        """
        Renames the provided file_name file and adds an iterator to count which number of renaming this is.
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
            > mv __{$file_name} R{$run_it}_{$file_name}
        :param run_it:
        :param file_name:
        :return:
        """
        dum_comm = f'mv __{file_name} R{run_it}_{file_name}'
        dum_run = sproc.run(dum_comm, shell=True, capture_output=True, text=True)
        return dum_run.stdout[:-1]
    
    @staticmethod
    def rename_data_files(run_it: int = 0):
        """
        General function that renames the following set of files and adds an iterator to count which number of renaming
        this is.
        {'trj.lammpstrj', 'trj.lassi', 'topo.lammpstrj', 'COMDen.dat', 'CLUS.dat', 'GR.dat', 'MolClus.dat'}
        This function is usually used to set up simulations that are using a previous simulation as a restart point.
        For the list above, runs
            > mv __{$file_name} R{$run_it}_{$file_name}
        :param run_it:
        :return:
        """
        for aFile in ['trj.lammpstrj', 'trj.lassi', 'topo.lammpstrj', 'COMDen.dat', 'CLUS.dat', 'GR.dat',
                      'MolClus.dat']:
            JobSubmission.rename_file(run_it=run_it, file_name=aFile)
        JobSubmission.rename_log(run_it)
        JobSubmission.rename_param(run_it)
    
    @staticmethod
    def get_all_segfault_runs(SimObj: SimulationSetup, print_to_screen: bool = True):
        """
        Loops over all the run directories and finds runs that have seg-faulted.
        :param SimObj:
        :param print_to_screen:
        :return:
        """
        dum_dir_list = SimObj._get_dirs_ofAll_perRep_perBox_perMol_perSys()
        tot_dir_list = []
        if print_to_screen:
            for dir_id, a_dir in enumerate(dum_dir_list):
                dum_log_text = JobSubmission.read_log_for_segfault_failure(log_file=f'{a_dir}log.txt')
                if len(dum_log_text):
                    this_dir = "/".join(a_dir.split("/")[-6:-1])
                    tot_dir_list.append(this_dir)
                    if print_to_screen:
                        print(this_dir)
    
        return tot_dir_list

    @staticmethod
    def get_all_crashed_runs(SimObj: SimulationSetup, print_to_screen: bool = True):
        """
        Loops over all the run directories and finds runs that have crashed due to failed sanity checks.
        :param SimObj:
        :param print_to_screen:
        :return:
        """
        dum_dir_list = SimObj._get_dirs_ofAll_perRep_perBox_perMol_perSys()
        tot_dir_list = []
        for dirID, a_dir in enumerate(dum_dir_list):
            _dum_log_text = JobSubmission.read_log_for_lassi_crash(log_file=f'{a_dir}log.txt')
            if len(_dum_log_text):
                this_dir = "/".join(a_dir.split("/")[-6:-1])
                tot_dir_list.append(this_dir)
                if print_to_screen:
                    print(this_dir)
        return tot_dir_list

    @staticmethod
    def resubmit_segfaulted_runs(crashed_list: list, SimObj: SimulationSetup, run_it: int,
                                 run_name: str, queue: str, only_show: bool = True):
        """
        For the list of runs that crashed due to seg-faults, we rename the existing log-file and resubmit the simulation. The idea
        being that the queue being used is a new one.
        :param crashed_list:
        :param SimObj:
        :param run_it:
        :param run_name:
        :param queue:
        :param only_show:
        :return:
        """
        dir_pre = SimObj.Sims_Path
    
        for dir_id, a_dir in enumerate(crashed_list):
            if not only_show:
                os.chdir(f"{dir_pre}{a_dir}")
                JobSubmission.rename_log(run_it=run_it)
        
            print(JobSubmission.submit_job(job_name=f'{run_name}_{dir_id}_{run_it}', only_show=only_show,
                                                 job_queue=queue))
            time.sleep(0.1)
    
        os.chdir(SimObj.CurrentDirectory)
        return None

    @staticmethod
    def resubmit_crashed_runs(crashed_list: list, SimObj: SimulationSetup, run_it: int,
                              run_name: str, queue: str, only_show: bool = True):
        """
        For the list of runs that crashed due to failed sanity checks, we:
            - save old param file
            - write new param file with the RNG_SEED = 0
            - rename the existing log file
            - resubmit the job
         rename the logs and resubmit the simulation. The idea
        being that the queue being used is a new one.
        :param crashed_list:
        :param SimObj:
        :param run_it:
        :param run_name:
        :param queue:
        :param only_show:
        :return:
        """
        dir_pre = SimObj.Sims_Path
    
        for dir_id, a_dir in enumerate(crashed_list):
            if not only_show:
                this_param = f'{a_dir}/param.key'
                IOUtils.copy_file_with_postfix(file_path=this_param, postfix='old')
                IOUtils.write_keyfile_with_new_seed_from_older_keyfile(old_file_path=f'{this_param}.old',
                                                               new_file_path=f'{this_param}',
                                                               new_seed=0)
                os.chdir(f"{dir_pre}{a_dir}")
                JobSubmission.rename_log(run_it=run_it)
        
            print(JobSubmission.submit_job(job_name=f'{run_name}_{dir_id}_{run_it}', only_show=only_show,
                                                 job_queue=queue))
            time.sleep(0.1)
    
        os.chdir(SimObj.CurrentDirectory)
        return None
    
    @staticmethod
    def check_and_resubmit_crashed_sims(SimObj: SimulationSetup, run_name: str, seg_queue: str, crsh_queue: str,
                                        run_it: int, print_to_screen: bool, only_show: bool):
        """
        Goes over all run conditions and checks for crashes, and depending on the type of crash, resubmits the simulation.
        Returns the total number of crashed simulations. Currently, we count the following crash-types:
         - seg-faults
         - sanity check failure in LaSSI
        :param SimObj:
        :param run_name:
        :param seg_queue:
        :param crsh_queue:
        :param run_it:
        :param print_to_screen:
        :param only_show:
        :return: number_of_crashed_runs
        """
        segfault_runs = JobSubmission.get_all_segfault_runs(SimObj=SimObj, print_to_screen=print_to_screen)
        JobSubmission.resubmit_segfaulted_runs(crashed_list=segfault_runs, SimObj=SimObj, run_it=run_it,
                                                     run_name=f"s{run_name}",
                                                     only_show=only_show, queue=seg_queue)
    
        crashed_runs = JobSubmission.get_all_crashed_runs(SimObj=SimObj, print_to_screen=print_to_screen)
        JobSubmission.resubmit_crashed_runs(crashed_list=crashed_runs, SimObj=SimObj, run_it=run_it,
                                                  run_name=f"c{run_name}",
                                                  only_show=only_show, queue=crsh_queue)
    
        return len(segfault_runs) + len(crashed_runs)

    @staticmethod
    def periodically_check_and_resubmit_crashed_sims(SimObj: SimulationSetup, run_name: str, seg_queue: str,
                                                     crsh_queue: str, print_to_screen: bool, only_show: bool,
                                                     num_loops: int, crsh_wait: int, success_wait: int):
        """
        A convenient wrapper to loop over and periodically check for crashed runs and resubmit them. If there were crashed
        simulations, we wait crsh_wait minutes before re-looping. If there were no crashes, we wait for success_wait minutes
        before looping over again.
        :param SimObj:
        :param run_name:
        :param seg_queue:
        :param crsh_queue:
        :param print_to_screen:
        :param only_show:
        :param num_loops:
        :param crsh_wait:
        :param success_wait:
        :return:
        """
    
        assert num_loops > 1, "There should be at least one loop!"
        assert crsh_wait > 0 and success_wait > 0, "Wait times must be positive numbers!"
        assert success_wait > crsh_wait, "Having success_wait < crsh_wait is not allowed!"
    
        for itID, anIter in enumerate(tqdm(range(1, num_loops + 1), ascii=True, desc=f'{"Checking For Crashes":<20}')):
    
            num_crsh = JobSubmission.check_and_resubmit_crashed_sims(SimObj=SimObj,
                                                                     run_name=run_name,
                                                                     seg_queue=seg_queue,
                                                                     crsh_queue=crsh_queue,
                                                                     print_to_screen=print_to_screen,
                                                                     only_show=only_show,
                                                                     run_it=anIter)
        
            if num_crsh > 0:
                print(f'Iter: {anIter:<4} had {num_crsh:<4} crashes.')
                time.sleep(crsh_wait * 60)
            else:
                print(f'Iter: {anIter:<4} had no crashes.')
                time.sleep(success_wait * 60)
    
        return None


