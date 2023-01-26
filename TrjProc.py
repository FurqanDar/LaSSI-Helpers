from __future__ import division
import numpy as np
import scipy as sp
import time
import pickle
import errno
import copy

__author__ = 'Furqan Dar'
__version__ = 0.1


def Extract_Frames_From(fileName):
    """
    Extract_Frames_From(fileName) expects a valid LAMMPS trajectory.
    Returns a NumPy array of all frames in order.
    Just goes through the entire file and finds where frames begin and end.
    Then I chop up and convert to numpy arrays.
    :param fileName:
    :return:
    """
    with open(fileName, 'r') as tFile:
        allDat = tFile.readlines()
        allDat = [aline[:-1] for aline in allDat]
        
        frmBgn = [lID + 1 for lID, aline in enumerate(allDat) if aline[:11] == 'ITEM: ATOMS']
        # The +1 is because the next line is the first atom
        
        frmEnd = [lID for lID, aline in enumerate(allDat) if aline == 'ITEM: TIMESTEP'][1:]
        frmEnd.append(len(allDat))
        # Need to include EOF, and first TIMESTEP is before the first frame
        
        thisTemp = []
        for id1, id2 in zip(frmBgn, frmEnd):
            oneFrame = allDat[id1:id2]
            oneFrame = np.array([np.fromstring(aline, sep=' ', dtype=int)[:] for aline in oneFrame], int)
            thisTemp.append(oneFrame)
        thisTemp = np.array(thisTemp)
    return thisTemp


def Extract_LAMMPSFrames_From(fileName):
    with open(fileName, 'r') as tFile:
        allDat = tFile.readlines()
        allDat = [aline[:-1] for aline in allDat]

        frmBgn = [lID + 1 for lID, aline in enumerate(allDat) if aline[:11] == 'ITEM: ATOMS']
        #The +1 is because the next line is the first atom

        frmEnd = [lID for lID, aline in enumerate(allDat) if aline == 'ITEM: TIMESTEP'][1:]
        frmEnd.append(len(allDat))
        #Need to include EOF, and first TIMESTEP is before the first frame

        thisTemp = []
        for id1, id2 in zip(frmBgn, frmEnd):
            oneFrame = allDat[id1:id2]
            oneFrame = np.array([np.fromstring(aline, sep=' ', dtype=float)[:] for aline in oneFrame], float)
            thisTemp.append(oneFrame)
        thisTemp = np.array(thisTemp)
    return thisTemp

"""
Slight extension of the function above where we only return the coordinates.
"""
def Extract_Coords_From(fileName):
    with open(fileName, 'r') as tFile:
        allDat = tFile.readlines()
        allDat = [aline[:-1] for aline in allDat]

        frmBgn = [lID + 1 for lID, aline in enumerate(allDat) if aline == 'ITEM: ATOMS id type mol x y z bP']
        #The +1 is because the next line is the first atom

        frmEnd = [lID for lID, aline in enumerate(allDat) if aline == 'ITEM: TIMESTEP'][1:]
        frmEnd.append(len(allDat))
        #Need to include EOF, and first TIMESTEP is before the first frame

        thisTemp = []
        for id1, id2 in zip(frmBgn, frmEnd):
            oneFrame = allDat[id1:id2]
            oneFrame = np.array([np.fromstring(aline, sep=' ', dtype=int)[3:6] for aline in oneFrame], int)
            thisTemp.append(oneFrame)
        thisTemp = np.array(thisTemp)
    return thisTemp

def Extract_LAMMPSCoords_From(fileName):
    with open(fileName, 'r') as tFile:
        allDat = tFile.readlines()
        allDat = [aline[:-1] for aline in allDat]

        frmBgn = [lID + 1 for lID, aline in enumerate(allDat) if aline[:11] == 'ITEM: ATOMS']
        #The +1 is because the next line is the first atom

        frmEnd = [lID for lID, aline in enumerate(allDat) if aline == 'ITEM: TIMESTEP'][1:]
        frmEnd.append(len(allDat))
        #Need to include EOF, and first TIMESTEP is before the first frame

        thisTemp = []
        for id1, id2 in zip(frmBgn, frmEnd):
            oneFrame = allDat[id1:id2]
            oneFrame = np.array([np.fromstring(aline, sep=' ', dtype=float)[-3:] for aline in oneFrame], float)
            thisTemp.append(oneFrame)
        thisTemp = np.array(thisTemp)
    return thisTemp

def Extract_LAMMPSCoor2ds_From(fileName):
    with open(fileName, 'r') as tFile:
        allDat = tFile.readlines()
        allDat = [aline[:-1] for aline in allDat]
        
        frmBgn = [lID + 1 for lID, aline in enumerate(allDat) if aline[:11] == 'ITEM: ATOMS']
        #The +1 is because the next line is the first atom
        
        frmEnd = [lID for lID, aline in enumerate(allDat) if aline == 'ITEM: TIMESTEP'][1:]
        frmEnd.append(len(allDat))
        #Need to include EOF, and first TIMESTEP is before the first frame
        
        thisTemp = []
        for id1, id2 in zip(frmBgn, frmEnd):
            oneFrame = allDat[id1:id2]
            oneFrame = np.array([np.fromstring(aline, sep=' ', dtype=float)[-3:] for aline in oneFrame], float)
            thisTemp.append(oneFrame)
        thisTemp = np.array(thisTemp)
    return thisTemp

"""
Calculates the euclidean distance between two points and handles periodic boundaries
"""
def Distance_On_Lattice(r1, r2, boxSize):
    rDis = np.zeros(3)
    rDiff = np.abs(r1-r2)
    for i in range(3):
        if rDiff[i] > boxSize[i]*0.5:
            rDis[i] = boxSize[i] - rDiff[i]
        else:
            rDis[i] = rDiff[i]
    return np.sqrt(np.sum(rDis**2.))

def Distance_On_Lattice_Sq(r1, r2, boxSize):
    rDis = np.zeros(3)
    rDiff=np.abs(r1-r2)
    for i in range(3):
        if rDiff[i] > boxSize[i]*0.5:
            rDis[i] = boxSize[i] - rDiff[i];
        else:
            rDis[i] = rDiff[i]
    return np.sum(rDis**2.)

def Distance_On_Lattice_NoPBC(r1, r2):
    rDiff=r1-r2
    return np.sqrt(np.sum(rDiff**2.))

def Distance_On_Lattice_NoPBC_Sq(r1, r2):
    rDiff=r1-r2
    return np.sum(rDiff**2.)

"""
Bead-to-bead distance maps.
"""
def Gen_DistanceMap_FromFrame(coord_list, boxSize):
    nAtoms  = len(coord_list)
    distMap = np.zeros((nAtoms, nAtoms))
    for id1 in range(nAtoms):
        for id2 in range(id1+1, nAtoms):
            distMap[id1, id2] += Distance_On_Lattice(coord_list[id1], coord_list[id2], boxSize)
    return distMap

def Gen_DistanceMaps_FromTraj(tot_traj, boxSize):
    totMaps = []
    for aFrame in tot_traj:
        coords_list = aFrame.T[3:6].T
        thisMap = Gen_DistanceMap_FromFrame(coords_list, boxSize)
        totMaps.append(thisMap)
    return np.array(totMaps)

def Gen_DistanceMaps_Norm(nAtoms):
    norm_arr = np.zeros((nAtoms, nAtoms))
    for id1 in range(nAtoms):
        for id2 in range(nAtoms):
            norm_arr[id1,id2] += 1
    return norm_arr

"""
Bead-to-bead internal scaling/distance maps. |i-j| \propto d(i,j)
"""
def Gen_IntDistance_FromFrame(coord_list, boxSize):
    nAtoms  = len(coord_list)
    distMap = np.zeros(nAtoms)
    for id1 in range(nAtoms):
        for id2 in range(id1+1, nAtoms):
            distMap[id2-id1] += 2.*Distance_On_Lattice(coord_list[id1], coord_list[id2], boxSize)
    return distMap

def Gen_IntDistances_FromTraj(tot_traj, boxSize):
    totMaps = []
    for aFrame in tot_traj:
        coords_list = aFrame.T[3:6].T
        thisMap = Gen_IntDistance_FromFrame(coords_list, boxSize)
        totMaps.append(thisMap)
    return np.array(totMaps)

def Gen_IntDistace_Norm(nAtoms):
    norm_arr = np.zeros(nAtoms)
    for id1 in range(nAtoms):
        for id2 in range(id1+1, nAtoms):
            norm_arr[id2-id1] += 1
    return norm_arr

"""
Radius of gyration of the chain
"""
def Gen_GyrTen_FromFrame(coord_list, boxSize):
    GyrTen = np.zeros((3,3))
    nAtoms = len(coord_list)
    crdsTrans = coord_list.T
    sysCOM = Calc_COM_FromFrame(crdsTrans.T, boxSize, False)
    cor_crds = []
    for i in range(3):
        crdsDiff = crdsTrans[i]-sysCOM[i];
        bPS = np.where(crdsDiff > boxSize[i]*0.5)[0]
        if len(bPS) > 0:
            crdsDiff[bPS] = crdsDiff[bPS]-boxSize[i]
            #print(i, "P", crdsDiff[bPS]-boxSize[i])
        bPS = np.where(crdsDiff < -boxSize[i]*0.5)[0]
        if len(bPS) > 0:
            crdsDiff[bPS] = crdsDiff[bPS] + boxSize[i]
            #print(i, "N", crdsDiff[bPS]+boxSize[i])
        cor_crds.append(crdsDiff)

    #print(sysCOM)
    for i in range(3):
        for j in range(i, 3):
            #GyrTen[i, j] = np.sum((crdsTrans[i]-sysCOM[i])*(crdsTrans[j]-sysCOM[j]))
            GyrTen[i, j] = np.sum(cor_crds[i]*cor_crds[j])
            GyrTen[i, j] /= nAtoms
            GyrTen[j, i] = GyrTen[i, j] + 0

    return GyrTen

def Gen_GyrTen_FromTraj(tot_traj, boxSize):
    totGyr = []
    for aFrame in tot_traj:
        coords_list = aFrame.T[3:6].T
        thisGyrTen = Gen_GyrTen_FromFrame(coords_list, boxSize)
        totGyr.append(thisGyrTen)
    return np.array(totGyr)

def Gen_GenShapeDescr_FromTraj(tot_traj, boxSize):
    totShape = []
    for aFrame in tot_traj:
        coords_list = aFrame.T[3:6].T
        thisGyrTen = Gen_GyrTen_FromFrame(coords_list, boxSize)
        thisShape  = Calc_GenShapeDescr_FromGyr(thisGyrTen)
        totShape.append(thisShape)
    return np.array(totShape)

"""
General calculations on lattice
"""
def Calc_COM_FromFrame(coord_list, boxSize, on_lattice=True):
    nAtoms = len(coord_list)
    zeta   = np.zeros(3);
    xi     = np.zeros(3);
    nConst = 2*np.pi/np.array(boxSize)
    for aBead in coord_list:
        zeta += np.sin(aBead*nConst)
        xi   += np.cos(aBead * nConst)
    zeta /= nAtoms
    xi   /= nAtoms

    sysCOM = (np.arctan2(-zeta, -xi) + np.pi)/nConst

    if on_lattice:
        return sysCOM.round(0)
    else:
        return sysCOM

def Calc_DiagGyrTen(RgT: np.ndarray):
    """
    Function that diagonalizes a given Gyration Tensor
    Remember that the eigenvalues need to be sorted from largest to smallest.
    :param RgT
    :return np.ndarray (3,) of ordered eigenvalues of RgT
    """
    eVals = np.linalg.eigvals(RgT)
    idx   = eVals.argsort() #Finding indecies for correct order
    eVals = eVals[idx]
    return eVals


def Calc_ASpher(RgEVals):
    """
    Function to calculate the asphericity given a diagonalized Gyration Tensor.
    Therefore, the input is really just a vector of length 3.
    Note that a uniform particle distribution inside a cube is also considered
    spherical, which is a drawback of this description.
    :param RgEVals:
    :return:
    """
    dV = 3./2.*RgEVals[2] - np.sum(RgEVals)/2.
    return dV


def Calc_ACylin(RgEVals):
    """
    Function to calculate the acylindricity from a diagonalized Gyration Tensor.
    Input structure same as above.
    :param RgEVals:
    :return:
    """
    return RgEVals[1] - RgEVals[0]

def Calc_GenShAn(RgEVals):
    """
    Function to calculate the relative shape anisotropy from a diagonalized Gyration Tensor.
    Input structure same as above.
    :param RgEVals:
    :return:
    """
    num = Calc_ASpher(RgEVals)**2 + (3./4.)*(Calc_ACylin(RgEVals)**2)
    denom = np.sum(RgEVals)**2
    return num/denom

def Calc_GenShapeDescr(RgEVals):
    """
    Function to calculate the general shape description from a diagonalized Gyration Tensor.
    Basically combines the three functions defined above, which should hopefully take care of
    uniform distributions produced by normal diffusion.
    :param RgEVals:
    :return:
    """
    GyrRad  = np.sum(RgEVals)
    ASp     = Calc_ASpher(RgEVals)
    ACyl    = Calc_ACylin(RgEVals)
    kappaSq = ASp ** 2 + (3. / 4.) * (ACyl ** 2)
    kappaSq = kappaSq / (GyrRad ** 2)
    return np.array([np.sqrt(GyrRad), ASp, ACyl, kappaSq])

def Calc_GenShapeDescr_FromGyr(GyrTen):
    """
    This version takes a full gyration tensor instead of just the eigen values.
    :param RgEVals:
    :return: Generalized shape descriptors: Gyration Radius, Ashpericity, Acyclindricity and Anisotropy
    """
    RgEVals = Calc_DiagGyrTen(GyrTen)
    GyrRad  = np.sum(RgEVals)
    ASp     = Calc_ASpher(RgEVals)
    ACyl    = Calc_ACylin(RgEVals)
    kappaSq = ASp**2 + (3./4.)*(ACyl**2)
    kappaSq = kappaSq/(GyrRad**2)
    return np.array([np.sqrt(GyrRad), ASp, ACyl, kappaSq])