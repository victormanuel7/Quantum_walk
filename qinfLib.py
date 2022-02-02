#!/usr/bin/python3

#              Load libraries...
# ==============================
from scipy import stats
from scipy import optimize

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


#                            define a function
# =========================================================================

# basisVec : Creates an element of the basis for the Hilbert space H of
#            dimension 'dim', in a vector form.
# Input  :
# dim    : (int) dimension of the Hilber space
# indx   : (int) index of the element different from zero (0<= indx <=dim-1)
# Output :
# vec    : vector of dim-rows and 1-column. All the elements are zero exept
#          the indx-th element with value 1.0
# Example: for a qubit the ground state |0> = (1,0)^T : basisVec(2,0)
# =========================================================================
def basisVec(dim, indx):
    vec = np.zeros((dim, 1))
    vec[indx, 0] = 1.0
    return (vec)


# dagger : transpose and conjugate the elements of a vector or matrix
# =========================================================================
def dagger(self):
    tc = self.conj().T
    return (tc)


# Normalize : normalize an arbitrary vector vec ==> vec/|vec|
# =========================================================================
def Normalize(vec):
    mgntd = np.sqrt(np.dot(dagger(vec), vec))
    vec = vec / mgntd
    return (vec)


# eucliNorm2 : square of euclidian norm of a vector X=(x1,...,xd)
#              (|X|)^2 = (sqrt(x1^2+...+xd^2))^2 = x1^2+...+xd^2
# =========================================================================
def eucliNorm2(vec):
    nrm2 = np.sum(np.absolute(vec) ** 2)
    return (nrm2)


# eucliNorm : square of euclidian norm of a vector X=(x1,...,xd)
#              |X| = sqrt(x1^2+...+xd^2)
# =========================================================================
def eucliNorm(vec):
    nrm = np.sqrt(eucliNorm2(vec))
    return (nrm)


# Projector : P = |psi><psi|
# Input :
# psi   : quantum state in 1-column vector representation
# Output:
# Prj   : square matrix as a result of
#         |psi><psi|=(c1,...,cd)^T * (c1*,...,cd*)
# =========================================================================
def Projector(psi):
    Prj = np.dot(psi, dagger(psi))
    return (Prj)


# QuantumState :
#
# =========================================================================
def QuantumState(dim):
    n = 2 * dim
    ndit = np.random.randn(n, 1)
    ndit = Normalize(ndit)
    psi = np.zeros((dim, 1), dtype=complex)
    psi[:, 0] = ndit[0:dim, 0] + (1j) * ndit[dim:, 0]
    return (psi)


# basisMat : Creates a basis element for matrices of dimension
#                  dim x dim.
# Input      :
# dim        : (int) dimension of a square matrix
# indx, jndx : (int) indexes for the element different from zero
# Output     : a square matrix with one element equal to 1.0 and the
#              remaining elements equal to zero
# =========================================================================
def basisMat(dim, indx, jndx):
    Mat = np.zeros((dim, dim))
    Mat[indx, jndx] = 1.0
    return (Mat)


# Hadamard : Unitary operation for qubits
#
#          | 1  1 |
#      H = |      | /sqrt(2)
#          | 1 -1 |
# =========================================================================
def Hadamard():
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return (H)


# Hadamard : Unitary operation for qutrits
#
#          | 1  1  1  |
#      H = | 1  w  w* | /sqrt(3)
#          | 1  w* w  |
# =========================================================================
def HadamardQutrit():
    w = -1.0 / 2.0 + 1j * (np.sqrt(3) / 2.0)
    w2 = -1.0 / 2.0 - 1j * (np.sqrt(3) / 2.0)
    H = np.array([[1, 1, 1], [1, w, w2], [1, w2, w]]) / np.sqrt(3)
    return (H)


# coinOperator :
#                               || 1         | 1         ||
#     |    |  __   /||   ___    ||   1       |   1       ||
#     |____| /\/\ / ||   ___    ||     ...   |     ...   ||
#     |    | \/\/   ||   ___    ||         1 |         1 ||
#     |    |       _||_         || ----------------------||  / sqrt(2)
#                               || 1         |-1         ||
#                               ||   1       |  -1       ||
#                               ||     ...   |     ...   ||
#                               ||         1 |        -1 ||
# =========================================================================
def coinOperator(dim):
    HxId = np.zeros((2 * dim, 2 * dim))
    Id = np.eye(dim)
    HxId[:dim, :dim] = Id  # Up-left identity
    HxId[:dim, dim:] = Id  # Up-right ...
    HxId[dim:, :dim] = Id  # down-left ...
    HxId[dim:, dim:] = -Id  # down-right ...
    HxId = HxId / np.sqrt(2)
    return (HxId)


# shiftOperator :
#
#   |0><0|otimes(sum_n |n-1><n|) + |1><1|otimes(sum_n |n+1><n|) =
#
#               || 0 1 0   0 |            ||
#               ||   0 1   . |     O      ||
#               ||     ... 1 |            ||
#               || 0       0 |            ||
#               || -----------------------||
#               ||           | 0        0 ||
#               ||           | 1 0        ||
#               ||     0     | . 1 ...    ||
#               ||           | 0    0 1 0 ||
# =========================================================================
def shiftOperator(dim):
    sop = np.zeros((2 * dim, 2 * dim))
    sop[:dim, :dim] = np.eye(dim, k=-1)
    sop[dim:, dim:] = np.eye(dim, k=1)
    return (sop)


# probabilityQW : Returns the probabilities for each position of the walker
#                 from the quantum state psi(k).
# =========================================================================
def probabilityQW(psi):
    nn = len(psi)
    mm = int(nn / 2)
    v1 = psi[:mm]
    v2 = psi[mm:]
    prb = np.absolute(v1) ** 2 + np.absolute(v2) ** 2
    return (prb)


# =========================================================================
#def TransitionMatrix(N):
#    M1 = np.eye(N, k=1)
#    M2 = np.eye(N, k=-1)
#    M3 = np.eye(N, k=(N - 1))
#    M4 = np.eye(N, k=-(N - 1))
#    TM = 0.5 * (M1 + M2 + M3 + M4)
#    return (TM)


# standarDeviation :
#
# =========================================================================
def standarDeviation(position, probability):
    pos2 = position ** 2
    stndrD = np.sqrt(np.dot(pos2, probability))
    return (stndrD)


# TransitionMatrix : Returns transition matrix for a linear random walk...
# =========================================================================
def TransitionMatrix(N):
    M1 = np.eye(N, k=1)
    M2 = np.eye(N, k=-1)
    # M3 = np.eye(N,k=(N-1))
    # M4 = np.eye(N,k=-(N-1))
    TM = 0.5 * (M1 + M2)
    TM[1, 0] = 1.0
    TM[N - 2, N - 1] = 1.0
    return (TM)


# ApplyMap :
# =========================================================================
def ApplyMap(map, rho):
    rho2 = np.zeros_like(rho)
    for element in map:
        tmp = np.dot(element, rho)
        rho2 = rho2 + np.dot(tmp, dagger(element))
    return (rho2)


# map1 : define a dynamical map for the open quantum walk
# =========================================================================
def map1(dim):
    lst = []
    TM  = TransitionMatrix(dim)
    for i in range(dim):
        tmp = np.zeros((dim, dim))
        tmp[:, i:(i+1)] = np.sqrt(TM[:, i:(i+1)])
        lst.append(tmp)
    return (lst)

# ========================================================================
# NonSelectiveBasisMeasurement:
# Non-Selective Projective Measurement over basis elements of a bipartite
# quantum system
# INPUT      :
# dim1       : dimension of the sub-system 1 (leftwards )
# dim2       : dimension of tne sub-system 2 (rightwards)
# modeSelect : select one of the following options
#                   0    measurement over the composite system
#                   1    measurement over the sub-system 1
#                   2    measurement over the sub-system 2
# rho        : initial density matrix for the composite system
# OUPUT      : newRho = Sum_j Pj (rho) Pj , where Pj are projectors
#              defined in accordance with 'modeSelect'
# ========================================================================
def NonSelectiveBasisMeasurement(dim1, dim2, modeSelect, rho):
    newRho = 0
    if modeSelect == 0:
        dim = dim1 * dim2
        for i in range(dim):
            Prj     = basisMat(dim, i, i)
            Left    = np.dot(Prj, rho)
            newRho += np.dot(Left, Prj)
    elif modeSelect == 1:
        for i in range(dim1):
            Prj     = basisMat(dim1, i, i)
            PrjExtended = np.kron(Prj, np.eye(dim2))
            Left    = np.dot(PrjExtended, rho)
            newRho += np.dot(Left, PrjExtended)
    elif modeSelect == 2:
        for i in range(dim2):
            Prj     = basisMat(dim2, i, i)
            PrjExtended = np.kron(np.eye(dim1), Prj)
            Left    = np.dot(PrjExtended, rho)
            newRho += np.dot(Left, PrjExtended)
    return (newRho)





