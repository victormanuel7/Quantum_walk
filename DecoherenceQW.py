#              Load libraries...
# ==============================
from scipy import stats
from scipy import optimize
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")

exec(open('qinfLib.py').read())
#from qinfLib import *  # quantum information Lib

# =============================================================================


maxTime=int(input('Ingrese el número de pasos: '))
N=maxTime*2+1
#N = 201       # dimension of the walker,
p=N
dim = 2*N     # dimension of the composite system,
#maxTime = 100  # maximum number of iteration,
s=int(input('Ingrese número de caso: '))

def tr_c(rho, p):
    # reshape to do the partial trace easily using np.einsum
    reshaped_dm = rho.reshape([2, p, 2, p])
    # compute the partial trace
    reduced_dm = np.einsum('ijik->jk', reshaped_dm)
    return reduced_dm


qb0  = basisVec(2,0)              # state for the coin : |0>
qb1  = basisVec(2,1)              #     ...            : |1>
qb01 = (qb0-1j*qb1)/np.sqrt(2)    #     ...            :(|0> - i|1>)/sqr(2)

Ndit = basisVec(N,int((N+1)/2)-1) # state for the walker

psi0 = np.kron(qb01, Ndit)        # state for the composite system
Rho0 = Projector(psi0)            # initial density matrix




p=N
#////////////////////////////////Creaci5ón de operador S/////////////////////////////
#C0=np.array([1,0])
#C1=np.array([0,1])
#H00=np.outer(C0,C0); H01=np.outer(C0,C1); H10=np.outer(C1,C0); H11=np.outer(C1,C1)
#Hadamard=(H00+H01+H10-H11)/sqrt(2.)
#Hadamard=([[1,1j],[1j,1]])/sqrt(2.)

U    = np.dot(shiftOperator(N),coinOperator(N)) # Evolution operator

#                                           it's evolution baby...
# ================================================================
Rho   = Rho0

xdata = np.arange(-(N-1)/2, (N+1)/2)
xeven = xdata[(xdata%2)==0] 

#fig, (ax1,ax2) = plt.subplots(1,1, figsize=(8,3))
fig, (ax1, ax2) = plt.subplots(1,2,constrained_layout=True, figsize=(8,3))


H00=np.outer(np.array([1,0]),np.array([1,0]))
H11=np.outer(np.array([0,1]),np.array([0,1]))
if (s==1):
    P1=np.kron(H00,np.eye(N))
    P2=dagger(P1)
    P3=np.kron(H11,np.eye(N))
    P4=dagger(P3)
elif(s==2):
    #Segundo caso
    for k in range(N):
        posn1 = np.zeros(N)
        posn1[k] = 1 
        Pj=np.kron(np.eye(2),np.outer(posn1,posn1))
        if(k==0):
            r=[Pj]
            r1=[dagger(Pj)]
        else:
            r+=[Pj]
            r1+=[dagger(Pj)]
else:
    #Tercer caso
    for k in range(N):
        posn1 = zeros(N)
        posn1[k] = 1 
        Pj1=kron(H00,outer(posn1,posn1))
        Pj2=kron(H11,outer(posn1,posn1))
        if(k==0):
            r=[Pj1]
            r+=[Pj2]
            r1=[dagger(Pj1)]
            r1+=[dagger(Pj2)]
        else:
            r+=[Pj1]
            r+=[Pj2]
            r1+=[dagger(Pj1)]
            r1+=[dagger(Pj2)]
Rho1=Rho
x_data = np.arange(-(N-1)/2,(N+1)/2) # Walker's positions
x_even = x_data[(x_data%2)==0]       # Walker's even-psitions
x_odd  = x_data[(x_data%2)!=0]   
ydata = np.arange(0, maxTime)
desv1=np.zeros(maxTime)
for k in range(0, 11):
    p = k / 10.0
    Rho=Rho1
    for i in range(maxTime):
        tmp = np.dot(U, np.dot(Rho, dagger(U))) 
        if(p!=0):
            if(s==1):
                Rho = (1-p) * tmp + p * (P1.dot(tmp.dot(P2))+P3.dot(tmp.dot(P4)))
            elif(s==2):
                Op=0
                for x in range(0,N):
                    Op+=r[x].dot(tmp.dot(r1[x]))
                Rho = (1-p) * tmp + p *(Op)
            else:
                Op=0
                for x in range(0,2*N):
                    Op+=r[x].dot(tmp.dot(r1[x]))
                Rho = (1-p) * tmp + p *(Op) 
        else:
            Rho=tmp
            
        rho_w1 = tr_c(Rho,N)
        prob  = np.diag(rho_w1)
        desv=0
        for x in range(i):
            desv+=(x**2)*prob[maxTime-x]
            desv+=(x**2)*prob[maxTime+x]
#        print(abs(np.sqrt(desv)))
        desv1[i]=abs(np.sqrt(desv))      
        
    rho_w = tr_c(Rho,N)      # reduced density matrix for the walker
    prob  = np.diag(rho_w) # probability distribution for the walker
    print('Traza normal: ',np.trace(Rho))
    prb2=prob
    
    if(p==0):
        ax1.plot(x_even,prb2[np.nonzero((x_data%2)==0)],lw=0.7,label='P= %.1f' %p,marker=".")
        ax2.plot(ydata,abs(desv1),lw=0.7,label='P= %.1f' %p,marker=".")
    elif(p==1):
        ax1.plot(x_even,prb2[np.nonzero((x_data%2)==0)],lw=0.7,label='P= %.1f' %p,marker="+")
        ax2.plot(ydata,abs(desv1),lw=0.7,label='P= %.1f' %p,marker="+")
    else:
        ax1.plot(x_even,prb2[np.nonzero((x_data%2)==0)],lw=0.7,label='P= %.1f' %p)
        ax2.plot(ydata,abs(desv1),lw=0.7,label='P= %.1f' %p)
    
    sns.set_style('darkgrid')
ax1.set_xlabel("Position")
ax1.set_ylabel("Probability")
ax1.legend(loc='upper right')

ax2.set_xlabel("Iteration")
ax2.set_ylabel("Standard Deviation")
ax2.legend(loc='upper right')
plt.show()