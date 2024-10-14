import numpy as np
import scipy as sc
import qutip as qt
import random as rnd
import math
from itertools import product


n_truncate=100  #Fock state cut-off
a = qt.destroy(n_truncate) # Generate annhiliation operator
a_dagger = qt.create(n_truncate)  # Generate creation operator
Iden= qt.identity(n_truncate)   # Generate identity oeprator

def Three_photon_state(triplicity,n_truncate):  # Example state for the general routine
    #triplicity real value
    a = qt.destroy(n_truncate)
    three_photon_operator = (1.0j*triplicity*(a**3+a.dag()**3)).expm()
    state = three_photon_operator * qt.fock(n_truncate, 0)
    return state

def Cubic_phase_state(cubicity, xi,n_truncate):  
    # q real ,xi complex 
    a = qt.destroy(n_truncate)
    qubic_operator = ((1j * cubicity / (2 * np.sqrt(2))) * (a + a.dag()) ** 3).expm()
    state = qubic_operator * qt.squeeze(n_truncate, xi) * qt.fock(n_truncate, 0)
    return state


def H_2(k,a,b):
    # evaluates the two-variable Hermite polynomial 
    p=np.arange(0,math.floor(k/2)+1)
    temp=a**(k-2*p)*b**p/( sc.special.factorial(p)*sc.special.factorial(k-2*p))
    return np.sum(temp)*sc.special.factorial(k)

def Fock_coefficient(p,q,optim_var):
    # computes Fock coefficients <q|S(xi) D(alpha)  |p>
    r=optim_var[0]
    theta=optim_var[1]              
    alpha=optim_var[2]+1.0j*optim_var[3]
    z=1/np.cosh(r)
    t=np.exp(1.0j*theta)*np.tanh(r)
    b=alpha*np.cosh(r)+np.conj(alpha)*np.exp(1.0j*theta)*np.sinh(r)
    a1=z*alpha
    a2=0.5*t
    b1=-z*np.conj(b)
    b1=-1*(np.conj(alpha)+alpha*np.conj(t))
    b2=-0.5*np.conj(t)
    j=np.min([p,q])
    temp=0
    for i in range(j+1):
        temp=temp+H_2(p-i,a1,a2)*H_2(q-i,b1,b2)*z**i /(sc.special.factorial(i)*sc.special.factorial(p-i)*sc.special.factorial(q-i))
    
    return temp*np.sqrt(sc.special.factorial(p)*sc.special.factorial(q)*z)*np.exp(-0.5*z*alpha*np.conj(b))




def Optim_function_Fock(optim_parameters,level_n,state):
    #Optimisation function optimized for states with few Fock states
    #state needs to be a state vector in Fock basis
    non_zeros=np.nonzero(state)[0]
    overlap=0.0
    if len(non_zeros)==1:
        for p in range(level_n+1):        
           overlap+= np.abs(Fock_coefficient(p, non_zeros[0], optim_parameters))**2
    else:
        for p in range(level_n+1):   
            temp1=0 
            for i in non_zeros:
                temp1+= state[i]*Fock_coefficient(p, i, optim_parameters)
            overlap+=np.abs(temp1)**2

    return (-1.0)*overlap



def Trial_state_1(n_truncate,r,exp_phi,alpha,p):
    #Computes the trial state using the stellar formalism
    operator= Iden
    if p >0:
        cr_a=np.cosh(r)*a_dagger
        sr_a=np.sinh(r)*exp_phi*a
        m_alpa=-1.0*alpha* Iden
        trafo_adagger= cr_a+sr_a+m_alpa
        for i in range(p):
            operator*= trafo_adagger
    return (1.0/np.sqrt(np.math.factorial(p)))*operator  * qt.squeeze(n_truncate,r*exp_phi)*qt.coherent(n_truncate,alpha)


def Trial_state_2(n_truncate,r,exp_phi,alpha,p):
    # Computes the trial state directly
    return  qt.squeeze(n_truncate,r*exp_phi)*qt.displace(n_truncate,alpha)*qt.fock(n_truncate,p)




def Optim_function_General(optim_parameters,n_truncate,level_n,state):
    #General optimization function
    #state needs to be a state vector in Fock basis
    #Change between Trial_state_1 and 2 
    r=optim_parameters[0]
    exp_phi=np.exp(1.0j*optim_parameters[1])
    alpha= optim_parameters[2]*np.exp(1.0j*optim_parameters[3])
    overlap=0.0
    for p in range(level_n+1):
        trial=Trial_state_2(n_truncate,r,exp_phi,alpha,p)
        overlap+= np.abs(trial.overlap(state))**2
        
    return (-1.0)*overlap



def find_optimal_fidelity_General(n_truncate, level_n,state):
    #optimisation parameters r,phi, Re(alpha),Im(alpha) 
    initial_guess =initial_guess =[rnd.uniform(0,1),rnd.uniform(0,2*np.pi),rnd.uniform(-2,2),rnd.uniform(-2,2)]
    result=sc.optimize.minimize(Optim_function_General,initial_guess, args=(n_truncate,level_n,state), method="BFGS")
    return (-1.0)*Optim_function_General(result.x,n_truncate,level_n,state)

def find_optimal_fidelity_Fock( level_n,state):
    #optimisation parameters r,phi, Re(alpha),Im(alpha) 
    initial_guess =initial_guess =[rnd.uniform(0,1),rnd.uniform(0,2*np.pi),rnd.uniform(-2,2),rnd.uniform(-2,2)]
    result=sc.optimize.minimize(Optim_function_Fock,initial_guess, args=(level_n,state), method="BFGS")
    return (-1.0)*Optim_function_Fock(result.x,level_n,state)


state = np.array([0,1])  # Fock state 1, use qutip to generate more complex states like the Three_photon_state
upper_Rank=2   # The highest Fock state for which the fidelity is computed
n_step= np.arange(0,upper_Rank)   
fid_array=[]
rep=100   # The optimization often gets stuck in local minima. This number is how often we repeat the optimization.


for n in n_step:
    print(n)
    temp_1=0
    for i in range(100):
        temp_2=find_optimal_fidelity_Fock(n,state)
        if temp_1<temp_2:
            temp_1=temp_2
    fid_array.append(temp_1)

print(fid_array)


np.save("stellar_fock_1.npy",fid_array)  # save data