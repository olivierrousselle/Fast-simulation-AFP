# -*- coding: utf-8 -*-

import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.constants as cst

start_time = time.time()

n_air = 1
θF = 48*2*np.pi/360 # Construction angle, in rad
α_t = 18*2*np.pi/360 # Angle of taper, in rad
angleB = 0*2*np.pi/360 # Angle of proton beam
yB = 2 # y-intercept of proton beam, in mm
Np = 100 # Number of proton tracks
λmin = 180*10**-9 # in m.
λmax = 650*10**-9
Emin = cst.h*cst.c/λmax*6.242*10**18 # in eV
Emax = cst.h*cst.c/λmin*6.242*10**18
n_min = 7004/(1.614*180-60.57)**2+1.449
n_max = 7004/(1.614*650-60.57)**2+1.449
n_mean = (n_min+n_max)/2
Z = 1
β = 1
def density(E):
    E = E/(6.242*10**18) # in J
    λ = cst.h*cst.c/E*10**9 # in nm
    n = 7004/(1.614*λ-60.57)**2+1.449 # Refraction index
    return 1-1/(n**2*β**2)
lz = 6 # Length of lz (in mm)
Steplength = 4*lz/np.sin(θF-angleB)*0.1 # in cm.
#N_mean = 370*Z**2*Steplength*abs(integrate.quad(density,Emin,Emax)[0])
N_mean = 370*Z**2*Steplength*(1-1/(n_mean**2*β**2))*(Emax-Emin)
print("N_mean = ", N_mean)       
M_E = np.linspace(Emin,Emax,10**6)
f = [density(E) for E in M_E] # Density distribution for energy
f = f/sum(f)
mat = M_E[np.random.choice(len(M_E),10**6,p=list(f))]/(6.242*10**18)

t = 2 # Train (1, 2, 3 or 4) 

fast_simulation_input = nb.int32
fast_simulation_out = nb.types.Tuple((nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int32, nb.int32))

@nb.jit(fast_simulation_out(fast_simulation_input), nopython=True)
def fast_simulation(Np):
    lx = [3, 5, 5, 5] # Length of lx (in mm) for each train
    lx1 = 3 # For the train 1 with taper
    lx2 = 2 
    # Matrix of radiator length for each train
    L_rad = [[63.4, 57.8, 52.2, 46.5], [59.2, 53.5, 47.9, 42.3], [52.9, 47.3, 41.7, 36.0], [46.7, 41.0, 35.4, 29.8]]
    # Matrix of light guide length for each train
    L_lg = [70.3, 65.2, 60.1, 55.0] 
    L1 = []
    L2 = []
    L3 = []
    L3_cut_edge_effect = []
    Time = []
    Case = []
    Bar = []
    count_tot = 0  # Number of photons generated
    count_passed = 0 # Number of passed photons
    for n in range(Np):
        N = np.random.poisson(N_mean) # Number of photon tracks per event
        for r in range(N):
            count_tot+=1
            xA_i = lx[t-1]/2 # Position of vertex point A
            zA_i = np.random.rand()*4*lz
            yA_i = zA_i/np.tan(θF-angleB)+yB
            StepLength = zA_i/np.sin(θF-angleB) # length between the side of radiator and vertex point A 
            E = mat[int(np.random.rand()*len(mat))] # Energy of photon
            λ = cst.h*cst.c/E*10**9 # Wavelength of photon, in nm 
            n_silica = 7004/(1.614*λ-60.57)**2+1.449 # Refraction index
            θch = np.arccos(1/(β*n_silica))        
            θc = np.arcsin(n_air/n_silica) # Critical angle
            ϕ = np.random.uniform(-np.pi,np.pi)
            if -np.pi<ϕ<np.pi/2:
                ϕ_geant = ϕ+np.pi/2  
            else:
                ϕ_geant = ϕ-3*np.pi/2   
            δ = np.arcsin(np.sin(θch)*np.cos(ϕ)) # Projection angles
            α = np.arcsin(np.tan(δ)*np.tan(ϕ))+angleB
            α_abs = abs(α)
            δ_abs = abs(δ)
            # Initial parameters for each track
            case = 0 # Case (0,1,2 or 3)
            case3 = False # True if the light ray hits the cut edge
            length = 0
            bounds = 0 # Number de times the light ray hits the sides of TOF
            taper = False # True if the light ray hits the taper (for train 1)
            cut_edge_effect = False # True if there is cutting edge effect
            condition = False
            detected = False
    
            b=int(zA_i/lz)+1 # Bar (1, 2, 3 or 4)
            
            if (α>=0): 
                xA = xA_i
                yA = yA_i
                zA = zA_i
                α1 = abs(θF-α_abs)
                condition = True
            else:
                l=StepLength+(yA_i-zA_i/np.tan(θF))*(np.cos(θF)+np.sin(θF)/np.tan(α_abs))  
                if l<b*lz/np.sin(θF): 
                    yA = l*np.cos(θF) # Coordinates of the new initial point A'
                    zA = l*np.sin(θF)
                    xA = xA_i+abs(yA-yA_i)*np.tan(δ)*np.sin(θF)/np.sin(α_abs)
                    length = np.sqrt((xA-xA_i)**2+(yA-yA_i)**2+(zA-zA_i)**2) # length [AA'] to add at the end
                    case3 = True # Photon hits the cut edge
                    bounds = bounds+1 
                    α1 = abs(θF-α_abs)
                    condition = True
                elif l>b*lz/np.sin(θF): 
                    if np.cos(θc)>np.sin(θF+α_abs)*np.cos(δ_abs):
                        if α_abs<np.pi/2-θF:
                            xA = xA_i
                            yA = yA_i
                            zA = zA_i 
                            α1 = θF+α_abs
                            condition = True
                        elif α_abs>np.pi/2-θF: 
                            xA = xA_i
                            slz = 0
                            for j in range(b-1):
                                slz += lz
                            zA = np.random.rand()*lz+slz # We choose a random point on the cut edge
                            yA = zA/np.tan(θF)
                            length = 2*lz
                            case3 = True
                            bounds = bounds+2
                            cond1 = False
                    else:
                        if b*lz/np.sin(θF)<l<4*lz/np.sin(θF):
                            cut_edge_effect = True
                            b = int(l/(lz/np.sin(θF)))+1
                            yA = l*np.cos(θF) # Coordinates of the new initial point A'
                            zA = l*np.sin(θF)
                            xA = xA_i+abs(yA-yA_i)*np.tan(δ)*np.sin(θF)/np.sin(α_abs)
                            length = np.sqrt((xA-xA_i)**2+(yA-yA_i)**2+(zA-zA_i)**2)
                            case3 = True   
                            bounds = bounds+1 
                            α1 = abs(θF-α_abs)
                            condition = True
            
            if condition:
                # conditions of total reflection
                cond0 = np.cos(θc)>np.sin(α1)*np.cos(δ_abs) # plan 0 of the radiator
                cond1 = np.cos(θc)>np.sin(δ_abs)  # plan 1 of the radiator
                cond2 = np.cos(θc)>np.cos(α1)*np.cos(δ_abs)  # plan 2 of the light guide
                #cond2 = np.sin(θF-θc)>np.sin(α_abs)*np.cos(δ_abs)
                cond3 = np.cos(θc)>np.sin(α_abs)*np.cos(δ_abs) # Cut edge (plan 3) 
                
            # Determination of the case (1, 2 or 3)
            if (condition and cond0 and cond1):
                yA = yA-lz/np.tan(θF)*(b-1)
                zA = zA-lz*(b-1)
                #xA = xA-sum(lx[j] for j in range(t-1))            
                δ1 = np.arctan(np.tan(δ)/np.cos(α1)) 
                if (δ1<0):
                    δ1_abs = abs(δ1)
                    if (xA/np.tan(δ1_abs)+yA > L_rad[t-1][b-1]-lx[t-1]):
                        case = 1
                    else:
                        d = (L_rad[t-1][b-1]-lx[t-1]-yA)*np.tan(δ1)+xA
                        int_d = int(d/lx[t-1])
                        d2 = lx[t-1]*np.tan(δ1)
                        if (int_d+1)%2==0:
                            if d+d2 > int_d*lx[t-1]-lx[t-1]:
                                case = 1
                                if t==1:
                                    d3 = (lx1+lx2/np.tan(α_t))*np.tan(δ1)
                                    dlim = int_d*lx[t-1]-lx1-lx2
                                    if d+d3 < dlim:
                                        taper=True
                            else:
                                case = 2
                                yC = abs(int_d*lx[t-1]-lx[t-1]-xA)/np.tan(δ1_abs)+yA
                                if t==1:
                                    d3 = (lx1+lx2)*np.tan(δ1)
                                    dlim = int_d*lx[t-1]-lx1-lx2/np.tan(α_t)
                                    if d+d3 > dlim:
                                        taper = True                                
                        else:
                            if (δ1_abs<np.pi/4):
                                case = 1
                            else:
                                case = 0  
                    
                else:
                    if ((lx[t-1]-xA)/np.tan(δ1)+yA>L_rad[t-1][b-1]):
                        case = 1
                    elif ((lx[t-1]-xA)/np.tan(δ1)+yA>L_rad[t-1][b-1]-lx[t-1]):
                        case = 2 
                        if t==1:
                            taper = True
                    else:
                        d = (L_rad[t-1][b-1]-lx[t-1]-yA)*np.tan(δ1)-(lx[t-1]-xA)
                        int_d = int(d/lx[t-1])
                        d2 = lx[t-1]*np.tan(δ1)
                        if int_d%2==0:
                            if d+d2 < int_d*lx[t-1]+lx[t-1]:
                                case = 1
                                if t==1:
                                    d3 = (lx1+lx2/np.tan(α_t))*np.tan(δ1)
                                    dlim = int_d*lx[t-1]+lx1+lx2
                                    if d+d3 > dlim:
                                        taper = True # Photon hits the taper
                            else:
                                case = 2
                                yC = (lx[t-1]-xA+int_d*lx[t-1]+lx[t-1])/np.tan(δ1)+yA
                                if t==1:
                                    d3 = (lx1+lx2)*np.tan(δ1)
                                    dlim = int_d*lx[t-1]+lx1+lx2/np.tan(α_t)
                                    if d+d3 < dlim:
                                        taper = True                      
                        else:
                            if (δ1<np.pi/4):
                                case = 1
                            else:
                                case = 0  
                    
            Labs = (121.66+4.65441*λ-0.00606166*λ**2+2.60047e-06*λ**3)/2 # Attenuaton length
            µabs = 1/Labs # Attenuation coefficient
            QE = 0.2 # Quantum reflectivity  
            
            if case==1:
                δ2 = abs(δ1)
                δ3 = abs(δ)
                α2 = α1 
                θin = np.arccos(np.cos(α2)*np.cos(δ3))
                ϕin = np.arctan(np.tan(α2)/np.tan(δ2))+np.pi/2
                R0 = ((n_silica-n_air)/(n_silica+n_air))**2
                R = R0+(1-R0)*(1-np.cos(θin))**5                
                if taper==True: # In the case where the train has taper (train1)
                    δ2 = abs(δ2-2*α_t)
                    δ3 = abs(δ3-2*np.arctan(np.cos(α2)*np.tan(α_t)))            
                bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(np.tan(α1)/lz+np.tan(δ1)/lx[t-1])+L_lg[t-1]*(np.tan(α2)/lz+np.tan(δ2)/5)) # Number of bounds on the sides of the detector
                if (case3==False):
                    #Calculation of length from vertex point to the photomultiplier
                    l = (L_rad[t-1][b-1]-yA)/(np.cos(δ)*np.cos(α1))+L_lg[t-1]/(np.cos(α2)*np.cos(δ3))
                    pabs = 1-np.exp(-µabs*l) # Probability of absorption
                    x1 = np.random.rand()
                    x2 = np.random.rand()
                    x3 = np.random.rand()
                    x4 = np.random.rand()
                    # Conditions of attenuation
                    if (x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R): 
                        L1.append(l)
                        case = 1 
                        detected = True # Photon reaches the PMT
                    else:
                        case=0
                else:
                    l = (L_rad[t-1][b-1]-yA)/(np.cos(δ)*np.cos(α1))+L_lg[t-1]/(np.cos(α2)*np.cos(δ3))+length
                    pabs = 1-np.exp(-µabs*l)
                    x1 = np.random.rand()
                    x2 = np.random.rand()
                    x3 = np.random.rand()
                    x4 = np.random.rand()
                    if (cond3 and x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R): 
                        L3.append(l)          
                        if (cut_edge_effect):
                            L3_cut_edge_effect.append(l)
                        case = 3
                        detected = True                     
                    else:
                        case=0  
                    
            if (cond2 and case==2):
                δ2 = np.pi/2-abs(δ1)
                δ3 = np.arcsin(np.cos(δ)*np.cos(α1))
                α2 = np.arctan(np.tan(α1)*np.tan(δ2))
                θin = np.arccos(np.cos(α2)*np.cos(δ3))
                ϕin = np.arctan(np.tan(α2)/np.tan(δ2))+np.pi/2
                R0 = ((n_silica-n_air)/(n_silica+n_air))**2
                R = R0+(1-R0)*(1-np.cos(θin))**5 # Schlick's approximation                  
                if taper==True:
                    δ2 = abs(δ2-2*α_t)
                    δ3 = abs(δ3-2*np.arctan(np.cos(α2)*np.tan(α_t)))
                bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(np.tan(α1)/lz+np.tan(δ1)/lx[t-1])+L_lg[t-1]*(np.tan(α2)/lz+np.tan(δ2)/5))
                if (case3==False):
                    l = (yC-yA)/(np.cos(δ)*np.cos(α1))+L_lg[t-1]/(np.cos(α2)*np.cos(δ3))
                    pabs = 1-np.exp(-µabs*l)
                    x1 = np.random.rand()
                    x2 = np.random.rand()
                    x3 = np.random.rand()
                    if (cond2 and x1<1-pabs and x2<0.99**bounds and x3>R): 
                        L2.append(l)
                        case = 2 
                        detected = True                    
                    else:
                        case=0
                else:     
                    l = (yC-yA)/(np.cos(δ)*np.cos(α1))+L_lg[t-1]/(np.cos(α2)*np.cos(δ3))+length
                    pabs = 1-np.exp(-µabs*l)
                    x1 = np.random.rand()
                    x2 = np.random.rand()
                    x3 = np.random.rand()
                    if (cond2 and cond3 and x1<1-pabs and x2<0.99**bounds and x3>R):
                        L3.append(l)           
                        if (cut_edge_effect):
                            L3_cut_edge_effect.append(l)
                        case = 3 
                        detected = True 
                    else:
                        case=0
            
            if detected:
                x_passed = np.random.rand()
                if x_passed<QE:
                    count_passed+=1
                Time.append(l/(n_silica*cst.c)*10**9) # Time of trajectory from vertex point to PMT
                Case.append(case)
                Bar.append(b)
                
    L1_numba = np.zeros(len(L1))
    for i in range(len(L1)):
        L1_numba[i] = L1[i]
    L2_numba = np.zeros(len(L2))
    for i in range(len(L2)):
        L2_numba[i] = L2[i]
    L3_numba = np.zeros(len(L3))
    for i in range(len(L3)):
        L3_numba[i] = L3[i] 
    L3_cut_edge_effect_numba = np.zeros(len(L3_cut_edge_effect))
    for i in range(len(L3_cut_edge_effect)):
        L3_cut_edge_effect_numba[i] = L3_cut_edge_effect[i] 
    Time_numba = np.zeros(len(Time))
    for i in range(len(Time)):
        Time_numba[i] = Time[i]
    Case_numba = np.zeros(len(Case))
    for i in range(len(Case)):
        Case_numba[i] = Case[i] 
    Bar_numba = np.zeros(len(Bar))
    for i in range(len(Bar)):
        Bar_numba[i] = Bar[i] 

    return L1_numba, L2_numba, L3_numba, L3_cut_edge_effect_numba, Time_numba, Case_numba,  Bar_numba, count_tot, count_passed
      
L1, L2, L3, L3_cut_edge_effect, Time, Case, Bar, count_tot, count_passed = fast_simulation(Np)
     
print("Time execution : %s seconds ---" % (time.time() - start_time))        

  
count1 = len(L1)
count2 = len(L2)
count3 = len(L3)
count = count1+count2+count3 # Number of tracks detected in the PMT
L = np.zeros(len(L1)+len(L2)+len(L3))
for i in range(len(L1)):
    L[i] = L1[i]
for i in range(len(L2)):
    L[len(L1)+i] = L2[i]
for i in range(len(L3)):
    L[len(L1)+len(L2)+i] = L3[i]    

Track = []
track = open("train2_100000.txt","r") # Open file of tracks
for line in track:
    Z=line.split(' ')
    Track.append([float(i) for i in Z])
track.close()
count_geant = 0
L_geant = []
Time_geant = []
N_geant = len(Track)
for r in range(N_geant):
    if (Track[r][9]!=0):
        L_geant.append(Track[r][9])
        count_geant+=1
        θch=float(Track[r][4])
        n_silica = 1/np.cos(θch)
        Time_geant.append(Track[r][9]/(n_silica*cst.c)*10**9)

print("Total number of photon tracks generated = ", count_tot)
print("Number of photons reaching the sensor = ", count)        
print("Percentage of photons reaching PMT - fast simulation = {0}%".format(round(count/count_tot*100)))
print("Case 1 = {0}%".format(round(count1/count*100)))
print("Case 2 = {0}%".format(round(count2/count*100)))
print("Case 3 = {0}%".format(round(count3/count*100))) 
print("Percentage of photons reaching PMT - Geant4 = {0}%".format(round(count_geant/N_geant*100)))
print("Percentage of photons passed = {0}%".format(round(count_passed/count_tot*100)))
   
plt.figure()
plt.hist(Case, color='white', edgecolor = 'blue')
plt.title('Case')
plt.xlabel("Case")
plt.ylabel("Number of tracks")

plt.figure()
plt.hist(Bar, color='white', edgecolor = 'blue')
plt.title('Number of photons reaching the sensor, per bar - train 2')
plt.xlabel("Bar")
plt.ylabel("Number of tracks")

plt.figure()
plt.hist(L, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
plt.title('All tracks - Train 2 - Fast simulation')
plt.xlabel("Length (mm)")
plt.ylabel("Number of tracks")
plt.legend()

plt.figure()
plt.hist(L_geant, histtype='stepfilled', color='white', edgecolor = 'red', alpha=0.8, range=(90,400), bins=500, label = "Geant4")
plt.title('Length of photon tracks - Train 2')
plt.xlabel("Length (mm)")
plt.ylabel("Number of tracks")
plt.legend()
