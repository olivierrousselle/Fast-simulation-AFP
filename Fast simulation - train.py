# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import time
from numpy.random import *
from numpy import *
from matplotlib.pyplot import *
from math import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.constants import *

start_time = time.time()

lx = [3, 5, 5, 5] # Length of lx (in mm) for each train
lx1=3 # For the train 1 with taper
lx2=2 
lz = 6 # Length of lx (in mm)
# Matrix of radiator length for each train
L_rad = [[63.4, 57.8, 52.2, 46.5], [59.2, 53.5, 47.9, 42.3], [52.9, 47.3, 41.7, 36.0], [46.7, 41.0, 35.4, 29.8]]
# Matrix of light guide length for each train
L_lg = [70.3, 65.2, 60.1, 55.0] 
n_air = 1
# Length of trajectoires for each case, with fast simulation
L1 = []
L2 = []
L3 = []
L3_cut_edge_effect  =[]
Time = []
Case = []
count_tot = 0  # Number of photons generateed
count = 0 # Number of photons detected 
count1 = 0 # Number of photons detected per case
count2 = 0
count3 = 0
count_passed = 0 # Number of passed photons
count_error = 0

θF = 48*2*pi/360 # Construction angle, in rad
α_t = 18*2*pi/360 # Angle of taper, in rad
angleB = 0*2*pi/360 # Angle of proton beam
yB = 2 # y-intercept of proton beam, in mm
Np = 30 # Number of proton tracks
λmin = 180*10**-9 # in m.
λmax = 650*10**-9
Emin = h*c/λmax*6.242*10**18 # in eV
Emax = h*c/λmin*6.242*10**18
n_min = 7004/(1.614*180-60.57)**2+1.449
n_max = 7004/(1.614*650-60.57)**2+1.449
n_mean = (n_min+n_max)/2
Z = 1
β = 1
def density(E):
    E = E/(6.242*10**18) # in J
    λ = h*c/E*10**9 # in nm
    n = 7004/(1.614*λ-60.57)**2+1.449 # Refraction index
    return 1-1/(n**2*β**2)
Steplength = 4*lz/sin(θF-angleB)*0.1 # in cm.
#N_mean = 370*Z**2*Steplength*abs(integrate.quad(density,Emin,Emax)[0])
N_mean = 370*Z**2*Steplength*(1-1/(n_mean**2*β**2))*(Emax-Emin)
M_E = linspace(Emin,Emax,10**6)
f = [density(E) for E in M_E] # Density distribution for energy
f = f/sum(f)
mat = M_E[random.choice(len(M_E),10**6,p=list(f))]/(6.242*10**18)

t = 2 # Train (1, 2, 3 or 4) 
 
for n in range(Np):
    N = poisson(N_mean) # Number of photon tracks per event
    for r in range(N):
        count_tot+=1
        xA_i = lx[t-1]/2 # Position of vertex point A
        zA_i = rand()*4*lz
        yA_i = zA_i/tan(θF-angleB)+yB
        StepLength = zA_i/sin(θF-angleB) # length between the side of radiator and vertex point A 
        E = mat[int(rand()*len(mat))] # Energy of photon
        λ = h*c/E*10**9 # Wavelength of photon, in nm 
        n_silica = 7004/(1.614*λ-60.57)**2+1.449 # Refraction index
        θch = arccos(1/(β*n_silica))        
        θc = arcsin(n_air/n_silica) # Critical angle
        ϕ = uniform(-pi,pi)
        if -pi<ϕ<pi/2:
            ϕ_geant=ϕ+pi/2  
        else:
            ϕ_geant=ϕ-3*pi/2   
        δ = arcsin(sin(θch)*cos(ϕ)) # Projection angles
        α = arcsin(tan(δ)*tan(ϕ))+angleB
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
            xA=xA_i
            yA=yA_i
            zA=zA_i
            α1=abs(θF-α_abs)
            condition = True
        else:
            l=StepLength+(yA_i-zA_i/tan(θF))*(cos(θF)+sin(θF)/tan(α_abs))  
            if l<b*lz/sin(θF): 
                yA = l*cos(θF) # Coordinates of the new initial point A'
                zA = l*sin(θF)
                xA = xA_i+abs(yA-yA_i)*tan(δ)*sin(θF)/sin(α_abs)
                length=sqrt((xA-xA_i)**2+(yA-yA_i)**2+(zA-zA_i)**2) # length [AA'] to add at the end
                case3 = True # Photon hits the cut edge
                bounds=bounds+1 
                α1=abs(θF-α_abs)
                condition = True
            elif l>b*lz/sin(θF): 
                if cos(θc)>sin(θF+α_abs)*cos(δ_abs):
                    if α_abs<pi/2-θF:
                        xA=xA_i
                        yA=yA_i
                        zA=zA_i 
                        α1=θF+α_abs
                        condition = True
                    elif α_abs>pi/2-θF: 
                        xA = xA_i
                        zA = rand()*lz+sum(lz for j in range(b-1)) # We choose a random point on the cut edge
                        yA = zA/tan(θF)
                        length = 2*lz
                        case3=True
                        bounds=bounds+2
                        cond1=False
                else:
                    if b*lz/sin(θF)<l<4*lz/sin(θF):
                        cut_edge_effect = True
                        b = int(l/(lz/sin(θF)))+1
                        yA = l*cos(θF) # Coordinates of the new initial point A'
                        zA = l*sin(θF)
                        xA = xA_i+abs(yA-yA_i)*tan(δ)*sin(θF)/sin(α_abs)
                        length=sqrt((xA-xA_i)**2+(yA-yA_i)**2+(zA-zA_i)**2)
                        case3=True   
                        bounds=bounds+1 
                        α1 = abs(θF-α_abs)
                        condition = True
        
        if condition:
            # conditions of total reflection
            cond0 = cos(θc)>sin(α1)*cos(δ_abs) # plan 0 of the radiator
            cond1 = cos(θc)>sin(δ_abs)  # plan 1 of the radiator
            cond2 = cos(θc)>cos(α1)*cos(δ_abs)  # plan 2 of the light guide
            cond3 = cos(θc)>sin(α_abs)*cos(δ_abs) # Cut edge (plan 3) 
            
        # Determination of the case (1, 2 or 3)
        if (condition and cond0 and cond1):
            yA = yA-lz/tan(θF)*(b-1)
            zA = zA-lz*(b-1)
            δ1 = arctan(tan(δ)/cos(α1)) 
            if (δ1<0):
                δ1_abs = abs(δ1)
                if (xA/tan(δ1_abs)+yA > L_rad[t-1][b-1]-lx[t-1]):
                    case=1
                else:
                    d = (L_rad[t-1][b-1]-lx[t-1]-yA)*tan(δ1)+xA
                    int_d = int(d/lx[t-1])
                    d2 = lx[t-1]*tan(δ1)
                    if (int_d+1)%2==0:
                        if d+d2 > int_d*lx[t-1]-lx[t-1]:
                            case = 1
                            if t==1:
                                d3 = (lx1+lx2/tan(α_t))*tan(δ1)
                                dlim = int_d*lx[t-1]-lx1-lx2
                                if d+d3 < dlim:
                                    taper=True
                        else:
                            case = 2
                            yC = abs(int_d*lx[t-1]-lx[t-1]-xA)/tan(δ1_abs)+yA
                            if t==1:
                                d3 = (lx1+lx2)*tan(δ1)
                                dlim = int_d*lx[t-1]-lx1-lx2/tan(α_t)
                                if d+d3 > dlim:
                                    taper=True                                
                    else:
                        if (δ1_abs<pi/4):
                            case = 1
                        else:
                            case = 0  
                
            else:
                if ((lx[t-1]-xA)/tan(δ1)+yA>L_rad[t-1][b-1]):
                    case=1
                elif ((lx[t-1]-xA)/tan(δ1)+yA>L_rad[t-1][b-1]-lx[t-1]):
                    case=2 
                    if t==1:
                        taper=True
                else:
                    d = (L_rad[t-1][b-1]-lx[t-1]-yA)*tan(δ1)-(lx[t-1]-xA)
                    int_d = int(d/lx[t-1])
                    d2 = lx[t-1]*tan(δ1)
                    if int_d%2==0:
                        if d+d2 < int_d*lx[t-1]+lx[t-1]:
                            case = 1
                            if t==1:
                                d3 = (lx1+lx2/tan(α_t))*tan(δ1)
                                dlim = int_d*lx[t-1]+lx1+lx2
                                if d+d3 > dlim:
                                    taper=True # Photon hits the taper
                        else:
                            case = 2
                            yC = (lx[t-1]-xA+int_d*lx[t-1]+lx[t-1])/tan(δ1)+yA
                            if t==1:
                                d3 = (lx1+lx2)*tan(δ1)
                                dlim = int_d*lx[t-1]+lx1+lx2/tan(α_t)
                                if d+d3 < dlim:
                                    taper=True                      
                    else:
                        if (δ1<pi/4):
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
            θin = arccos(cos(α2)*cos(δ3))
            ϕin = arctan(tan(α2)/tan(δ2))+pi/2
            R0 = ((n_silica-n_air)/(n_silica+n_air))**2
            R = R0+(1-R0)*(1-cos(θin))**5                
            if taper==True: # In the case where the train has taper (train1)
                δ2 = abs(δ2-2*α_t)
                δ3 = abs(δ3-2*arctan(cos(α2)*tan(α_t)))            
            bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(tan(α1)/lz+tan(δ1)/lx[t-1])+L_lg[t-1]*(tan(α2)/lz+tan(δ2)/5)) # Number of bounds on the sides of the detector
            if (case3==False):
                #Calculation of length from vertex point to the photomultiplier
                l = (L_rad[t-1][b-1]-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l) # Probability of absorption
                x1 = rand()
                x2 = rand()
                x3 = rand()
                x4 = rand()
                # Conditions of attenuation
                if (x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R): 
                    L1.append(l)
                    count1+=1
                    case = 1 
                    detected = True # Photon reaches the PMT
                else:
                    case=0
            else:
                l = (L_rad[t-1][b-1]-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                x4 = rand()
                if (cond3 and x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R): 
                    L3.append(l)          
                    if (cut_edge_effect):
                        L3_cut_edge_effect.append(l)
                    count3+=1
                    case = 3
                    detected = True                     
                else:
                    case=0  
                
        if (cond2 and case==2):
            δ2 = pi/2-abs(δ1)
            δ3 = arcsin(cos(δ)*cos(α1))
            α2 = arctan(tan(α1)*tan(δ2))
            θin = arccos(cos(α2)*cos(δ3))
            ϕin = arctan(tan(α2)/tan(δ2))+pi/2
            R0 = ((n_silica-n_air)/(n_silica+n_air))**2
            R = R0+(1-R0)*(1-cos(θin))**5 # Schlick's approximation                  
            if taper==True:
                δ2 = abs(δ2-2*α_t)
                δ3 = abs(δ3-2*arctan(cos(α2)*tan(α_t)))
            bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(tan(α1)/lz+tan(δ1)/lx[t-1])+L_lg[t-1]*(tan(α2)/lz+tan(δ2)/5))
            if (case3==False):
                l = (yC-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                if (cond2 and x1<1-pabs and x2<0.99**bounds and x3>R): 
                    L2.append(l)
                    count2+=1
                    case = 2 
                    detected = True                    
                else:
                    case=0
            else:     
                l = (yC-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                if (cond2 and cond3 and x1<1-pabs and x2<0.99**bounds and x3>R):
                    L3.append(l)           
                    if (cut_edge_effect):
                        L3_cut_edge_effect.append(l)
                    count3+=1
                    case = 3 
                    detected = True 
                else:
                    case=0
        
        if detected:
            x_passed = rand()
            if x_passed<QE:
                count_passed+=1
            Time.append(l/(n_silica*c)*10**9) # Time of trajectory from vertex point to PMT
            Case.append(case) 
        
print("Time execution : %s seconds ---" % (time.time() - start_time))

count = count1+count2+count3
L = L1+L2+L3

print("Mean number of photons generated per event = ", round(N_mean))
print("Total number of photon tracks generated = ", round(count_tot))
print("Number of photons reaching PMT = ", count)        
print("Percentage of photons reaching PMT = {0}%".format(round(count/count_tot*100)))
print("Case 1 = {0}%".format(round(count1/count*100)))
print("Case 2 = {0}%".format(round(count2/count*100)))
print("Case 3 = {0}%".format(round(count3/count*100)))
print("Percentage of photons passed = {0}%".format(round(count_passed/count_tot*100)))
    
figure()
hist(Case, color='white', edgecolor = 'blue')
title('Case - Train 2')
xlabel("Case")
ylabel("Number of tracks")

figure()
hist(L1, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 1 - Train 2')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()

figure()
hist(L2, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 2 - Train 2')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()

figure()
hist(L3, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 3 - Train 2')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()
  
figure()
hist(L, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - Train 2')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()

figure()
hist(Time, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(200,1000), bins=500, label="Fast simulation")
title('Time of photon tracks - Train 2')
xlabel("Time (ns)")
ylabel("Number of tracks")
legend()
