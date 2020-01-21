# -*- coding: utf-8 -*-

import time
from numpy.random import *
from numpy import *
from matplotlib.pyplot import *
from math import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.constants import *
from scipy import integrate

# Example for bar 2A
lx = 5 
lz = 6
L_rad = 59.15 # Length of radiator (mm)
L_lg = 65.2 # Length of light guid (mm)
n_air = 1
L1 = [] 
L2 = []
L3 = []
Time = [] # Time from vertex to PMT
θF = 48*2*pi/360 # Construction angle
angleB = 0*2*pi/360 # Angle of proton beam
yB = 2 # y-intercept of proton beam (mm)

count_tot = 0 # Number of photons generated
count = 0 # Number of photons detected 
count1 = 0 # Number of photons detected per case
count2 = 0
count3 = 0
count_passed = 0
count_error = 0
Np = 59 # Number of proton beams
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
Steplength = lz/sin(θF-angleB)*0.1 # in cm
N_mean = 370*Z**2*Steplength*(1-1/(n_mean**2*β**2))*(Emax-Emin)
M_E = linspace(Emin,Emax,10**6)
f = [density(E) for E in M_E] # Density distribution for energy
f = f/sum(f)
mat = M_E[random.choice(len(M_E),10**6,p=list(f))]/(6.242*10**18)

start_time = time.time()

for n in range(Np):
    N = poisson(N_mean) # Number of photon tracks per event
    for r in range(N):
        count_tot+=1
        xA_i = lx/2 # Coordinates of vertex point A
        zA_i = rand()*lz
        yA_i = zA_i/tan(θF-angleB)+yB      
        StepLength = zA_i/sin(θF-angleB) # length between the side of radiator and vertex point A
        E = mat[int(rand()*len(mat))] # Energy of photon
        λ = h*c/E*10**9 # Wavelength of photon (nm) 
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
        case3 = False
        length = 0
        bounds = 0
        detected = False
        
        if (α>=0): 
            case3 = False
            xA=xA_i
            yA=yA_i
            zA=zA_i
            α1=abs(θF-α)
        else:
            l=StepLength+(yA_i-zA_i/tan(θF))*(cos(θF)+sin(θF)/tan(α_abs))  
            if (α<0 and l>lz/sin(θF) and α_abs<pi/2-θF):
                 case3 = True # Photon hits the cut edge
                 xA=xA_i
                 yA=yA_i
                 zA=zA_i
                 α=α_abs
                 α1=θF+α
            elif (α<0 and l<lz/sin(θF)):
                yA = l*cos(θF) # Coordinates of the new initial point A'
                zA = l*sin(θF)
                xA = xA_i+abs(yA-yA_i)*tan(δ)*sin(θF)/sin(α_abs)
                length=sqrt((xA-xA_i)**2+(yA-yA_i)**2+(zA-zA_i)**2) # length [AA'] to add at the end
                case3 = True
                α1=abs(θF-α_abs)
                bounds=bounds+1

            elif (α<0 and l>lz/sin(θF) and α_abs>pi/2-θF):
                xA = xA_i
                zA = rand()*lz # We choose a random point on the cut edge
                yA = zA/tan(θF)
                length = 2*lz
                case3=True
                α1=α_abs+θF
                bounds=bounds+2
        
        # conditions of total reflection
        cond0 = cos(θc)>sin(α1)*cos(δ_abs) # plan 0 of the radiator
        cond1 = cos(θc)>sin(δ_abs)  # plan 1 of the radiator
        cond2 = cos(θc)>cos(α1)*cos(δ_abs)  # plan 2 of the light guide
        cond3 = cos(θc)>sin(α_abs)*cos(δ_abs) # Cut edge (plan 3)

        case=0 
        if (cond0 and cond1):
            δ1 = arctan(tan(δ)/cos(α1)) 
            if (δ1<0):
                δ1_abs = abs(δ1)
                if (xA/tan(δ1_abs)+yA > L_rad-lx):
                    case=1
                else:
                    d = (L_rad-lx-yA)*tan(δ1)+xA
                    int_d = int(d/lx)
                    d2 = lx*tan(δ1)
                    if (int_d+1)%2==0:
                        if (d+d2 > int_d*lx-lx):
                            case = 1
                        else:
                            case = 2
                            yC = abs(int_d*lx-lx-xA)/tan(δ1_abs)+yA
                    else:
                        if (δ1_abs<pi/4):
                            case = 1
                        else:
                            case = 0
                
            else:
                if ((lx-xA)/tan(δ1)+yA>L_rad):
                    case=1
                elif ((lx-xA)/tan(δ1)+yA>L_rad-lx):
                    case=2 
                else:
                    d = (L_rad-lx-yA)*tan(δ1)-(lx-xA)
                    int_d = int(d/lx)
                    d2 = lx*tan(δ1)
                    if int_d%2==0:
                        if d+d2 < int_d*lx+lx:
                            case = 1
                        else:
                            case = 2
                            yC = (lx-xA+int_d*lx+lx)/tan(δ1)+yA                      
                    else:
                        if (δ1<pi/4):
                            case = 1
                        else:
                            case = 0  
        
        Labs = 121.66+4.65441*λ-0.00606166*λ**2+2.60047e-06*λ**3 # Attenuaton length
        µabs = 1/Labs # Attenuation coefficient
        QE = 0.2 # Quantum reflectivity 
                
        if (cond0 and cond1 and case==1):
            δ2 = abs(δ1)
            δ3 = abs(δ)
            α2 = α1
            θin = arccos(cos(α2)*cos(δ3))
            ϕin = arctan(tan(α2)/tan(δ2))+pi/2
            R0 = ((n_silica-n_air)/(n_silica+n_air))**2
            R = R0+(1-R0)*(1-cos(θin))**5
            bounds = bounds + int((L_rad-yA-lx)*(tan(α1)/lz+tan(δ1)/lx)+L_lg*(tan(α2)/lz+tan(δ2)/lx)) # Number of bounds on the sides of the detector
            if (case3==False): 
                # Calculation of length from vertex point to the photomultiplier
                l = (L_rad-yA)/(cos(δ)*cos(α1))+L_lg/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l) # Probability of absorption
                # Conditions of reflection and attenuation
                x1 = rand()
                x2 = rand()
                x3 = rand()
                x4 = rand()
                if (x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R): 
                    L1.append(l)
                    count1+=1 
                    case = 1
                    detected = True # Photon reaches the PMT
                else:
                    case=0   
            else:
                l = (L_rad-yA)/(cos(δ)*cos(α1))+L_lg/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l)                
                x1 = rand()
                x2 = rand()
                x3 = rand()
                x4 = rand()
                if (cond3 and x1<0.9 and x2<1-pabs and x3<0.99**bounds and x4>R):
                    L3.append(l)             
                    count3+=1
                    case = 3
                    detected = True                   
                else:
                    case=0  
                
        if (cond0 and cond1 and cond2 and case==2):
            δ2 = pi/2-abs(δ1)
            δ3 = arcsin(cos(δ)*cos(α1))
            α2 = arctan(tan(α1)*tan(δ2))
            θin = arccos(cos(α2)*cos(δ3))
            ϕin = arctan(tan(α2)/tan(δ2))+pi/2
            R0 = ((n_silica-n_air)/(n_silica+n_air))**2
            R = R0+(1-R0)*(1-cos(θin))**5  # Schlick's approximation  
            bounds = bounds + int((L_rad-yA-lx)*(tan(α1)/lz+tan(δ1)/lx)+L_lg*(tan(α2)/lz+tan(δ2)/lx))
            if (case3==False):
                l = (yC-yA)/(cos(δ)*cos(α1))+L_lg/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                # Conditions of attenuation
                if (x1<1-pabs and x2<0.99**bounds and x3>R):
                    L2.append(l)
                    count2+=1
                    case = 2
                    detected = True
                else:
                    case=0

            else:     
                l = (yC-yA)/(cos(δ)*cos(α1))+L_lg/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                if (cond3 and x1<1-pabs and x2<0.99**bounds and x3>R): 
                    L3.append(l)
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

print("Time execution : %s seconds ---" % (time.time() - start_time))

count = count1+count2+count3 # Number of tracks detected in the PMT
L = L1+L2+L3

print("Mean number of photons generated per event = ", round(N_mean))
print("Total number of photon tracks generated = ", round(count_tot))
print("Number of photons reaching PMT = ", count)        
print("Percentage of photons reaching PMT = {0}%".format(round(count/count_tot*100)))
print("Case 1 : {0}%".format(round(count1/count*100)))
print("Case 2 : {0}%".format(round(count2/count*100)))
print("Case 3 : {0}%".format(round(count3/count*100)))
print("Percentage of photons passed = {0}%".format(round(count_passed/count_tot*100)))

figure()
hist(L1, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 1 - Bar 2A')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()
figure()
hist(L2, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 2 - Bar 2A')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()
figure()
hist(L3, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - case 3 - Bar 2A')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()  
figure()
hist(L, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(90,400), bins=500, label="Fast simulation")
title('Length of photon tracks - Bar 2A')
xlabel("Length (mm)")
ylabel("Number of tracks")
legend()
figure()
hist(Time, histtype='stepfilled', color='white', edgecolor = 'blue', alpha=0.8, range=(200,1000), bins=500, label="Fast simulation")
title('Time of photon tracks - Bar 2A')
xlabel("Time (ns)")
ylabel("Number of tracks")
legend()
