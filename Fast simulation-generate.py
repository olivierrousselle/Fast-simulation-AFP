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
start_time = time.time()

lx = [3, 5, 5, 5] # Length of lx (in mm) for each train
lx1=3 # For the train 1 with taper
lx2=2 
lz = 6 # Length of lx (in mm)
θF = 48*2*pi/360 # Construction angle in rad
α_t = 18*2*pi/360 # Angle of taper in rad
angle = 0 # Angle between photon beam and cut edge

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

θ_ch = []
file = open("θ_ch.txt","r") # Cherenkov angle distribution
for line in file:
    θ_ch.append(float(line))
file.close()

count = 0 # Counter of number of photons detected 
count1 = 0 # Counter of number of photons detected per case
count2 = 0
count3 = 0
N = 100000 # Number of tracks simulated

t = 2 # Train (1, 2, 3 or 4) 
 
# Computation for each photon track
if (N!=0):
    for r in range(N):
        xA_i = lx[t-1]/2 # Position of vertex points
        zA_i = rand()*4*lz
        yA_i = tan(pi/2-θF)*zA_i+2.0
        StepLength = zA_i/sin(θF)  # length between the side of radiator and vertex point A      
        θch=θ_ch[int(rand()*len(θ_ch))] # Cherenkov angle
        θch2.append(θch)
        λ = (60.57+sqrt(7004/(1/cos(θch)-1.449)))/1.614 # Wavelenght of photon
        λ2.append(λ)
        n_silica = 1/cos(θch) # refraction index
        θc = arcsin(n_air/n_silica) # critical angle
        ϕ = uniform(-pi,pi)
        δ = arcsin(sin(θch)*cos(ϕ)) # Projection angles
        α = arcsin(tan(δ)*tan(ϕ))+angle
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
                case3 = True
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
                        yA = tan(pi/2-θF)*zA+2
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
                    if int_d%2==0:
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
                                    taper=True
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
                
        Labs = 450
        µabs = 1/Labs # Attenuation coefficient
        
        # length of photon trajectory
        if case==1:
            δ2 = abs(δ1)
            δ3 = abs(δ)
            α2 = α1       
            if taper==True: # In the case where the train has taper (train1)
                δ2 = abs(δ2-2*α_t)
                δ3 = abs(δ3-2*arctan(cos(α2)*tan(α_t)))            
            bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(tan(α1)/lz+tan(δ1)/lx[t-1])+L_lg[t-1]*(tan(α2)/lz+tan(δ2)/5)) # Number of bounds on the sides of the detector
            if (case3==False):
                #Calculation of length from vertex point to the photomultiplier
                l1 = (L_rad[t-1][b-1]-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l1) # Prabability of absorption
                x1 = rand()
                x2 = rand()
                x3 = rand()
                # Conditions of attenuation
                if (x1<0.9 and x2<1-pabs and x3<0.99**bounds): 
                    L1.append(l1)
                    count1 = count1+1
                    case = 1                       
                else:
                    case=0
            else:
                l3 = (L_rad[t-1][b-1]-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l3)
                x1 = rand()
                x2 = rand()
                x3 = rand()
                if (cond3 and x1<0.9 and x2<1-pabs and x3<0.99**bounds): 
                    L3.append(l3)          
                    if (cut_edge_effect):
                        L3_cut_edge_effect.append(l3)
                    count3=count3+1
                    case = 3               
                else:
                    case=0  
                
        if (cond2 and case==2):
            δ2 = pi/2-abs(δ1)
            δ3 = arcsin(cos(δ)*cos(α1))
            α2 = arctan(tan(α1)*tan(δ2))
            if taper==True:
                δ2 = abs(δ2-2*α_t)
                δ3 = abs(δ3-2*arctan(cos(α2)*tan(α_t)))
            bounds = bounds + int((L_rad[t-1][b-1]-yA-lx[t-1])*(tan(α1)/lz+tan(δ1)/lx[t-1])+L_lg[t-1]*(tan(α2)/lz+tan(δ2)/5))
            if (case3==False):
                l2 = (yC-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))
                pabs = 1-exp(-µabs*l2)
                x1 = rand()
                x2 = rand()
                if (cond2 and x1<1-pabs and x2<0.99**bounds): 
                    L2.append(l2)
                    count2 = count2+1
                    case = 2                      
                else:
                    case=0
            else:     
                l3 = (yC-yA)/(cos(δ)*cos(α1))+L_lg[t-1]/(cos(α2)*cos(δ3))+length
                pabs = 1-exp(-µabs*l3)
                x1 = rand()
                x2 = rand()
                if (cond2 and cond3 and x1<1-pabs and x2<0.99**bounds):
                    L3.append(l3)           
                    if (cut_edge_effect):
                        L3_cut_edge_effect.append(l3)
                    count3=count3+1
                    case = 3          
                else:
                    case=0
    
        Case.append(case) 
        
print("Time execution : %s seconds ---" % (time.time() - start_time))

count = count1+count2+count3 # Number of tracks detected in the PMT

# Histograms of fast simulation and Geant4 lengths 
figure()
hist(L1, color='white', edgecolor = 'blue', range=(90,400), bins=500)
title('Case 1')
xlabel("Length (mm)")
ylabel("Number of tracks")

figure()
hist(L2, color='white', edgecolor = 'blue', range=(90,400), bins=500)
title('Case 2')
xlabel("Length (mm)")
ylabel("Number of tracks")

figure()
hist(L3, color='white', edgecolor = 'blue', range=(90,400), bins=500)
title('Case 3')
xlabel("Length (mm)")
ylabel("Number of tracks")

figure()
hist(L3_cut_edge_effect, color='white', edgecolor = 'blue', range=(90,400), bins=500)
title('Case 3 - cut edge effect')
xlabel("Length (mm)")
ylabel("Number of tracks")

figure()
hist(L1+L2+L3, color='white', edgecolor = 'blue', range=(90,400), bins=500)
title('All cases')
xlabel("Length (mm)")
ylabel("Number of tracks")