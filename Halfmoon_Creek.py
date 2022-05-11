# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:41:18 2019

@author: enricaviparelli
"""

"""This code models tracer dispersal in an equilibrium bed of uniform sediment"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import csv
    
def main():

    """input parameters"""
 
    exp = 'HC27_fbc'                 #name of the output files
    B = np.float(8)                 #channel width (m)
    L = np.float(1000)              #reach length (m)
    Slope = np.float(0.0086)        #bed slope
    D = np.float(55)                #grain size (mm)
    R = np.float(1.65)              #submerged specific gravity of the sediment 
    g = np.float(9.81)              #acceleration of gravity (m/s2)
    Qs = np.float(160)              #feed rate (t/yr)
    tstar = np.float(0.046)         #Shields number Mueller and Pitlick (2005)
    Durcal = np.float(9)            #duration of the calculations in years
    N = np.int(101)                 #number of computational nodes in the flow direction
    Nver = np.int(2500)             #number of computational nodes in the vertical direction    
    Dt = np.float(0.005)            #temporal increment in days
    Dy = np.float(0.01)             #spatial distance between computational nodes in the vertical direction (m)
    l_nd = np.float(150)            #non-dimensional step length
    sigma = np.float(0.22)          #amplitude of the bed level changes (m)
    cb = np.float(0.65)             # 1 minus porosity
    ftx = np.float(3.5)            #streamwise length of tracer installation (m)
    fty = np.float(0.955)            #transverse length of tracer installation (m)
    timeyear = np.float(31557600)   #number of seconds in one year
    Dtprint = np.float(1)           #time interval between two printouts in years
    
    
    D = D / 1000.               
    Qs = Qs*1000/timeyear           #feed rate (kg/s)
    Gs = Qs/(1000.*(R+1.))          #volumetric feed rate
    qs = Gs/B                       #volumetric feed rate per unit channel width
    Dt = Dt*60*60*24
    
    x = np.zeros(N)                 #streamwise coordinate
    Dx = L/(N-1)                    #spatial distance between computational nodes in the streamwise direction
    tstar_ref = 2.18*Slope+0.021    #Critical Shields number for the Parker (1979) model (Mueller and Pitlick, 2005)
    qtc = 11.2*np.power((tstar-tstar_ref),4.5)*np.power((R*g*D),0.5)*D/np.power(tstar,3.)    #sediment transport capacity at bankfull flow
    If = np.float(0.05)
    E = qtc/(D*l_nd)                 #entrainment rate (m/s)
    no = -28.67*tstar+3.15          #parameter to compute the entrainment offset distance as a function of the standard deviation of bed elevations
    Nin = np.int(30/Dx)+1           #number of computational nodes where tracers are installed
    
    """definition of variables"""
    
    time = np.float(0)              #temporal coordinate in seconds
    tprint = Dtprint                #time of the first printout in years
    iterable = (i for i in range(N))
    x = Dx*np.fromiter(iterable, np.float)    
    iterable = (i for i in range(Nver))
    y = Dy*np.fromiter(iterable, np.float) - Nver*Dy/2 +Dy/2. 
    
    ft = np.zeros((N,Nver))         #volume fraction content of tracers in the deposit
    ft_up = np.zeros((N, Nver))     #volume fraction content of tracers in te deposit at x - lambda
    ft_new= np.zeros((N, Nver))
    fft_up = np.ones(Nver)*0.
    D_x = np.zeros(N)
    Ft = np.zeros(N)
    fsubt = np.zeros((N,4))
    Entrain = np.zeros(Nver)        #entrainment rate
    Vt = np.float(0)
    N1 = np.int(0) 
    N2 = np.int(0) 
    N3 = np.int(0) 
    N4 = np.int(0) 
    Ntop = np.int(0) 
    
    SL = D*l_nd                     #particle step length
    JJump = np.divide(SL, Dx)
    jjump = JJump.astype(int)+2
    B_cond = np.ones((jjump,Nver))
    
    """Definition of layers"""
    
    Ltop = 5.*sigma                
    ytop = np.divide(Ltop,Dy)       
    Ntop = Nver//2 + ytop.astype(int)+1   
    L1 = 1.5 * sigma
    y1 = np.divide(L1,Dy)
    N1 = Nver//2 - (y1.astype(int)+1)    
    L2 = 1.5 * sigma
    y2 = np.divide(L2,Dy)
    N2 = N1 - (y2.astype(int)+1)
    L3 = 1.5*sigma
    y3 = np.divide(L3,Dy)
    N3 = N2 - (y3.astype(int)+1) 
    L4 = 1.5*sigma
    y4 = np.divide(L4,Dy)
    N4 = N3 - (y4.astype(int)+1)
    
    """probability of bed elevation and entrainment"""
    
    yo = no * sigma                 #entrainment offset distance (m)
    bb = (Ltop-0.)/sigma
    Ps = 1- sp.truncnorm.cdf(y, -5000., bb, loc = 0., scale = sigma)
    bb = (Ltop-yo)/sigma
    pent = sp.truncnorm.pdf(y, -5000., bb, loc = yo, scale = sigma)

    """calculation of layer thickness"""
    
    PP = np.sum(Ps[N4: Ntop])*Dy  
    P1 = np.sum(Ps[N1: Ntop])*Dy  
    P2 = np.sum(Ps[N2: N1])*Dy  
    P3 = np.sum(Ps[N3: N2])*Dy  
    P4 = np.sum(Ps[N4: N3])*Dy  

    
    del L, R, g, Gs, tstar, Slope, ytop, y1, y2, y3, y4, bb, qtc, tstar_ref, Qs, iterable, l_nd, 
    
    """tracer intial condition"""
    
    for k in range (N1,Ntop-1,1):
        ft[0:Nin, k] = (ftx/(Nin*Dx))*(fty/B)*(D/(P1*cb))       
    
   
    """calculatioin of the initial concentration of tracers in each layer"""
    
    for i in range(N):    
        Ft[i] = np.sum(ft[i,N4: Ntop]*Ps[N4: Ntop])*Dy /PP
        fsubt[i,0] = np.sum(ft[i,N1: Ntop]*Ps[N1: Ntop])*Dy/P1
        fsubt[i,1] = np.sum(ft[i,N2:N1]*Ps[N2:N1])*Dy /P2  
        fsubt[i,2] = np.sum(ft[i,N3:N2]*Ps[N3:N2])*Dy /P3  
        fsubt[i,3] = np.sum(ft[i,N4:N3]*Ps[N4:N3])*Dy /P4  
    
    """calculatioin of tracer dispersal"""
    
    Vt = qs*Ft[N-1]*Dt*B
    Entrain = E/(cb*Ps)
    ff = np.where(Entrain==np.inf)
    vv = ff[0]
    Entrain[vv] = 0.
    aa= np.isnan(Entrain)
    Entrain[aa]=0.
    
    del ff, vv, aa
    
    print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4, timeyear)    
    
    while (time/timeyear) <= Durcal:
        
        time = time + Dt
        B_cond = B_cond*fft_up  
        funct = np.append(B_cond,ft,axis=0)
        ft_up = funct[0:N,:]*(SL-(jjump-2)*Dx)/Dx+ funct[1:N+1,:]*((jjump-1)*Dx-SL)/Dx
        
        for i in range(N):
            D_x = np.sum(ft_up[i,:] * pent)*Dy*pent
            ft_new[i,:] = ft[i,:] + If*Entrain*(D_x - ft[i,:]*pent)*Dt
            colu = np.where(ft_new[i,:]<0.)
            ft_new[i,colu]=0.
            Ft[i] = np.sum(ft_new[i,N4: Ntop]*Ps[N4: Ntop])*Dy/PP
            fsubt[i,0] = np.sum(ft_new[i,N1: Ntop]*Ps[N1: Ntop])*Dy/P1
            fsubt[i,1] = np.sum(ft_new[i,N2:N1]*Ps[N2:N1])*Dy/P2  
            fsubt[i,2] = np.sum(ft_new[i,N3:N2]*Ps[N3:N2])*Dy/P3  
            fsubt[i,3] = np.sum(ft_new[i,N4:N3]*Ps[N4:N3])*Dy/P4 

        ft = ft_new
        Vt = Vt + qs*Ft[N-1]*Dt*B
        
        if (time/timeyear)>= tprint:
            print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4, timeyear)
            tprint = tprint + Dtprint
    
    print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4, timeyear)
    
    
def print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4, timeyear):
    
    output = np.zeros((N, 6)) 

    filename = exp + '_s_sub_'+ str(round(time/timeyear,1)) +'.csv'
    output[:,0] = x    
    output[:,1] = Ft
    output[:,2:6] = fsubt
    
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab')
        writer.writerows(output)
        
    filename = exp + '_dep_'+ str(round(time/timeyear,1)) +'.csv'
    r_t = Ps[N4:Ntop]*Dy
    t_e = np.cumsum(r_t) + y[N4]
    output1 = np.zeros((Ntop-N4,5))
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab')
        for i in range(N):
            output1[:,4] = ft[i, N4:Ntop]
            output1[:,3] = Ps[N4:Ntop]
            output1[:,2] = t_e
            output1[:,1] = y[N4:Ntop]
            output1[:,0] = np.ones(Ntop-N4)*x[i]
            writer.writerows(output1)  

        
if __name__ == '__main__':

    main()  
