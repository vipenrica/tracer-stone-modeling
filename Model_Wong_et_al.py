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
 
    exp = 'Group04_sub1_5min'       #name of the output files
    B = np.float(0.5)               #flume width (m)
    L = np.float(15.25)             #flume length (m)
    D = np.float(7.2)               #grain size (mm)
    D = D / 1000.   
    Qw = np.float(0.093)            #flow rate (m3/s)
    Qs = np.float(0.150)            #feed rate (kg/s)
    R = np.float(1.55)              #submerged specific gravity of the sediment 
    g = np.float(9.81)              #acceleration of gravity (m/s2)
    Durcal = np.float(300)          #duration of the calculations in seconds
    no = np.float(0.5)             #parameter to compute the entrainment offset distance as a function of the standard deviation of bed elevations
    
    
    N = np.int(50)                  #number of computational nodes in the flow direction
    Nver = np.int(500)              #number of computational nodes in the vertical direction    
    Dt = np.float(0.1)              #temporal increment seconds
    
    l_nd = np.float(150)            #non-dimensional step length
    cb = np.float(0.65)             # 1 minus porosity
    ar = np.float(8.1)              #corefficient of the Manning-Strickler relation
    ks = 2.0*D                      #roughness height
    
    x = np.zeros(N)                 #streamwise coordinate
    Dx = L/(N-1)                    #spatial distance between computational nodes in the streamwise direction
    Dy = np.float(0.001)            #spatial distance between computational nodes in the vertical direction (m)
    
    ftx = np.float(0.1)             #streamwise length of tracer installation (m)
    fty = np.float(0.1)             #transverse length of tracer installation (m)
    
    Seed_Layer = np.int(3)		    #bed layer where tracers are installed. Can be equal to 0 (surface), 1, 2 or 3
    
    """definition of variables"""
    
    time = np.float(0)              #temporal coordinate in seconds
    iterable = (i for i in range(N))
    x = Dx*np.fromiter(iterable, np.float)    
    iterable = (i for i in range(Nver))
    y = Dy*np.fromiter(iterable, np.float) - Nver*Dy/2 +Dy/2. 
    
    ft = np.zeros((N,Nver))         #volume fraction content of tracers in the deposit
    ft_up = np.zeros((N, Nver))     #volume fraction content of tracers in the deposit at x - lambda
    ft_new= np.zeros((N, Nver))
    fft_up = np.ones(Nver)*0.
    D_x = np.zeros(N)
    Ft = np.zeros(N)
    fsubt = np.zeros((N,3))
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
    
    """calculation of the flow conditions"""
    
    qw = Qw/B                       #flow rate per unit channel width
    Gs = Qs/(1000.*(R+1.))          #volumetric feed rate
    qs = Gs/B                       #volumetric feed rate per unit channel width
    qs_star = qs/(np.sqrt(R*g*D)*D) #Einstein number
    E = qs/(D*l_nd)                 #entrainment rate (m/s)
    taustar = np.power((qs_star/2.66),2./3.)+0.0549     #Shields number computed as in Wong et al. 2007 relation
    S = np.power((R*D*taustar*np.power(np.power(ar,2)*g/(np.power(ks,1./3.)*np.power(qw,2.)),3./10.)),10./7.)   #bed slope
    
    sigma = D * 3.09 * np.power((taustar - 0.0549),0.56)    #standard deviation of bed level changes computed as in Wong et al. 2007 

    
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
    
    #PP = np.sum(Ps[N4: Ntop])*Dy  
    P1 = np.sum(Ps[N1: Ntop])*Dy  
    P2 = np.sum(Ps[N2: N1])*Dy  
    P3 = np.sum(Ps[N3: N2])*Dy  
    P4 = np.sum(Ps[N4: N3])*Dy  
    
    #PP_sub = np.sum(Ps[N4:N1])*Dy 
    
    del L, R, g, ar, ks, qw, Gs, taustar, S, ytop, y1, y2, y3, y4, bb
    
    """tracer initial condition"""
    
    if Seed_Layer == 0:
        ft[0, N1: Ntop] = (ftx/Dx)*(fty/B)
    elif Seed_Layer == 1:
        ft[0, N2: N1] = (ftx/Dx)*(fty/B)
    elif Seed_Layer == 2:
        ft[0, N3: N2] = (ftx/Dx)*(fty/B)
    else:
        ft[0, N4: N3] = (ftx/Dx)*(fty/B)
    
    """calculatioin of the initial concentration of tracers in each layer"""
    
    for i in range(N):    
        Ft[i] = np.sum(ft[i,N1: Ntop]*Ps[N1: Ntop])*Dy /P1
        fsubt[i,0] = np.sum(ft[i,N2:N1]*Ps[N2:N1])*Dy /P1  
        fsubt[i,1] = np.sum(ft[i,N3:N2]*Ps[N3:N2])*Dy /P2  
        fsubt[i,2] = np.sum(ft[i,N4:N3]*Ps[N4:N3])*Dy /P3  
    
    """calculatioin of tracer dispersal"""
    
    Vt = qs*Ft[N-1]*Dt*B
    Entrain = E/(cb*Ps)
    ff = np.where(Entrain==np.inf)
    vv = ff[0]
    Entrain[vv] = 0.
    aa= np.isnan(Entrain)
    Entrain[aa]=0.
    
    del ff, vv, aa
    
    print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4)    
    
    while time <= Durcal:
        
        time = time + Dt
        B_cond = B_cond*fft_up  
        funct = np.append(B_cond,ft,axis=0)
        ft_up = funct[0:N,:]*(SL-(jjump-2)*Dx)/Dx+ funct[1:N+1,:]*((jjump-1)*Dx-SL)/Dx
        
        for i in range(N):
            D_x = np.sum(ft_up[i,:] * pent)*Dy*pent
            ft_new[i,:] = ft[i,:] + Entrain*(D_x - ft[i,:]*pent)*Dt
            colu = np.where(ft_new[i,:]<0.)
            ft_new[i,colu]=0.
            colu1 = np.where(ft_new[i,:]>1.)
            ft_new[i,colu1]=1.
            Ft[i] = np.sum(ft_new[i,N1: Ntop]*Ps[N1: Ntop])*Dy/P1
            fsubt[i,0] = np.sum(ft_new[i,N2:N1]*Ps[N2:N1])*Dy/P2  
            fsubt[i,1] = np.sum(ft_new[i,N3:N2]*Ps[N3:N2])*Dy/P3  
            fsubt[i,2] = np.sum(ft_new[i,N4:N3]*Ps[N4:N3])*Dy/P4 

        ft = ft_new
        Vt = Vt + qs*Ft[N-1]*Dt*B
        
    print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4)
    
    
    
    
def print_tracers(exp, x, Ft, fsubt, ft, Ps, y, time, N, Dy, Ntop, N4):
    
    output = np.zeros((N, 5)) 

    filename = exp + '_s_sub_'+ str(round(time/60,1)) +'.csv'
    output[:,0] = x    
    output[:,1] = Ft
    output[:,2:5] = fsubt
    
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab')
        writer.writerows(output)
        
    filename = exp + '_dep_'+ str(round(time/60,1)) +'.csv'
    r_t = Ps[N4:Ntop]*Dy
    t_e = np.cumsum(r_t) + y[N4]
    output1 = np.zeros((Ntop-N4,3))
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab')
        for i in range(N):
            output1[:,2] = ft[i, N4:Ntop]
            output1[:,1] = t_e
#            output1[:,1] = y[Nsub:Ntop]
            output1[:,0] = np.ones(Ntop-N4)*x[i]
            writer.writerows(output1)  

        
if __name__ == '__main__':

    main()  
