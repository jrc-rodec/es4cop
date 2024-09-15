# -*- coding: utf-8 -*-

import numpy as np

# bound constraint handling by reflecting back into the feasible box
def keep_range(y,lower_bounds,upper_bounds):
    bwidth =  upper_bounds -  lower_bounds
    n=np.size(y);
    for i in range(n):
        if y[i]<lower_bounds[i]:
            exceed = lower_bounds[i]-y[i]
            if exceed >= bwidth[i]:
            	exceed = exceed - np.floor(exceed/bwidth[i])*bwidth[i]
            y[i] = lower_bounds[i] + exceed
        if y[i]>upper_bounds[i]:
            exceed = y[i]-upper_bounds[i]
            if exceed >= bwidth[i]:
                exceed = exceed - np.floor(exceed/bwidth[i])*bwidth[i]
            y[i] = upper_bounds[i] - exceed
    return y

#####
# constraint relaxation by epsilon-level ranking

# subroutine: 
def eps_rank(f1,cv1,f2,cv2,epsilon):
    if cv1 == cv2:
        z=(f1 <= f2)
    elif (cv1 <= epsilon) and (cv2<= epsilon):
        z=(f1 <= f2)
    else:
        z=(cv1 < cv2)
    return z

# main routine:
def eps_sort(fit,cvio,epsilon):
    n = np.size(fit)
    ind = np.linspace(0,n-1,n)
    for k in range(n-1):
        i = n-1-k
        for j in range(i):
            if eps_rank(fit[j],cvio[j],fit[j+1],cvio[j+1],epsilon)==0:
                k=ind[j]
                f=fit[j]
                c=cvio[j]
                ind[j]=ind[j+1]
                fit[j]=fit[j+1]
                cvio[j]=cvio[j+1]
                ind[j+1]=k
                fit[j+1]=f
                cvio[j+1]=c
    ranking = np.int32(ind)
    return ranking

#####
# gradient-based repair

def gradientMutation(y,gg,hh):
    # preset parameters of forward differences approach for Jacobian approximation
    eta     = 1e-4;                            
    dim  = np.size(y)
    # create vector of constraint-wise constraint violations
    Cx = np.hstack((gg, hh))
    nn=np.size(Cx)
    deltaG  = np.max([gg*0,gg],0)
    
    dCx  = np.zeros((nn,dim))
    Dd   = np.eye(dim)
    dx   = np.transpose(np.tile(y,(dim,1)))+eta*Dd
    
    for i in range(dim):
        [f, gv, hv] = const_fun01(dx[:,i])  # LOOK UP function evaluation
        dg = np.hstack((gv, hv))
        dCx[:,i] = dg
        
    # approaximate Jacobian        
    nabC    = 1/eta*(dCx - np.transpose(np.tile(Cx,(dim,1))))
    delC    = np.hstack((deltaG,hh))
    
    # compute Moore-Penrose inverse of nabC
    inv_nabC= np.linalg.pinv(nabC,1e-12) # hard coded parameter!!!
    deltaX  = -np.matmul(inv_nabC,delC)
    
    # repair the infeasible candidate solution x
    y       = (y+deltaX);
    return y