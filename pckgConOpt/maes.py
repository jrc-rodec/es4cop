# -*- coding: utf-8 -*-

import numpy as np

# bound constraint handling by reflecting back into the feasible box
def keep_range(y,lower_bounds,upper_bounds):
    bwidth =  upper_bounds -  lower_bounds
    n=np.size(y)
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
def eps_sort(ffv,ccv,epsilon):
    fit=ffv.copy()
    cvio=ccv.copy()
    n = np.size(fit)
    ind = np.linspace(0,n-1,n)
    for k in range(n-1):
        i = n-1-k
        for j in range(i):
            if eps_rank(fit[j],cvio[j],fit[j+1],cvio[j+1],epsilon)==0:
                ph1=ind[j]
                ph2=fit[j]
                ph3=cvio[j]
                ind[j]=ind[j+1]
                fit[j]=fit[j+1]
                cvio[j]=cvio[j+1]
                ind[j+1]=ph1
                fit[j+1]=ph2
                cvio[j+1]=ph3
    ranking = np.int32(ind)
    return ranking

#####
# gradient-based repair

def gradientMutation(y,gg,hh,objFun):
    # preset parameters of forward differences approach for Jacobian approximation
    eta  = 1e-4;         # learning rate for Jacobian approx.                   
    dim  = np.size(y)
    # create vector of constraint-wise constraint violations
    Cx = np.hstack((gg, hh))
    nn=np.size(Cx)
    deltaG  = np.max([gg*0,gg],0)
    
    dCx  = np.zeros((nn,dim))
    Dd   = np.eye(dim)
    dx   = np.transpose(np.tile(y,(dim,1)))+eta*Dd
    
    for i in range(dim):
        [f, gv, hv] = objFun(dx[:,i])  # LOOK UP function evaluation
        dg = np.hstack((gv, hv))
        dCx[:,i] = dg
        
    # approaximate Jacobian        
    nabC    = 1/eta*(dCx - np.transpose(np.tile(Cx,(dim,1))))
    delC    = np.hstack((deltaG,hh))
    
    # compute Moore-Penrose inverse of nabC
    inv_nabC= np.linalg.pinv(nabC,1e-12) # hard coded parameter!!!
    #deltaX  = -np.matmul(inv_nabC,delC)
    deltaX  = -inv_nabC@delC
    
    # repair the infeasible candidate solution x
    y       = (y+deltaX)
    return y


#### MAES

def MAES(yInit,mu,lam,sigma,maxIter,budget,objFun,lower_bounds,upper_bounds,bch):
# Implementation of the MA-ES by
# H. -G. Beyer and B. Sendhoff, "Simplify Your Covariance Matrix Adaptation Evolution Strategy," 
    # in IEEE Transactions on Evolutionary Computation, vol. 21, no. 5, pp. 746-759, Oct. 2017,
    # doi: 10.1109/TEVC.2017.2680320.
    
    (dim,) = np.shape(yInit)
    newpop_y = np.zeros((dim,lam))  # initialize new population matrix 
    newpop_f = np.zeros((lam))
    newpop_cv = np.zeros((lam))
    fevals = 0
    g = 0
    termination = False
    # Initialize MAES standard strategy parameters and constants
    ps = np.zeros(dim)           # evolution paths for sigma
    M  = np.eye(dim)               # initial transformation matrix
    weights = np.log(mu+1/2)*np.ones(mu)-np.log(np.linspace(1,mu,mu)) # array for weighted recombination
    weights = weights/np.sum(weights)    # normalize recombination weights array
    mueff=1/np.sum(weights**2)           # variance-effectiveness of sum w_i x_i
          
    cs = (mueff+2) / (dim+mueff+5)  # learning rate for cumulation for sigma control
    sqrt_s = np.sqrt(cs*(2-cs)*mueff)  # factor in path update 
    c1 = 2 / ((dim+1.3)**2+mueff)   # learning rate for rank-one update of M
    cmu = np.min([1-c1, 2*(mueff-2+1/mueff) / ((dim+2)**2+mueff)]) # and for rank-mu update of M
    # damps = 1 + 2*np.max(0, np.sqrt((mueff-1)/(dim+1))-1) + cs # damping for sigma usually close to 1
    
    # Initialize random population of lambda individuals 
    [fv, gv, hv] = objFun(yInit)
    # fv = sphere(yInit)
    #[fv, gv, hv]  = const_fun01(yInit)
    cv  = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>10**(-4)))])
    
    fevals = fevals + 1
    
    lb_fv  = fv				# best fitness value of current population
    lb_y   = yInit
    lb_cv  = cv

    gb_fv   = lb_fv		# global best fitness observed over all populations
    gb_y    = lb_y 
    gb_cv  = lb_cv 

    # Capture algorithm dynamics
    dyn_gen        = []
    dyn_fev        = []
    dyn_fit        = []
    dyn_cv         = []
    dyn_sig        = []
    dyn_ynorm      = []
    dyn_gen.append(0)
    dyn_fev.append(fevals)
    dyn_fit.append(lb_fv)
    dyn_cv.append(lb_cv)
    dyn_sig.append(sigma)
    dyn_ynorm.append(np.linalg.norm(lb_y))

    yParent = yInit
     
    while not termination:
        
        # create new generation of offspring candidate solutions
        newpop_z = np.random.randn(dim,lam)
        newpop_d = np.matmul(M,newpop_z)
        newpop_y = np.transpose(np.tile(yParent,(lam,1)))+sigma*newpop_d        
        for k in range(lam):
            newpop_y[:,k] = bch(newpop_y[:,k], lower_bounds, upper_bounds)
            newpop_f[k], gv, hv = objFun(newpop_y[:,k])
            newpop_cv[k] = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>10**(-4)))])
        ranking = np.argsort(newpop_f)
        fevals = fevals + lam

        best_ind = ranking[0]
        lb_fv = newpop_f[best_ind]             # best feasible fitness value of current population
        lb_y = newpop_y[:,best_ind]           # best feasible individual of current population
        lb_cv = newpop_cv[best_ind]

        # Sort by fitness and compute weighted mean into xmean
        parent_z = np.matmul(newpop_z[:,ranking[0:mu]], weights)  # recombination
        parent_d = np.matmul(newpop_d[:,ranking[0:mu]], weights)  # recombination
        yParent  = yParent + sigma*parent_d # update population certroid 

        fParent, gv, hv = objFun(yParent)
        cvParent = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>10**(-4)))])
        fevals = fevals + 1

        # Cumulation: Update evolution paths
        ps = (1-cs) * ps + sqrt_s * parent_z

        # Update transformation matrix 
        M = (1 - 0.5*c1 - 0.5*cmu) * M + (0.5*c1)*np.matmul(M,np.outer(np.transpose(ps),ps))
        for m in range(mu):
            M = M + (0.5*cmu*weights[m])*np.outer(np.transpose(newpop_d[:,ranking[m]]),newpop_z[:,ranking[m]])
                
        # Adapt mutation strength sigma
        sigma = sigma * np.exp((cs/2)*(np.linalg.norm(ps)**2/dim - 1))
        
        if lb_fv <= gb_fv:
            gb_fv  = lb_fv		# global best fitness observed over all populations
            gb_y   = lb_y 
            gb_cv  = lb_cv 
                        
        # termination criteria
        if g >= maxIter:
            termination = True;
        elif fevals >= budget:
            termination = True;
        
        # update generation counter
        g=g+1 
        
        # logging the algorithm dynamics with respect to the global best observation
        dyn_gen.append(g)
        dyn_fev.append(fevals)
        dyn_fit.append(gb_fv)
        #dyn_cv.append(lb_cv)
        dyn_cv.append(gb_cv)
        dyn_sig.append(sigma)
        dyn_ynorm.append(np.linalg.norm(gb_y))
                         
    return gb_y, gb_fv, dyn_gen, dyn_fev, dyn_fit, dyn_cv,  dyn_sig, dyn_ynorm


################# epsMAg-ES #################################################################

def epsMAgES(mu,lam,sigma,lower_bounds,upper_bounds,delta,maxIter,budget,max_reps,objFun):
# Implementation of the epsilonMAg-ES for constrained optimiazation
# according to
# M. Hellwig and H.-G. Beyer, "A Matrix Adaptation Evolution Strategy for Constrained
# Real-Parameter Optimization", 2018 IEEE Congress on Evolutionary
# Computation (CEC), IEEE, 2018, https://dx.doi.org/10.1109/CEC.2018.8477950
    dim = np.size(lower_bounds)
    newpop_y = np.zeros((dim,lam))  # initialize new population matrix 
    newpop_f = np.zeros((lam))
    newpop_cv = np.zeros((lam))
    fevals = 0
    g = 0
    termination = False
    # Initialize MAES standard strategy parameters and constants
    ps = np.zeros(dim)           # evolution paths for sigma
    M  = np.eye(dim)               # initial transformation matrix
    weights = np.log(mu+1/2)*np.ones(mu)-np.log(np.linspace(1,mu,mu)) # array for weighted recombination
    weights = weights/np.sum(weights)    # normalize recombination weights array
    mueff=1/np.sum(weights**2)           # variance-effectiveness of sum w_i x_i
          
    cs = (mueff+2) / (dim+mueff+5)  # learning rate for cumulation for sigma control
    sqrt_s = np.sqrt(cs*(2-cs)*mueff)  # factor in path update 
    c1 = 2 / ((dim+1.3)**2+mueff)   # learning rate for rank-one update of M
    cmu = np.min([1-c1, 2*(mueff-2+1/mueff) / ((dim+2)**2+mueff)]) # and for rank-mu update of M
    # damps = 1 + 2*np.max(0, np.sqrt((mueff-1)/(dim+1))-1) + cs # damping for sigma usually close to 1
    # upper limit of admissilbe mutation strength values
    sigmax = np.max((upper_bounds-lower_bounds)/2)  # NEW: set sigmax to half the maximum boundary width, formerly this parameter was simply set to 100.
    
    # Initialize random population of lambda individuals 
    for k in range(lam):
        # create initial population of uniformly distributed vectors within box-constraints 
        newpop_y[:,k]   = lower_bounds+(upper_bounds-lower_bounds)*np.random.rand(dim) 
        # evaluate initial population
        [fv, gv, hv]  = objFun(newpop_y[:,k])
        newpop_f[k] = fv
        # calculate constraint violation 
        newpop_cv[k]  = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>delta))])
        # count constrained function evaluations
        fevals = fevals + 1
       
    # initial parameters of the epsilon constraint handling approach
    TC = 1000
    nn = int(np.ceil(0.9*lam))
    ind = eps_sort(newpop_f,newpop_cv,0) # initial sorting w.r.t. epsilon-level zero <-- lexicographical ordering 
    EPSILON = np.mean(newpop_cv[ind[1:nn]])
    Epsilon= EPSILON   
    #CP=3 #np.max([3,(-5-np.log(EPSILON))/np.log(0.05)])
    CP = np.max([3,(-5-np.log(EPSILON))/np.log(0.05)])
        
    ranking    = eps_sort(newpop_f,newpop_cv,Epsilon) # now perform epsilon Ranking
    ParentPop  = newpop_y[:,ranking[1:mu]]
    yParent    = np.sum(ParentPop,1)/mu
    
    best_ind   = ranking[1]
    lb_fv   = newpop_f[best_ind]				# best fitness value of current population
    lb_y     = newpop_y[:,best_ind]
    lb_cv  = newpop_cv[best_ind]

    gb_fv  = lb_fv		# global best fitness observed over all populations
    gb_y   = lb_y 
    gb_cv  = lb_cv 

    # Capture algorithm dynamics
    dyn_gen        = []
    dyn_fev        = []
    dyn_fit        = []
    dyn_cv         = []
    dyn_sig        = []
    dyn_ynorm      = []
    dyn_gen.append(0)
    dyn_fev.append(fevals)
    dyn_fit.append(lb_fv)
    dyn_cv.append(lb_cv)
    dyn_sig.append(sigma)
    dyn_ynorm.append(np.linalg.norm(lb_y))
     
    while not termination:
        # compute Moore-Penrose inverse of M for the back-calculation step
        # which is applied to correct the mutation vectors corresponding to repaired solutions
        Minv = np.linalg.pinv(M,1e-12)
                
        # create new generation of offspring candidate solutions
        newpop_z = np.random.randn(dim,lam)
        newpop_d = np.matmul(M,newpop_z)
        newpop_y = np.transpose(np.tile(yParent,(lam,1)))+sigma*newpop_d
        
        for k in range(lam):        
            # initialization of repair count
            repi = 0
            # check for bound constraint satisfaction and repair if necessary
            new_y = keep_range(newpop_y[:,k],lower_bounds,upper_bounds)
            # check whether repair has been carried out in order to apply back-calculation and count 
            if  np.sum(new_y == newpop_y[:,k])<dim:
                repi = 1
            # evaluation of offspring candiadate solution (in bounds) 
            [fval, gv, hv] = objFun(new_y)
            #compute constraint violation
            convio  = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>delta))])
            #if type(hv) == list:
            #    hv = [np.max([0,np.abs(v) - delta]) for v in hv]
            #else:
            #    if np.abs(hv) <= delta:
            #        hv = 0.
            #formerly: convio          = sum([sum(gv.*(gv>0)), sum(abs(hv).*(abs(hv)>input.delta))])./(problem.gn(CEC_fun_no) + problem.hn(CEC_fun_no));            
            fevals = fevals  + 1
            # initialize individual repair step count
            reps=1            
            # apply gradient-based mutation step if conditions are satisfied
            if g % dim == 0 and np.random.rand(1) <= 0.2: 
                while convio > 0 and reps <= max_reps:
                    new_mutant = gradientMutation(new_y,np.array(gv),np.array(hv),objFun)
                    new_mutant = keep_range(new_mutant,lower_bounds,upper_bounds)
                    [fval, gv, hv]  = objFun(new_mutant)                                                               
                    convio = np.sum([np.sum(gv*(gv>0)), np.sum(np.abs(hv)*(np.abs(hv)>delta))])
                    fevals = fevals + dim +1
                    reps=reps+1
                    new_y = new_mutant
                    if np.sum(new_y == newpop_y[:,k])<dim:
                        repi = 1
            newpop_y[:,k] = new_y
            # apply back-calculation if necessary (if repair was performed at some point)
            if repi > 0:
                newpop_d[:,k] = (newpop_y[:,k]-yParent)/sigma
                newpop_z[:,k] = np.matmul(Minv,newpop_d[:,k])
            newpop_f[k]  = fval
            newpop_cv[k] = convio
           
          
        # Implementation of Epsilon Constraint Ordering 
        # feasible (constraint violation below epsilon value!!!) solutions dominate infeasible ones AND feasible solutions are sorted according to their fitness values 
        ranking = eps_sort(newpop_f,newpop_cv,Epsilon)
        
        best_ind = ranking[0]
        lb_fv = newpop_f[best_ind]             # best feasible fitness value of current population
        lb_y = newpop_y[:,best_ind]           # best feasible individual of current population
        lb_cv = newpop_cv[best_ind]
        
        # Sort by fitness and compute weighted mean into xmean
        parent_z = np.matmul(newpop_z[:,ranking[0:mu]], weights)  # recombination
        
        parent_d = np.matmul(newpop_d[:,ranking[0:mu]], weights)  # recombination
        yParent  = yParent + sigma*parent_d # update population certroid
               
        # Cumulation: Update evolution paths
        ps = (1-cs) * ps + sqrt_s * parent_z

          # Update transformation matrix 
        M = (1 - 0.5*c1 - 0.5*cmu) * M + (0.5*c1)*np.matmul(M,np.outer(np.transpose(ps),ps))
        for m in range(mu):
            M = M + (0.5*cmu*weights[m])*np.outer(np.transpose(newpop_d[:,ranking[m]]),newpop_z[:,ranking[m]])
                
        # Adapt mutation strength sigma
        #sigma = sigma * np.exp((cs/2)*(np.linalg.norm(ps)**2/dim - 1))
        

        # Reset step to prevent unstable pseudo inverse calculations
        #liMM = (M > 1e+12 | isnan(M) | isinf(M);
        liMM = np.any(np.abs(M) > 10**12) or np.any(np.isnan(M)) or np.any(np.isinf(M))      
        #siMM = M < -1e+12 | isnan(M) | isinf(M);
        #siMM = M < -1e+12 | isnan(M) | isinf(M)
        #if sum(sum(liMM))>1 || sum(sum(siMM))>1
        #    M = eye(input.dim);
        #    ps = ones(input.dim,1);
        #change_flag = False
        if liMM:
            M = np.eye(dim)
            ps = np.ones(shape=(dim,))
            #print(sigma)
            #change_flag = True
        #end
                
        # Adapt mutation strength sigma
        sigma = np.min([sigma * np.exp((cs/2)*(np.linalg.norm(ps)**2/dim - 1)),sigmax]) 
        
        #if sigma == sigmax:
        #    print(g)
        #    print(cs)
        #    print(M)
        #    print(np.exp((cs/2)*(np.linalg.norm(ps)**2/dim - 1)))
        #    print(np.linalg.norm(ps))
        #    print(parent_z)
        #if change_flag:
        #    print(sigma)
       
        # update the best solution found so far                        
        #formerly: if (best_conv==0 && global_best.conv==0 && best_val < global_best.val) ||...
        #       (best_conv==global_best.conv && best_val < global_best.val) || best_conv<global_best.conv

        tmp1 = (lb_cv == 0) and (gb_cv == 0) and (lb_fv < gb_fv)
        tmp2 = (lb_cv == gb_cv) and (lb_fv < gb_fv)
        tmp3 = (lb_cv < gb_cv) and (lb_fv < gb_fv)
        if tmp1 or tmp2 or tmp3:
            gb_fv = lb_fv		# global best fitness observed over all populations
            gb_y = lb_y 
            gb_cv = lb_cv 
        
 
        #if eps_rank(lb_fv,lb_cv,gb_fv,gb_cv,Epsilon):
        #    #print([lb_fv,lb_cv,gb_fv,gb_cv])
        #    gb_fv   = lb_fv		# global best fitness observed over all populations
        #    gb_y    = lb_y 
        #    gb_cv  = lb_cv 
                        
        # termination criteria
        if g >= maxIter:
            termination = True;
        elif fevals >= budget:
            termination = True;
        
        # update generation counter
        g=g+1 
        
        # update epsilon-level threshold
        if g<TC:
          Epsilon=EPSILON*((1-g/TC)**CP)
        else:
          Epsilon=0
            
        # logging the algorithm dynamics with respect to the global best observation
        dyn_gen.append(g)
        dyn_fev.append(fevals)
        dyn_fit.append(gb_fv)
        dyn_cv.append(gb_cv)
        dyn_sig.append(sigma)
        dyn_ynorm.append(np.linalg.norm(gb_y))
                         
    return gb_y, gb_fv, gb_cv, dyn_fit, dyn_cv, dyn_sig, dyn_ynorm, dyn_gen