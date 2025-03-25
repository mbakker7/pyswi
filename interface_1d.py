import numpy as np
from scipy.optimize import root

def jac(x, hold, *args, fun=None):
    dp = -1e-6
    ntot = len(x)
    d = dp * np.eye(ntot)
    rv = np.zeros((ntot, ntot))
    funx = fun(x, hold, *args)
    for n in range(ntot):
        rv[:, n] = (fun(x + d[n], hold, *args) - funx) / dp
    return rv

def fdfresh1d_step(hf, hfold, zeta, zetaold, k, S, Se, zb, zt, Qf, fixed, ghb, rhof, rhos,  
                   delx, delt, steady=False, hsea=0, storage_only=False, budget=False):
    """One-dimensional transient flow
    """
    alphaf = rhof / (rhos - rhof)
    if steady:
        zetaold = alphaf * (rhos / rhof * hsea - hfold)
        zeta = alphaf * (rhos / rhof * hsea - hf)
        zetaold = np.maximum(zetaold, zb)
        zeta = np.maximum(zeta, zb)
    topold = np.minimum(hfold, zt)
    botold = np.maximum(zetaold, zb)
    bfold = np.maximum(topold - botold, 0)
    top = np.minimum(hf, zt)
    bot = np.maximum(zeta, zb)
    bf = np.maximum(top - bot, 0) # thickness cannot be negative
    #storage1 = S * delx * (hf - hfold) / delt
    #storage2 = -S * delx * (zeta - zetaold) / delt
    storage1 = S * delx * (bf - bfold) / delt
    storage2 = Se * bf * delx * (hf - hfold) / delt
    #bf = (hf - zeta)
    #print(bf)
    #bf = np.minimum(np.maximum(hf, zb), zt) - np.maximum(np.minimum(zeta, zt), zb)
    #print(bf)
    bf = np.where(hf[:-1] >= hf[1:], bf[:-1], bf[1:]) # upstream weighing
    bf = np.maximum(1e-3, bf) # make sure at least 1 mm
    C = k * bf / delx
    A = np.diag(C, 1) + np.diag(C, -1)
    A -= np.diag(np.sum(A, 1))
    rhs = -Qf + storage1 + storage2
    if ghb is not None:
        for icol, hstar, Cstar in ghb:
            #if (hfold[icol] >= hstar) and (zetaold[icol] <= zt[icol]): # only freshwater outflow when hf > hstar and zeta below top
                #print(icol, hstar, Cstar)
                A[icol, icol] -= Cstar
                rhs[icol] -= Cstar * hstar
            #else:
                #print(icol, hf[icol], zeta[icol])
    if fixed is not None:
        ifixed = np.where(~np.isnan(fixed))
        A[ifixed] = 0
        A[ifixed, ifixed] = 1
        rhs[ifixed] = fixed[ifixed]
    if budget: # compute and return the water budget
        Qsource = Qf * delt
        Qfixed = np.zeros(len(hf))
        Qghb = np.zeros(len(hf))
        if fixed is not None:
            Qx = C * (hf[:-1] - hf[1:])
            Qin = np.zeros(len(hf)) # flow into the cell
            Qin[1:] = Qx # inflow
            Qin[:-1] -= Qx # outflow
            Qfixed[ifixed] = -Qin[ifixed] * delt # flow into the cell is a sink
            Qsource[ifixed] = 0 # no source on constant head cells
        if ghb is not None:
            ighb = [g[0] for g in ghb]
            hghb = [g[1] for g in ghb]
            Cghb = [g[2] for g in ghb]
            Qghb[ighb] = Cghb * (hghb - hf[ighb]) * delt
        storage_increase = (storage1 + storage2) * delt
        return Qsource, Qfixed, Qghb, storage_increase
    if storage_only:
        return -rhs
    sol = A @ hf - rhs
    return sol

def fdsalt1d_step(zeta, zetaold, hf, hfold, k, S, Se, zb, zt, Qs, fixed, ghb, rhof, rhos,
                  delx, delt, storage_only=False, budget=False):
    """One-dimensional transient flow
    """
    alphas = rhos / (rhos - rhof)
    topold = np.minimum(zetaold, zt)
    bsold = np.maximum(topold - zb, 0)
    top = np.minimum(zeta, zt)
    bs = np.maximum(top - zb, 0) # bs cannot be negative
    hsold = (rhos - rhof) / rhos * zetaold + rhof / rhos * hfold # head in saltwater
    hs = (rhos - rhof) / rhos * zeta + rhof / rhos * hf # head in saltwater
    storage1 = S * delx * (bs - bsold) / delt
    storage2 = Se * bs * delx * (hs - hsold) / delt
    bs = np.where(hs[:-1] >= hs[1:], bs[:-1], bs[1:]) # upstream weighing
    bs = np.maximum(1e-3, bs) # make sure at least 1 mm
    C = k * bs / delx
    A = np.diag(C, 1) + np.diag(C, -1)
    A -= np.diag(np.sum(A, 1))
    rhs = -Qs + storage1 + storage2
    if ghb is not None:
        #print("hello ghb")
        for icol, hstar, Cstar in ghb:
            #if (zetaold[icol] >= zt[icol]): # or (hs[icol] <= hstar): # if zeta at the top and/or hs < hstar
                A[icol, icol] -= Cstar
                rhs[icol] -= Cstar * hstar
    if fixed is not None:
        ifixed = np.where(~np.isnan(fixed))
        A[ifixed] = 0
        A[ifixed, ifixed] = 1
        rhs[ifixed] = fixed[ifixed]
    if budget: # compute and return the water budget
        Qsource = Qs * delt # no sources yet in salt
        Qsource = np.zeros(len(hs)) # no sources yet in salt
        Qfixed = np.zeros(len(hs))
        if fixed is not None:
            Qx = C * (hs[:-1] - hs[1:])
            Qin = np.zeros(len(hs)) # flow into the cell
            Qin[1:] = Qx # inflow
            Qin[:-1] -= Qx # outflow
            Qfixed[ifixed] = -Qin[ifixed] * delt # flow into the cell is a sink
            Qsource[ifixed] = 0 # no source on constant head cells
        storage_increase = (storage1 + storage2) * delt
        return Qsource, Qfixed, storage_increase
    if storage_only:
        return -rhs
    return A @ hs - rhs # still doing saltwater here, but hf and zeta should do the same

def fdfreshsalt_step(solnew, solold, k, S, Se, zb, zt, Qf, Qs, fixed, ghb, rhof, rhos,
                     delx, delt):
    # solnew is vector with fresh heads and then zeta
    # for now 'fixed' is the same for fresh and zeta and there are separate Qf and Qs is only
    # for fresh water                        
    ncol = len(solnew) // 2
    hf = solnew[:ncol]
    zeta = solnew[ncol:]
    hfold = solold[:ncol]
    zetaold = solold[ncol:]
    hfsol = fdfresh1d_step(hf, hfold, zeta, zetaold, k, S, Se, zb, zt, Qf, fixed, ghb, rhof, rhos,
                          delx, delt)
    zetasol = fdsalt1d_step(zeta, zetaold, hf, hfold, k, S, Se, zb, zt, Qs, fixed, ghb, rhof, rhos,
                         delx, delt)
    return np.hstack((hfsol, zetasol))

def swi_simulate_steady(hfini, k, S, Se, zb, zt, Qf, Qs, fixed, ghb, rhof, rhos,
                 delx, hsea, perlen, nstep, tmultiply, maxiter=100, silent=True, debug=False):
    if tmultiply > 1:
        dt0 = perlen * (1 - tmultiply) / (1 - tmultiply ** nstep)
    else:
        dt0 = perlen / nstep
    tarray = np.zeros(nstep + 1)
    htol = 1e-6 # absolute convergence criterium for heads
    ncol = len(hfini)
    sol = np.zeros((nstep + 1, ncol))
    sol[0] = hfini
    for istep in range(nstep):
        dt = dt0 * tmultiply ** istep
        tarray[istep + 1] = tarray[istep] + dt
        solnew = sol[istep] + 0.1
        for jiter in range(maxiter):
            sol[istep + 1] = solnew
            R = fdfresh1d_step(solnew, sol[istep], 0, 0, k, S, Se, zb, zt, Qf, fixed, ghb, rhof, rhos,
                                    delx, dt, True)
            J = jac(solnew, sol[istep], 0, 0, k, S, Se, zb, zt, Qf, fixed, ghb, rhof, rhos,
                                    delx, dt, True, fun=fdfresh1d_step)
            solnew = np.linalg.solve(J, -R + J @ solnew)
            #print(np.max(np.abs(hsol[istep + 1] - hnew)))
            if np.max(np.abs(sol[istep + 1] - solnew)) < htol:
                if not silent:
                    print(f'iterations: {jiter + 1}')
                sol[istep + 1] = solnew
                break
            if jiter == maxiter - 1:
                print(f'zero based time step: {istep}')
                print(f'Error: convergence not reached after maxiter={maxiter} iterations')
                sol[istep + 1] = solnew
                break
    hfsol = sol
    alphaf = rhof / (rhos - rhof)
    zetasol = alphaf * (rhos / rhof * hsea - hfsol)
    zetasol = np.maximum(zetasol, zb)
    zetasol = np.minimum(zetasol, zt)
    if debug:
        return hfsol,zetasol, tarray, R, J
    return hfsol, zetasol, tarray

def swi_simulate(hfini, zetaini, k, S, Se, zb, zt, Qf, Qs, fixed, ghb, rhof, rhos,
                 delx, perlen, nstep, tmultiply, maxiter=100, silent=True, debug=False):
    if tmultiply > 1:
        dt0 = perlen * (1 - tmultiply) / (1 - tmultiply ** nstep)
    else:
        dt0 = perlen / nstep
    tarray = np.zeros(nstep + 1)
    htol = 1e-6 # absolute convergence criterium for heads
    ncol = len(hfini)
    sol = np.zeros((nstep + 1, 2 * ncol))
    sol[0, :ncol] = hfini
    sol[0, ncol:] = zetaini
    for istep in range(nstep):
        dt = dt0 * tmultiply ** istep
        tarray[istep + 1] = tarray[istep] + dt
        solnew = sol[istep]
        for jiter in range(maxiter):
            sol[istep + 1] = solnew
            R = fdfreshsalt_step(solnew, sol[istep], k, S, Se, zb, zt, Qf, Qs, fixed, ghb, rhof, rhos,
                                    delx, dt)
            J = jac(solnew, sol[istep], k, S, Se, zb, zt, Qf, Qs, fixed, ghb, rhof, rhos,
                                    delx, dt, fun=fdfreshsalt_step)
            if silent == False:
                print('R')
                print(R)
                print('J')
                #print(np.diag(J, -1))
                #print(np.diag(J))
                #print(np.diag(J, 1))
                print(J)
            solnew = np.linalg.solve(J, -R + J @ solnew)
            if silent == False:
                print('solnew')
                print(solnew)
            #print(np.max(np.abs(hsol[istep + 1] - hnew)))
            if np.max(np.abs(sol[istep + 1] - solnew)) < htol:
                if not silent:
                    print(f'iterations: {jiter + 1}')
                sol[istep + 1] = solnew
                break
            if jiter == maxiter - 1:
                print(f'zero based time step: {istep}')
                print(f'Error: convergence not reached after maxiter={maxiter} iterations')
                sol[istep + 1] = solnew
                break
    hfsol = sol[:, :ncol]
    zetasol = sol[:, ncol:]
    zetasol = np.maximum(zetasol, zb) # zeta is solution variable, so need to adjust
    zetasol = np.minimum(zetasol, zt)
    hssol = (rhos - rhof) / rhos * zetasol + rhof / rhos * hfsol
    if debug:
        return hfsol, hssol, zetasol, tarray, R, J
    return hfsol, hssol, zetasol, tarray