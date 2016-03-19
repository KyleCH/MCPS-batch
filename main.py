#!/usr/bin/env python3

import os

import numpy as np
from pint import UnitRegistry
import scipy.constants as const

import parameters as param
ureg = param.ureg
Q_ = param.Q_
n = param.n

# ======================================================================

# Speed of light in vacuum.
c = Q_(const.value('speed of light in vacuum'),
       const.unit('speed of light in vacuum'))

# Metric tensor.
metric = np.asarray(
    [[-1., 0., 0., 0.],
     [ 0., 1., 0., 0.],
     [ 0., 0., 1., 0.],
     [ 0., 0., 0., 1.]])

# ======================================================================

def setutc():
    import datetime
    import string
    utcwhen = str(datetime.datetime.utcnow())
    utcwhen = utcwhen[0:string.rfind(utcwhen, '.')]
    utcwhen_ = string.replace(utcwhen, ' ', '_')
    utcwhen_ = string.replace(utcwhen_, ':', '-')
    return utcwhen, utcwhen_

# ======================================================================

def f(t, X, U, qmr, F):
    return c*np.asarray([
        y[4:8]/y[4],
        qmr/y[4]*np.einsum('ij,j->i', F, y[4:8])])

# ======================================================================

def step(h, t, y, qmr, F):
    k1 = f(t, y, qmr, F)
    hh = .5*h
    k2 = f(t+hh, y+hh*k1, qmr, F)
    k3 = f(t+hh, y+hh*k2, qmr, F)
    k4 = f(t+h, y+h*k3, qmr, F)
    return t+hh, y+h*(k1+2.*(k2+k3)+k4)/6.

# ======================================================================

def iterator(fname, h, t0, X0, U0, qmr, F):
    t = t0
    y = np.asarray([X0, U0])
    with open(fname, 'w') as f:
        f.write('# t\t'
            +'X[0]\tX[1]\tX[2]\tX[3]\t'
            #+'U[0]\tU[1]\tU[2]\tU[3]\t'
            #+'F[0, 0]\tF[0, 1]\tF[0, 2]\tF[0, 3]\t'
            #+'F[1, 0]\tF[1, 1]\tF[1, 2]\tF[1, 3]\t'
            #+'F[2, 0]\tF[2, 1]\tF[2, 2]\tF[2, 3]\t'
            #+'F[3, 0]\tF[3, 1]\tF[3, 2]\tF[3, 3]\n'
            +'U[0]\tU[1]\tU[2]\tU[3]\n'
            +str(t) + '\t'
            +str(y[0,0])+'\t'+str(y[0,1])+'\t'+str(y[0,2])+'\t'+str(y[0,3])+'\t'
            +str(y[1,0])+'\t'+str(y[1,1])+'\t'+str(y[1,2])+'\t'+str(y[1,3]) #+'\t'
            #+str(F[0,0])+'\t'+str(F[0,1])+'\t'+str(F[0,2])+'\t'+str(F[0,3])+'\t'
            #+str(F[1,0])+'\t'+str(F[1,1])+'\t'+str(F[1,2])+'\t'+str(F[1,3])+'\t'
            #+str(F[2,0])+'\t'+str(F[2,1])+'\t'+str(F[2,2])+'\t'+str(F[2,3])+'\t'
            #+str(F[3,0])+'\t'+str(F[3,1])+'\t'+str(F[3,2])+'\t'+str(F[3,3])
            )
        for i in range(n):
            t, y = step(h, t, y, qmr, F)
            fname.write(str(t)+'\t'
                +str(y[0,0])+'\t'+str(y[0,1])+'\t'+str(y[0,2])+'\t'+str(y[0,3])+'\t'
                +str(y[1,0])+'\t'+str(y[1,1])+'\t'+str(y[1,2])+'\t'+str(y[1,3]) #+'\t'
                #+str(F[0,0])+'\t'+str(F[0,1])+'\t'+str(F[0,2])+'\t'+str(F[0,3])+'\t'
                #+str(F[1,0])+'\t'+str(F[1,1])+'\t'+str(F[1,2])+'\t'+str(F[1,3])+'\t'
                #+str(F[2,0])+'\t'+str(F[2,1])+'\t'+str(F[2,2])+'\t'+str(F[2,3])+'\t'
                #+str(F[3,0])+'\t'+str(F[3,1])+'\t'+str(F[3,2])+'\t'+str(F[3,3])
                )
        

# ======================================================================
# Setup output directory and parameter log.
# ======================================================================

# Set UTC timestamp for execution/output directory.
utcwhen, utcwhen_ = setutc()

# Create output directory.
opath = './output/'
if not os.path.isdir(opath):
    os.makedirs(opath)
opath += utcwhen_ + '/'
if not os.path.isdir(opath):
    os.makedirs(opath)
else:
    print('Error: an output directory already exists with the current UTC '
        'timestamp: '+utx)
    raise SystemExit

# Open parameter log.
with open(opath + 'parameter_log.txt', 'w') as plog:

    # Write UTC timestamp and non-looped parameters to parameter log.
    plog.write(
        'UTC date-time stamp: ' + utcwhen + '\n\n'
        + 'Number of iterations (n): ' + str(n))

# ======================================================================
# Loop over parameters.
# ======================================================================

    num_t0 = param.t0.shape[0]
    num_r0 = param.r0.shape[0]
    num_vhat0 = param.vhat0.shape[0]
    num_q = param.q.shape[0]
    num_m = param.m.shape[0]
    num_T0 = param.T0.shape[0]
    num_By = param.By.shape[0]
    num_h = param.h.shape[0]

    for i_t0 in range(num_t0):
        t0 = param.t0[i_t0]
        print('t0:', t0)
        
        for i_r0 in range(num_r0):
            r0 = param.r0[i_r0]
            print('r0:', r0)
            
            # Initial four-position.
            X0 = np.append([t0*c], r0)
            
            for i_vhat0 in range(num_vhat0):
                vhat0 = param.vhat0[i_vhat0]
                print('vhat0:', vhat0)
                
                for i_q in range(num_q):
                    q = param.q[i_q]
                    print('q:', q)
                    
                    for i_m in range(num_m):
                        m = param.m[i_m]
                        print('m:', m)
                        
                        # Rest mass energy equivalent.
                        mc2 = m * c**2
                        
                        # Charge mass ratio.
                        qmr = q / m
                        
                        for i_T0 in range(num_T0):
                            T0 = param.T0[i_T0]
                            print('T0:', T0)
                            
                            # Initial speed.
                            gamma0 = T0 / mc2 + 1
                            speed0 = c * np.sqrt(1 - gamma0**-2)
                            
                            # Initial velocity.
                            v0 = vhat0 * speed0
                            
                            # Initial four-velocity.
                            U0 = gamma0*np.append([c], r0)
                            
                            for i_By in range(num_By):
                                By = param.By[i_By]
                                print('By:', By)
                                
                                Ex_c = Q_(0., 'newton/coulomb') / c
                                Ey_c = Ex_c
                                Ez_c = Ex_c
                                
                                Bx = Q_(0., 'mT')
                                
                                Bz = Bx
                                
                                F = np.asarray(
                                    [[  0., Ex_c, Ey_c, Ez_c],
                                     [Ex_c,   0.,   Bz,  -By],
                                     [Ey_c,  -Bz,    0.,  B_x],
                                     [Ez_c,   By,   -Bx,  0.]])
                                
                                for i_h in range(num_h):
                                    h = param.h[i_h]
                                    print('h:', h)
                                    
                                    # Set output file name.
                                    ofile_utcwhen, ofile_utcwhen_ = setutc()
                                    ofile_name = ofile_utcwhen_ + '.txt'
                                    
                                    # Write outpit file name and parameters to
                                    # parameter log.
                                    plog.write('\n\n' + ofile_name
                                        + '\n\th:\t' + str(h)
                                        + '\n\tm:\t' + str(m)
                                        + '\n\tq:\t' + str(q)
                                        + '\n\tT0:\t' + str(T0)
                                        + '\n\tt0:\t' + str(t0)
                                        + '\n\tr0:\t' + str(r0)
                                        + '\n\tv0:\t' + str(v0)
                                        + '\n\tX0:\t' + str(X0)
                                        + '\n\tU0:\t' + str(U0))
                                    
                                    # Solve.
                                    iterator(fname, h, t0, X0, U0, qmr, F)