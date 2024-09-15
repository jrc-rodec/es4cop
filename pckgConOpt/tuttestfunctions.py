# -*- coding: utf-8 -*-
from pckgConOpt import o1, o2, o3, o4
import numpy as np


def get_check_dimension(x,o):
    d = x.shape[0]
    if d > o.shape[0]:
        raise ValueError('Dimension of solution too large. Must be equal or less than {}'.format(o.shape[0]))
    return d


def cop1(x):
    # bound constraints: [-100,100]
    d = get_check_dimension(x,o1)
    y = x - o1[:d]
    f = 0
    for i in range(d):
        f = f + np.sum(y[:i])**2
    g = np.sum(y**2 - 5000*np.cos(0.1*np.pi*y) - 4000)
    h = 0.
    return f,g,h

def cop2(x):
    # bound constraints: [-100,100]
    d = get_check_dimension(x,o2)
    y = x - o2[:d]
    f = 0
    for i in range(d):
        f = f + np.sum(y[:i])**2
    g = np.sum(y**2 - 5000*np.cos(0.1*np.pi*y) - 4000)
    h = -np.sum(y*np.sin(0.1*np.pi*y))
    return f,g,h

def cop3(x):
    # bound constraints: [-20,20]
    d = get_check_dimension(x,o3)
    y = x - o3[:d]
    f  = np.sum(y**2 - 10*np.cos(2*np.pi*y) + 10)
    h1 = np.sum(-y*np.sin(y))
    h2 = np.sum(y*np.sin(np.pi*y))
    h3 = np.sum(-y*np.cos(y))
    h4 = np.sum(y*np.cos(np.pi*y))
    tmp = 2*np.sqrt(np.abs(y))
    h5 = np.sum(y*np.sin(tmp))
    h6 = np.sum(-y*np.sin(tmp))
    h = np.array([h1,h2,h3,h4,h5,h6])
    g = 0
    return f,g,h

def cop4(x):
    # bound constraints: [-100,100]
    d = get_check_dimension(x,o4)
    y = x - o4[:d]
    f = np.sum(y)
    h = np.sum((y[:d-1] - y[1:d])**2)
    g = np.prod(y)
    return f,g,h

def cop5(x):
    # weight minimization of a speed reducer
    # bound constraints: 
    # xmin = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]
    # xmax = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
    d = x.shape[0]
    if d != 7:
        raise ValueError('Dimension of input must 7!')
    f = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2]-43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + \
        7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    g1 = -x[0]*x[1]**2*x[2] + 27
    g2 = -x[0]*x[1]**2*x[2]**2 + 397.5
    g3 = -x[1]*x[5]**4*x[2]*x[3]**(-3) + 1.93
    g4 = -x[1]*x[6]**4*x[2]/x[4]**3 + 1.93
    g5 = 10*x[5]**(-3)*np.sqrt(16.91*10**6 + (745*x[3]/(x[1]*x[2]))**2) - 1100
    g6 = 10*x[6]**(-3)*np.sqrt(157.5*10**6 + (745*x[4]/(x[1]*x[2]))**2) - 850
    g7 = x[1]*x[2] - 40
    g8 = -x[0]/x[1] + 5
    g9 = x[0]/x[1] - 12
    g10 = 1.5*x[5] - x[3] + 1.9
    g11 = 1.1*x[6] - x[4] + 1.9
    g = np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])
    h = 0
    return f,g,h
