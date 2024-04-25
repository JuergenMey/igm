# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:23:04 2024

@author: Jürgen
"""



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from igm.modules.utils import *
from scipy.sparse import coo_matrix, spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import lsqr


def params(parser):
    parser.add_argument(
        "--hillslope_diffusivity",
        type=float,
        default=1,
        help="Hillslope diffusivity in m^2/yr.",    
    )
    parser.add_argument(
        "--hillslope_diffusion_update_freq",
        type=float,
        default=1,
        help="Hillslope_diffusion_update_freq (yr)",    
    )
    
   
def initialize(params, state):
    
    state.tcomp_hillslope_diffusion = []
    state.tlast_hillslope_diffusion = tf.Variable(params.time_start, dtype=tf.float32)
    
    state.hillslope_diffusion = tf.zeros_like(state.topg)
    
    nrc = tf.size(state.topg)  # number of cells in domain
    siz = tf.shape(state.topg)  # size of domain
    X = tf.transpose(tf.reshape(tf.Variable(np.arange(1,nrc+1)),siz)) # create index matrix
    ic = np.full((siz[0]+2, siz[1]+2), np.nan)
    ic[1:-1, 1:-1] = X
    # ic = tf.Variable(ic)
    I = ~np.isnan(ic)
    # I = tf.logical_not(tf.math.is_nan(ic))
    # icd = tf.zeros([tf.reduce_sum(tf.cast(I, tf.int32)),4])
    icd = np.zeros((np.sum(I), 4))
    
    # Shift logical matrix I across the neighbors
    Ir = np.roll(I, 1, axis=1)
    Id = np.roll(I, 1, axis=0)
    Il = np.roll(I, -1, axis=1)
    Iu = np.roll(I, -1, axis=0)
    ict = np.transpose(ic)
    icd[:,0] = ict[Id[:,:]]
    icd[:,1] = ict[Ir[:,:]]
    icd[:,2] = ict[Iu[:,:]]
    icd[:,3] = ict[Il[:,:]]
    
    ic = np.tile(ict[I[:]],(1,4));
    icd = icd.flatten(order = 'F')  
    
    # Remove NaNs in neighbors
    ic = ic[:,~np.isnan(icd)]
    icd = icd[~np.isnan(icd)]
    
    ic = ic.flatten()
    nnz = len(ic)
    
    # calculate laplacian
    L = coo_matrix((np.ones(nnz), (ic-1, icd-1)), shape=(62001, 62001)).tocsr()
    L = diags(np.asarray(L.sum(axis=1)).squeeze(),format='csr') - L;
    
    # and the diffusion matrix
    D = np.ones_like(state.topg)*params.hillslope_diffusivity
    D = D.flatten();
    
    state.nrc = nrc
    state.L = L
    state.D = D
    

            
         
def update(params, state):
    if (state.t - state.tlast_hillslope_diffusion) >= params.hillslope_diffusion_update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "update topg_hillslope_diffusion at time : " + str(state.t.numpy())
            )
 
        if hasattr(state, "logger"):
            state.logger.info("Update hillslope diffusion at time : " + str(state.t.numpy()))
        state.tcomp_hillslope_diffusion.append(time.time())
        nrc = state.nrc
        L = state.L
        D = state.D
        
        
        try:
            D  = eye(nrc) + spdiags(D,0,nrc,nrc)*state.dt.numpy()/(2*state.dx.numpy()*state.dx.numpy())*L     
        except:
            D  = eye(nrc) + spdiags(D,0,nrc,nrc)*state.dt/(2*state.dx.numpy()*state.dx.numpy())*L 
            
        D = D.astype(dtype='float32')
        I = tf.math.is_nan(state.topg)
        state.topg = tf.where(tf.math.is_nan(state.topg), tf.zeros_like(state.topg), state.topg)
        Z1 = np.reshape(state.topg, [-1]) 
        
        # Solve the linear least squares problem A_sparse * x = b using lsqr
        Z1, istop, itn, r1norm = lsqr(D, Z1)[:4]
        Z1 = np.float32(Z1)
        Z = tf.reshape(Z1,tf.shape(state.topg));
        state.hillslope_diffusion = state.hillslope_diffusion+(state.topg-Z)
        state.topg = tf.reshape(Z1,tf.shape(state.topg))
        
        state.tlast_hillslope_diffusion.assign(state.t)

        state.tcomp_hillslope_diffusion[-1] -= time.time()
        state.tcomp_hillslope_diffusion[-1] *= -1
        
def finalize(params, state):
    pass