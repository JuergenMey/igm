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
        default=0.01,
        help="hillslope diffusivity in m^2/yr.",    
    )
    parser.add_argument(
        "--hillslope_erosion_coefficient",
        type=float,
        default=0.01,
        help="Hillslope erosion coefficient",    
    )
    parser.add_argument(
        "--critical_slope",
        type=float,
        default=0.7,
        help="Critical slope",    
    )
    parser.add_argument(
        "--hillslope_update_freq",
        type=float,
        default=1,
        help="Hillslope_diffusion_update_freq (yr)",    
    )
    parser.add_argument(
        "--maxb",
        type=float,
        default=0.7,
        help="Maximum_bed_gradient",    
    )
    
   
def initialize(params, state):
    
    state.tcomp_hillslope = []
    state.tlast_hillslope = tf.Variable(params.time_start, dtype=tf.float32)
    
    state.hillslope_erosion = tf.zeros_like(state.topg)
    state.hillslope_erate = tf.zeros_like(state.topg)
    state.sed = tf.zeros_like(state.topg)
    
    if not hasattr(state,'sed'):
        state.sed = tf.zeros_like(state.topg)
    state.bed = state.topg - state.sed
    state.nrc = tf.size(state.topg)
   
         
def update(params, state):
    if (state.t - state.tlast_hillslope) >= params.hillslope_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update hillslope erosion at time : " + str(state.t.numpy()))
        state.tcomp_hillslope.append(time.time())
        
        state.bed = state.topg - state.sed
        sc = params.critical_slope
        Ks = params.hillslope_diffusivity
        Ke = params.hillslope_erosion_coefficient
        
        #/***************** get gradients *****************/
        # x-dir
        hp_dbdx = (-tf.roll(state.topg,-1,1)+state.topg)/state.dx;
        hp_dhdx = (-tf.roll(state.usurf,-1,1)+state.usurf)/state.dx;
        hp_dtdx = hp_dhdx;  
        hp_dbdx = tf.where(hp_dbdx < -params.maxb, -params.maxb, hp_dbdx)
        hp_dbdx = tf.where(hp_dbdx > params.maxb, params.maxb, hp_dbdx)
       
        # y-dir
        vp_dbdy = (state.topg-tf.roll(state.topg,1,0))/state.dx;
        vp_dhdy = (state.usurf-tf.roll(state.usurf,1,0))/state.dx;
        vp_dtdy = vp_dhdy;
        vp_dbdy = tf.where(vp_dbdy < -params.maxb, -params.maxb, vp_dbdy)
        vp_dbdy = tf.where(vp_dbdy > params.maxb, params.maxb, vp_dbdy)
        
        dbdx = 0.5*(hp_dbdx + tf.roll(hp_dbdx,1,1));
        dhdx = 0.5*(hp_dhdx + tf.roll(hp_dhdx,1,1));
        dtdx = 0.5*(hp_dtdx + tf.roll(hp_dtdx,1,1));

        dbdy = 0.5*(vp_dbdy + tf.roll(vp_dbdy,-1,0));
        dhdy = 0.5*(vp_dhdy + tf.roll(vp_dhdy,-1,0));
        dtdy = 0.5*(vp_dtdy + tf.roll(vp_dtdy,-1,0));

        # Compute slope magnitude       
        bslope = getmag(dbdx,dbdy);
        hslope = getmag(dhdx,dhdy);
        tslope = getmag(dtdx,dtdy);
        
        
        #hp_dbdx, vp_dbdy = compute_gradient_tf(state.topg, state.dx, state.dx)
        
        hillslope_erosion = state.hillslope_erosion
        hillslope_erate = state.hillslope_erate
        sed = state.sed;
        
        #/***************** Erosion (critical slope) *****************/
        
        ###### h-points ################
        
        fac = 1.0 - tf.square(tf.divide(hp_dbdx,sc));
        fac = tf.where(fac < 1.0e-4, tf.ones_like(fac)*1.0e-4, fac)
        sdiff = Ke/fac;
        ero = sdiff*hp_dbdx*state.dt/state.dx*tf.sqrt(1.0+tf.square(bslope));        
        
        maxero = -state.bed+tf.roll(state.bed,-1,1)-sc*state.dx;
        maxero = tf.where(maxero  < 0.0, tf.zeros_like(maxero), maxero);
        ero = tf.where(tf.logical_and(ero < -maxero, ero < 0),-maxero, ero)
        ero = tf.where(state.thk > 5.0, tf.zeros_like(ero), ero)
        csero = tf.roll(ero,1,1);
        ero_l = csero<0.0;
        state.bed = tf.where(ero_l,state.bed+csero,state.bed)
        hillslope_erosion = tf.where(ero_l,hillslope_erosion-csero,hillslope_erosion)
        hillslope_erate = tf.where(ero_l,-csero/state.dt,hillslope_erate)
        sed = tf.where(ero_l,sed-csero,sed)
        
        maxero = state.bed-tf.roll(state.bed,-1,1)-sc*state.dx;
        maxero = tf.where(maxero  < 0.0, tf.zeros_like(maxero), maxero);
        ero = tf.where(tf.logical_and(ero > maxero, ero > 0),maxero, ero) 
        ero = tf.where(state.thk > 5.0, tf.zeros_like(ero), ero)
        ero_u = ero>0;                      
        # if (cells[i][j].ice > 5) ero = 0.0;
        state.bed = tf.where(ero_u,state.bed - ero,state.bed)
        hillslope_erosion = tf.where(ero_u,hillslope_erosion + ero,hillslope_erosion)
        hillslope_erate = tf.where(ero_u,ero/state.dt,hillslope_erate)
        sed = tf.where(ero_u,sed + ero,sed)
        
        ########## v-points ################
        
        fac = 1.0 - tf.square(tf.divide(vp_dbdy,sc));
        fac = tf.where(fac  < 1.0e-4, tf.ones_like(fac)*1.0e-4, fac)
        sdiff = Ke/fac;
        ero = sdiff*vp_dbdy*state.dt/state.dx*tf.sqrt(1.0+tf.square(bslope));

        maxero = tf.roll(state.bed,1,0)-state.bed-sc*state.dx;
        maxero = tf.where(maxero  < 0.0, tf.zeros_like(maxero), maxero);
        ero = tf.where(tf.logical_and(ero < -maxero, ero < 0),-maxero, ero)  
        ero = tf.where(state.thk > 5.0, tf.zeros_like(ero), ero)
        csero = tf.roll(ero,-1,0);
        ero_l = csero<0.0;
        state.bed = tf.where(ero_l,state.bed + csero,state.bed)
        hillslope_erosion = tf.where(ero_l, hillslope_erosion - csero, hillslope_erosion)
        hillslope_erate = tf.where(ero_l, -csero/state.dt, hillslope_erate)
        sed = tf.where(ero_l ,sed-csero, sed)

        maxero = state.bed-tf.roll(state.bed,1,0)-sc*state.dx;
        maxero = tf.where(maxero  < 0.0, tf.zeros_like(maxero), maxero);
        ero = tf.where(tf.logical_and(ero > maxero, ero > 0),maxero, ero)
        ero = tf.where(state.thk > 5.0, tf.zeros_like(ero), ero)
        ero_u = ero>0;
        state.bed = tf.where(ero_u,state.bed - ero,state.bed)
        hillslope_erosion = tf.where(ero_u, hillslope_erosion + ero, hillslope_erosion)
        hillslope_erate = tf.where(ero_u, ero/state.dt, hillslope_erate)
        sed = tf.where(ero_u, sed+ero, sed)
        
        
        ##### hillslope sediment transport ##############
        
        dH =  tf.zeros_like(state.topg)

        # /*h-points for horizontal transport*/

        fac = 1.0 - tf.square(tf.divide(hp_dtdx,sc));
        fac = tf.where(fac  < 0.001, tf.ones_like(fac)*0.001, fac)
        sdiff = Ks/fac;       
        dHs = -sdiff*hp_dtdx*state.dt/state.dx;
        dHs_neg = dHs<0;
        sed_neg = sed <= 0.0;
        id1 = dHs_neg & sed_neg;
        dHs = tf.where(id1, tf.zeros_like(dHs),dHs)
        dHs_smaller_sed_neg = dHs < -sed;
        id2 = dHs_neg & dHs_smaller_sed_neg;
        dHs = tf.where(id2, -sed, dHs)
        sed_left = tf.roll(sed,-1,1);
        sed_left_neg = sed_left <= 0;
        id3 = ~dHs_neg & sed_left_neg;
        dHs = tf.where(id3,tf.zeros_like(dHs),dHs)
        dHs_larger_sed_left = dHs > sed_left;
        id4 = ~sed_left_neg & dHs_larger_sed_left;
        dHs = tf.where(id4, sed_left, dHs)
        dH = dH + dHs;
        dH = dH - tf.roll(dHs, 1, 1)

        # /*v-points*/
        fac = 1.0 - tf.square(tf.divide(vp_dtdy,sc));
        fac = tf.where(fac  < 0.001, tf.ones_like(fac)*0.001, fac)
        sdiff = Ks/fac;
        dHs = -sdiff*vp_dtdy*state.dt/state.dx;
        dHs_neg = dHs < 0.0;
        sed_neg = sed <= 0.0;
        id1 = dHs_neg & sed_neg;
        dHs = tf.where(id1, tf.zeros_like(dHs),dHs)
        dHs_smaller_sed_neg = dHs < -sed;
        id2 = dHs_neg & dHs_smaller_sed_neg;
        dHs = tf.where(id2, -sed, dHs)
        sed_down = tf.roll(sed,1,0)
        sed_down_neg = sed_down <= 0;
        id3 = ~dHs_neg & sed_down_neg;
        dHs = tf.where(id3, tf.zeros_like(dHs), dHs)
        dHs_larger_sed_down = dHs > sed_down;
        id4 = ~sed_down_neg & dHs_larger_sed_down;
        dHs = tf.where(id4, sed_down, dHs)
        dH = dH + dHs;
        dH = dH - tf.roll(dHs,-1,0);

        state.sed = sed + dH;
        state.hillslope_erosion = hillslope_erosion
        state.hillslope_erate = hillslope_erate
        # mean_dHs = mean_dHs + tf.reduce_sum(tf.abs(dH));
        # mean_dHs = tf.divide(mean_dHs,nc)*2.0 
        # mean_dHs_hillslope = mean_dHs;

        # state.topg = state.topg + dH
        state.hillslope = dH

        
        
       
        #################################################
        
        state.tlast_hillslope.assign(state.t)

        state.tcomp_hillslope[-1] -= time.time()
        state.tcomp_hillslope[-1] *= -1
        
def finalize(params, state):
    pass