# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:50:36 2024

@author: mey
"""

#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from igm.modules.utils import *



def params(parser):
    parser.add_argument(
        "--uplift_rate",
        type=float,
        default=0.001,
        help="Uplift rate in m/yr.",
    )
    parser.add_argument(
        "-uplift_update_freq",
        type=float,
        default=1.0,
        help="Apply uplift only each X years.",
    )
   
def initialize(params, state):
    
    state.tcomp_uplift = []
    state.tlast_uplift = tf.Variable(params.time_start, dtype=tf.float32)
    

def update(params, state):
   
   

    if (state.t - state.tlast_uplift) >= params.uplift_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update uplift at time : " + str(state.t.numpy()))

        state.tcomp_uplift.append(time.time())

      
        # add the uplift to the topography 
       
        state.topg = state.topg + (state.t - state.tlast_uplift) * params.uplift_rate
       

        state.tlast_uplift.assign(state.t)

        state.tcomp_uplift[-1] -= time.time()
        state.tcomp_uplift[-1] *= -1


def finalize(params, state):
    pass
