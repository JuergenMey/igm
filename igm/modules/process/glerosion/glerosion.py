#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--glerosion_cst",
        type=float,
        default=2.7 * 10 ** (-7),
        help="Erosion multiplicative factor, here taken from Herman, F. et al. \
              Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
    )
    parser.add_argument(
        "--glerosion_exp",
        type=float,
        default=2,
        help="Erosion exponent factor, here taken from Herman, F. et al. \
               Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
    )
    parser.add_argument(
        "--glerosion_update_freq",
        type=float,
        default=1.0,
        help="Update the erosion only each X years (Default: 100)",
    )
    parser.add_argument(
        "--glerosion_sediment_fac",
        type=float,
        default=2.0,
        help="Erosion efficiency of sediments relative to bedrock.",
    )


def initialize(params, state):
    state.glerosion = tf.zeros_like(state.topg)
    state.tcomp_glerosion = []
    state.tlast_erosion = tf.Variable(params.time_start, dtype=tf.float32)
    if not hasattr(state,'sed'):
        state.sed = tf.zeros_like(state.topg)


def update(params, state):
    if (state.t - state.tlast_erosion) >= params.glerosion_update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "update topg_glacial_erosion at time : " + str(state.t.numpy())
            )

        state.tcomp_glerosion.append(time.time())

        velbase_mag = getmag(state.U[0], state.V[0])
        
        sed = tf.where(state.sed > 0, tf.ones_like(state.sed)*params.glerosion_sediment_fac, 1)
        # apply erosion law, erosion rate is proportional to a power of basal sliding speed
        dtopgdt = params.glerosion_cst * (velbase_mag**params.glerosion_exp)*sed
        # when gflex and glerosion are used, the erosion has to be applied to topg0 (done in the gflex module)
        state.erosion = (state.t - state.tlast_erosion) * dtopgdt  
        state.glerosion += (state.t - state.tlast_erosion) * dtopgdt 
        state.topg = state.topg - (state.t - state.tlast_erosion) * dtopgdt
        state.sed = state.sed - (state.t - state.tlast_erosion) * dtopgdt
        state.sed = tf.where(state.sed < 0, tf.zeros_like(state.sed), state.sed)

        # THIS WORK ONLY FOR GROUNDED ICE, TO BE ADAPTED FOR FLOATING ICE
        state.usurf = state.topg + state.thk

        state.tlast_erosion.assign(state.t)

        state.tcomp_glerosion[-1] -= time.time()
        state.tcomp_glerosion[-1] *= -1


def finalize(params, state):
    pass
