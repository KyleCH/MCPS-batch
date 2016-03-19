#!/usr/bin/env python3

import numpy as np
from pint import UnitRegistry
import scipy.constants as const
ureg = UnitRegistry()
Q_ = ureg.Quantity

# ======================================================================
# Set parameters.
# ======================================================================

# Number of iterations.
n = int(1e3)

# ----------------------------------------------------------------------
# Loopped parameters.
# ----------------------------------------------------------------------

# Initial time.
t0 = Q_(np.asarray([0.]), 'second')

# Initial position.
r0 = Q_(np.asarray([[0., 0., 0.]]), 'meter')

# Initial velocity unit vector.
vhat0 = np.asarray([[0., 0., 1.]])

# Charge.
q = Q_(np.asarray([1.]), 'e')

# Mass.
m = [Q_(const.value('proton mass'), const.unit('proton mass'))]

# Initial kinetic energy.
T0 = Q_(np.asarray([6.]), 'MeV')

# Y-component of magnetic field.
By = Q_(np.asarray([5.]), 'mT')

# Step size.
h = Q_(np.asarray([1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21]),
    'second')
