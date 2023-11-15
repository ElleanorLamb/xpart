# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .general import _pkg_root, _print
from .particles import Particles, ParticlesBase, pmass

from .build_particles import build_particles
from .matched_gaussian import (generate_matched_gaussian_bunch,
                               generate_matched_gaussian_beam,
                               split_scheme)

from .transverse_generators import generate_2D_polar_grid
from .transverse_generators import generate_2D_uniform_circular_sector
from .transverse_generators import generate_2D_pencil
from .transverse_generators import generate_2D_pencil_with_absolute_cut
from .transverse_generators import generate_2D_gaussian

from .longitudinal import generate_longitudinal_coordinates
from .longitudinal.generate_longitudinal import _characterize_line

from .constants import PROTON_MASS_EV, ELECTRON_MASS_EV

from .monitors import PhaseMonitor

from ._version import __version__

def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    xp.Particles = pp.PyHtXtParticles

def disable_pyheadtail_interface():
    import xpart as xp
    xp.Particles = xp.particles.Particles
