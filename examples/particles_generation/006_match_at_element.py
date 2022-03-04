import json
import numpy as np

import xpart as xp
import xtrack as xt
import xobjects as xo

ctx = xo.context_default

# Load machine model (from pymask)
filename = ('../../../xtrack/test_data/lhc_no_bb/line_and_particle.json')
with open(filename, 'r') as fid:
    input_data = json.load(fid)
tracker = xt.Tracker(_context=ctx, line=xt.Line.from_dict(input_data['line']),
                     reset_s_at_end_turn=False)
assert not tracker.iscollective
tracker.line.particle_ref = xp.Particles.from_dict(input_data['particle'])

r_sigma = 1
theta = np.linspace(0, 2*np.pi, 1000)

at_element = 'ip2'
particles = xp.build_particles(tracker=tracker,
                   x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                   scale_with_transverse_norm_emitt=(2.5e-6, 2.5e-6),
                   at_element=at_element)

tw = tracker.twiss(at_elements=[at_element])

assert np.isclose(
    np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)

tracker.track(particles, turn_by_turn_monitor='ONE_TURN_EBE')
mon = tracker.record_last_track
i_ele_start = tracker.line.element_names.index(at_element)
assert np.all(mon.at_element[:, :i_ele_start] == 0)
assert np.all(mon.at_element[:, i_ele_start] == i_ele_start)
assert np.all(mon.at_element[:, -1] == len(tracker.line.element_names) -1)

tw0 = tracker.twiss(at_elements=[0])
assert np.isclose(
    np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)

# Check multiple turns
at_element = 'ip2'
particles = xp.build_particles(tracker=tracker,
                   x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                   scale_with_transverse_norm_emitt=(2.5e-6, 2.5e-6),
                   at_element=at_element)

tw = tracker.twiss(at_elements=[at_element])

assert np.isclose(
    np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)

tracker.track(particles, num_turns=3)

tw0 = tracker.twiss(at_elements=[0])
assert np.isclose(
    np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)
assert np.all(particles.at_turn==3)
assert np.allclose(particles.s, 3*tracker.line.get_length(), rtol=0, atol=1e-7)

# Check collective case
line_w_collective = xt.Line.from_dict(input_data['line'])
for ip in range(8):
    line_w_collective.element_dict[f'ip{ip+1}'].iscollective = True
tracker = xt.Tracker(_context=ctx, line=line_w_collective,
                     reset_s_at_end_turn=False)
assert tracker.iscollective
tracker.line.particle_ref = xp.Particles.from_dict(input_data['particle'])
assert len(tracker._parts) == 16


at_element = 'ip2'
particles = xp.build_particles(tracker=tracker,
                   x_norm=r_sigma*np.cos(theta), px_norm=r_sigma*np.sin(theta),
                   scale_with_transverse_norm_emitt=(2.5e-6, 2.5e-6),
                   at_element=at_element)

tw = tracker.twiss(at_elements=[at_element])

assert np.isclose(
    np.sqrt(tw['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=1e-3, atol=0)

tracker.track(particles, num_turns=3)

tw0 = tracker.twiss(at_elements=[0])
assert np.isclose(
    np.sqrt(tw0['betx'][0]*2.5e-6/particles.beta0[0]/particles.gamma0[0]),
    np.max(np.abs(particles.x - np.mean(particles.x))), rtol=2e-3, atol=0)
assert np.all(particles.at_turn==3)
assert np.allclose(particles.s, 3*tracker.line.get_length(), rtol=0, atol=1e-7)