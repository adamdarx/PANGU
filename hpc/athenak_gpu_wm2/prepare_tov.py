"""Translate the existing magnetized-collapse decks, retaining the physical mesh."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent

def read_deck(text):
    result = {}; section = None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            section = line[1:-1]; result[section] = {}
        elif '=' in line and section:
            key, value = line.split('=', 1); result[section][key.strip()] = value.strip()
    return result

records = []
for n in (64, 96):
    path = ROOT/'source-records/tov/inputs'/('collapse_n%d.in' % n)
    source = read_deck(path.read_text())
    mesh = source['parthenon/mesh']; nr = source['numerical_relativity']
    assert int(mesh['nx1']) == n and source['problem']['rhoc'] == '0.007993'
    blocks = [('job', {'basename': 'athenak_tov_collapse_n%d' % n}),
              ('mesh', {k: v for k, v in mesh.items() if k != 'refinement'}),
              ('meshblock', source['parthenon/meshblock']),
              ('mesh_refinement', {'refinement': 'static'}),
              ('time', {'evolution': 'dynamic', 'integrator': 'rk2', 'cfl_number': nr['cfl'],
                        'tlim': '100', 'nlim': '-1', 'ndiag': '20'}),
              ('coord', {'general_rel': 'true', 'excise': 'false', 'm': '1', 'a': '0'}),
              ('mhd', {'eos': 'ideal', 'dyn_eos': 'ideal', 'dyn_error': 'reset_floor',
                       'reconstruct': 'plm', 'rsolver': 'hlle', 'dfloor': '1e-16',
                       'tfloor': '1e-16', 'dthreshold': '1.02', 'gamma': '1.999999999999',
                       'gamma_max': '1000', 'max_bsq': '1e12', 'dyn_scratch': '1',
                       'fofc': 'true', 'enforce_maximum': 'false'}),
              ('adm', {}),
              ('z4c', {'diss': '0', 'damp_kappa1': '0', 'damp_kappa2': '0',
                       'chi_psi_power': '-4', 'chi_div_floor': '-1000', 'lapse_oplog': '2',
                       'lapse_advect': '1', 'slow_start_lapse': 'false',
                       'shift_Gamma': '1', 'shift_eta': '2', 'shift_advect': '1'}),
              ('problem', dict(source['problem'], isotropic='true', minkowski='false',
                               p_pert='0', user_hist='true')),
              ('fastflow', {'num_horizons': '1', 'ntheta': '40', 'lmax': '12',
                            'initial_radius_0': '2', 'start_time_0': '0', 'stop_time_0': '1e30',
                            'flow_iterations_0': '300', 'hmean_tol_0': '1e-5',
                            'mass_tol_0': '1e-5', 'output_grid': 'false'})]
    for key in ('write_final_csv',): blocks[-2][1].pop(key, None)
    for axis in (1, 2, 3):
        blocks[1][1]['ix%d_bc' % axis] = 'diode'
        blocks[1][1]['ox%d_bc' % axis] = 'diode'
    for name, values in source.items():
        if name.startswith('parthenon/static_refinement'):
            blocks.append(('refined_region'+name.split('refinement')[1], values))
    blocks += [('output1', {'file_type': 'hst', 'dt': '0.1', 'data_format': '%20.15e'})]
    for i, variable in enumerate(('mhd_w_bcc', 'mhd_u', 'z4c', 'adm', 'mhd_divb', 'con_H', 'con_M'), 2):
        blocks.append(('output%d' % i, {'file_type': 'bin', 'variable': variable, 'dt': '5'}))
    blocks.append(('output9', {'file_type': 'rst', 'dt': '25'}))
    text = '# Native AthenaK GPU reference: original magnetized TOV collapse, target100.\n'
    text += '# SMR requires even nghost: native nghost4 uses FD6; PANGU uses FD4.\n'
    text += '# Atmosphere reset and diode/NR outflow implementations differ; no performance claim.\n'
    text += '\n'.join('<'+name+'>\n'+'\n'.join(k+' = '+v for k,v in values.items())+'\n' for name,values in blocks)
    output = ROOT/'inputs'/('tov_n%d.athinput' % n); output.parent.mkdir(exist_ok=True)
    output.write_text(text)
    records.append(dict(resolution=n, input=str(output), sha256=hashlib.sha256(text.encode()).hexdigest(),
                        pangu_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        target=100, meshblock_cells=n//4))
(ROOT/'TOV_INPUTS.json').write_text(json.dumps(dict(cases=records,
    differences=['PANGU FD4 versus native AthenaK FD6 (even ghost count required for SMR)',
                 'Independent atmosphere reset: tfloor1e-16 gives pressure1e-32 at density floor only',
                 'Native diode fluid and radiative spacetime boundaries versus PANGU nr_outflow',
                 'Native ideal-gas gamma differs from2 by1e-12 to avoid endpoint remapping',
                 'Native FastFlow horizon diagnostics require independent quality screening']),indent=2)+'\n')
