#!/usr/bin/env python3
"""Candidate restart inventory using text XDMF; PANGU validates the HDF5 data.

No h5py/numpy/HDF5 tools required. XML is emitted at the end of HDF5 output,
but is NOT proof of a complete HDF5 file. Always run PANGU's restart preflight.
"""
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def candidates(directory):
    result = []
    for path in Path(directory).glob('gr_torus_sane.restart.*.rhdf'):
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError('empty restart')
            with path.open('rb') as stream:
                if stream.read(8) != b'\x89HDF\r\n\x1a\n':
                    raise ValueError('missing HDF5 signature')
            tree = ET.parse(str(path) + '.xdmf')
            times = {float(node.attrib['Value']) for node in tree.iter('Time')}
            if len(times) != 1:
                raise ValueError('missing/inconsistent XDMF time')
            time = times.pop()
            if not math.isfinite(time) or time < 0:
                raise ValueError('invalid time')
            cycles = {int(node.attrib['Value']) for node in tree.iter('Information')
                      if node.attrib.get('Name') == 'Cycle'}
            if len(cycles) != 1 or min(cycles) < 0:
                raise ValueError('missing/inconsistent cycle')
            result.append((time, cycles.pop(), str(path.resolve())))
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            print(f'Skip {path.name}: {error}', file=sys.stderr)
    return sorted(result, key=lambda item: (item[1], item[0], item[2]), reverse=True)


if __name__ == '__main__':
    for time, _, path in candidates(sys.argv[1]):
        print(f'{time:.17g}\t{path}')
