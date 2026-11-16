"""Oracles for unequal-grid error estimates and fail-closed evidence handling."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from compact_physical_acceptance import evaluate,richardson


class PhysicalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.base=Path(self.directory.name)
        self.source=self.base/'observed.json';self.source.write_text('{"physical_mass":1}')
        self.evidence=[dict(path=self.source.name,sha256=hashlib.sha256(self.source.read_bytes()).hexdigest())]
        self.protocol=dict(id='oracle',tolerance_origin='test budget',common={'mass_error':dict(max=.01)},cases={'tov':{},'bbh':{}},waveform_phase_additional_requirement=dict(max_radians=.1))

    def tearDown(self):self.directory.cleanup()

    def document(self,value):return dict(case='tov',metrics={'mass_error':dict(value=value,evidence=self.evidence)})

    def test_known_second_order_unequal_grids(self):
        h=np.array([1/64,1/96,1/128]);v=1.4+7*h*h
        result=richardson(h,v)
        self.assertAlmostEqual(result['observed_order'],2,places=8)
        self.assertAlmostEqual(result['finest_absolute_error'],7*h[-1]**2,places=12)

    def test_complex_fourth_order_trajectory(self):
        h=np.array([.3,.2,.15]);exact=np.exp(1j*np.linspace(0,1,10))
        error=(2+3j)*np.linspace(1,2,10)
        result=richardson(h,np.array([exact+error*x**4 for x in h]))
        self.assertAlmostEqual(result['observed_order'],4,places=8)
        self.assertAlmostEqual(result['finest_absolute_error'],np.linalg.norm(error*h[-1]**4),places=11)

    def test_nonmonotone_cannot_claim_order(self):
        with self.assertRaises(ValueError):richardson([.3,.2,.1],[1,2,1.5])

    def test_identical_values_do_not_imply_infinite_order(self):
        with self.assertRaises(ValueError):richardson([.3,.2,.1],[1,1,1])

    def test_verified_inside_budget_passes(self):
        self.assertEqual(evaluate(self.document(.001),self.protocol,self.base)['status'],'passed')

    def test_measured_failure_is_not_pending(self):
        self.assertEqual(evaluate(self.document(.02),self.protocol,self.base)['status'],'failed')

    def test_missing_measurement_cannot_pass(self):
        result=evaluate(dict(case='tov',metrics={}),self.protocol,self.base)
        self.assertEqual(result['status'],'pending');self.assertFalse(result['physical_acceptance'])

    def test_missing_source_cannot_pass(self):
        self.source.unlink();self.assertEqual(evaluate(self.document(0),self.protocol,self.base)['status'],'pending')

    def test_changed_source_cannot_pass(self):
        self.source.write_text('{"physical_mass":99}')
        self.assertEqual(evaluate(self.document(0),self.protocol,self.base)['status'],'pending')

    def test_nan_cannot_pass(self):
        self.assertEqual(evaluate(self.document(float('nan')),self.protocol,self.base)['status'],'failed')

    def test_negative_error_cannot_pass(self):
        self.assertEqual(evaluate(self.document(-1),self.protocol,self.base)['status'],'failed')

    def test_non_head_on_requires_phase(self):
        doc=self.document(0);doc.update(case='bbh',family='quasicircular')
        result=evaluate(doc,self.protocol,self.base)
        self.assertEqual(result['status'],'pending');self.assertIn('reference_phase_max_rad',result['pending'])


if __name__=='__main__':unittest.main()
