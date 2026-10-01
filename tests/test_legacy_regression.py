"""Legacy linear pipeline (optBeam.ipynb example 3fz) must match the frozen reference."""
import json
from functools import partial

import numpy as np
import pytest
import torch

from legacy_setup import ROOT, VAE_PATH, load_vae, build_3fz, volume_max

REF = json.load(open(ROOT / 'tests' / 'data' / 'legacy_3fz.json'))
pytestmark = pytest.mark.skipif(not VAE_PATH.exists(), reason='savedNet checkpoint missing')


@pytest.fixture(scope='module')
def net():
    return load_vae()


def test_section_model_matches_notebook(net):
    from sectionModel import propPredict, shapePredict
    x = torch.tensor(REF['probe_x'], dtype=torch.float64)
    np.testing.assert_allclose(propPredict(x, net).detach().numpy(), REF['propPredict'], rtol=1e-12)
    cp, th = shapePredict(x, net)
    np.testing.assert_allclose(cp.detach().numpy(), REF['shapePredict_cp'], rtol=1e-12)
    np.testing.assert_allclose(th.numpy(), REF['shapePredict_theta'], rtol=1e-12)


@pytest.mark.parametrize('section', ['circle', 'z'])
def test_objective_and_gradient(net, section):
    from sectionModel import propPredict
    opt, x0 = build_3fz(section, partial(propPredict, net=net))
    J, dJ = opt.objectiveSE(x0, Jac=True)
    vc, dvc = opt.VolumeCon(x0, con_value=0.5 * volume_max(opt))
    ref = REF[section]
    assert opt.frameFE.numEle == ref['nEle']
    np.testing.assert_allclose(np.asarray(J).item(), ref['J'], rtol=1e-10)
    np.testing.assert_allclose(dJ.reshape(-1), ref['dJ'], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(vc.item(), ref['vc'], rtol=1e-10)
    np.testing.assert_allclose(dvc.reshape(-1), ref['dvc'], rtol=1e-8, atol=1e-12)


def test_short_mma_history():
    opt, x0 = build_3fz('circle')
    setup = {'objective': 'SE', 'constraints': {'Volume': 0.5 * volume_max(opt)}}
    opt.optimizerRun(setup, x0.copy(), algorithm='MMA',
                     options={'maxiter': 10, 'disp': False, 'move_limit': 0.5, 'maxfun': 500,
                              'kkttol': 1e-5, 'miniter': 10})
    hist = [np.asarray(v).item() for v in opt.objective_values]
    np.testing.assert_allclose(hist, REF['mma_circle']['obj_hist'], rtol=1e-6)
