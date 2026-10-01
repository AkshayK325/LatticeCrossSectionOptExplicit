"""Rebuild the optBeam.ipynb example '3fz' setup (cells 2, 19, 21) outside the notebook."""
import pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAE_PATH = ROOT / 'savedNet' / 'splineNet50k_n_cp_5_2x512nfeatures_14.pth'


def load_vae():
    from vaeNet import VariationalAutoencoderModel
    ck = torch.load(VAE_PATH, map_location='cpu', weights_only=False)
    net = VariationalAutoencoderModel(ck['input_data'], ck['scaling_info'], ck['identifiers'],
                                      ck['architecture_config'], useCPU=True)
    net.load_model_from_file(ck['model_state'])
    return net


def build_3fz(section, prop_predict=None):
    """Return (optFrame, x0) exactly as notebook cells 19/21 build them for '3fz', symType 'Independent'."""
    from examples import getExample
    from frameFE3 import FrameFE
    from optimizeFrame3 import OptimizeFrame

    H = 0.5 * 25.4
    params = {'base': H, 'height': H, 'phi': 0.0, 'delta': 0.0, 'nx': 3, 'ny': 3, 'nz': 1,
              'Name': 'C4', 'Shape': 'Square'}
    exampleInfo, nodeXY, connectivity, bc, radiiNodIndex, radiiElemIndex = getExample('3fz', params)
    meshSetting = {'nodeXY': nodeXY, 'connectivity': connectivity, 'numElemsPerBeam': 4, 'elemSize': 3.0,
                   'ElemType': 'TS', 'radiiElemIndex': radiiElemIndex, 'radiiNodIndex': radiiNodIndex,
                   'numUnitLatElem': max(radiiElemIndex) + 1, 'numUnitLatNode': max(radiiNodIndex) + 1}
    analysis = {'Type': 'Linear', 'solver': 'pypardiso', 'matrixType': 'Sparse', 'Section': 'circle'}
    frameFE = FrameFE(meshSetting, bc, analysis)

    rng = np.random.default_rng(4)
    if section == 'circle':
        x0 = rng.uniform(size=(connectivity.shape[0], 1))
        min_x, max_x = 1e-4, 1.5
    else:
        x0 = rng.uniform(size=(connectivity.shape[0], 4))
        min_x, max_x = 0.0, 1.0
    varSetup = {'Section': section, 'min_x': min_x, 'max_x': max_x,
                'symArray': np.arange(connectivity.shape[0]).tolist(), 'varType': 'Element',
                'propPredict': prop_predict}
    opt = OptimizeFrame(frameFE, {'E': torch.tensor([17.0])}, exampleInfo, varSetup)
    return opt, x0.reshape((-1, 1))


def volume_max(opt):
    return opt.frameFE.eleLength.sum().detach() * 8.0
