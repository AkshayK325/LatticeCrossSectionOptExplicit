"""Material data from config/materials.yaml (units mm / N / s / MPa / tonne)."""
import pathlib
import yaml

MATERIALS_YAML = pathlib.Path(__file__).resolve().parents[1] / 'config' / 'materials.yaml'

# 8-term generalized Maxwell (Prony) series for printed TPU, relaxation-fitted:
# Pelayo et al., Polymers 13(14):2365, 2021, doi:10.3390/polym13142365.  sum(g) = 0.34296
PRONY_TPU_G = (0.142920, 0.021163, 0.029420, 0.035212, 0.013225, 0.036500, 0.0008057, 0.063713)
PRONY_TPU_TAU = (0.054063, 0.146140, 0.395040, 1.067800, 2.886600, 7.802800, 21.092000, 57.015000)


def load_material(name, E=None):
    """Return dict(E, nu, rho, G) for `name`. `E` overrides the YAML value (e.g. build-direction modulus)."""
    m = yaml.safe_load(open(MATERIALS_YAML))['materials'][name]
    E = float(m['E'] if E is None else E)
    nu = float(m['nu'])
    return {'E': E, 'nu': nu, 'rho': float(m['rho']), 'G': E / (2 * (1 + nu))}


def tpu90a_impact():
    """Project default (plan 3): TPU_90A, Z modulus 4.4 MPa taken as E_inf, so E0 = E_inf / (1 - sum g)."""
    m = load_material('TPU_90A', E=4.4)
    m['E_inf'] = m['E']
    m['E'] = m['E_inf'] / (1 - sum(PRONY_TPU_G))
    m['G'] = m['E'] / (2 * (1 + m['nu']))
    m['prony_g'], m['prony_tau'] = PRONY_TPU_G, PRONY_TPU_TAU
    return m


def epu45():
    """Project material (from 2026-10-01): Carbon EPU 45, visco-hyperelastic (config/materials.yaml EPU_45).
    'E', 'G' are LONG-TERM small-strain moduli (E_inf = 6 C10); 'E0', 'G0' instantaneous (explicit dt)."""
    m = yaml.safe_load(open(MATERIALS_YAML))['materials']['EPU_45']
    E_inf, nu, g = float(m['E']), float(m['nu']), [float(x) for x in m['prony_g']]
    E0 = E_inf / (1 - sum(g))
    return {'name': 'EPU_45', 'E': E_inf, 'nu': nu, 'rho': float(m['rho']), 'G': E_inf / (2 * (1 + nu)),
            'E0': E0, 'G0': E0 / (2 * (1 + nu)), 'yeoh': [float(c) for c in m['yeoh']],
            'prony_g': g, 'prony_tau': [float(t) for t in m['prony_tau']], 'tds': m['tds']}


project_material = epu45


def yeoh_uniaxial(lam, C):
    """Incompressible Yeoh uniaxial nominal stress P(lambda) [MPa] (force per undeformed area)."""
    x = lam * lam + 2 / lam - 3
    return 2 * (lam - lam**-2) * (C[0] + 2 * C[1] * x + 3 * C[2] * x * x)


def yeoh_energy(lam, C):
    """Strain energy density W(lambda) [MPa] for incompressible uniaxial stretch."""
    x = lam * lam + 2 / lam - 3
    return C[0] * x + C[1] * x**2 + C[2] * x**3
