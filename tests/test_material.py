import math
from material import load_material, tpu90a_impact, PRONY_TPU_G


def test_yaml_and_override():
    m = load_material('TPU_90A')
    assert (m['E'], m['nu'], m['rho']) == (5.3, 0.45, 1.24e-9)
    mz = load_material('TPU_90A', E=4.4)
    assert mz['E'] == 4.4 and math.isclose(mz['G'], 4.4 / 2.9)


def test_impact_default():
    m = tpu90a_impact()
    assert math.isclose(sum(PRONY_TPU_G), 0.3429587, rel_tol=1e-6)
    assert math.isclose(m['E_inf'], 4.4)
    assert math.isclose(m['E'], 6.6967, rel_tol=1e-4)  # E0 = 4.4 / 0.657
