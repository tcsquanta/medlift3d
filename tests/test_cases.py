"""Case I/O and CT table removal."""
import numpy as np

from medlift3d.data.cases import case_from_phantom, load_case, remove_table, save_case
from medlift3d.phantom import NoduleSpec, make_chest_phantom


def test_table_removed_body_kept():
    ph = make_chest_phantom(spacing=(4.0, 4.0, 4.0), fov_mm=(360.0, 320.0, 280.0), n_vessels=4, seed=1,
                            nodules=[NoduleSpec((-70.0, 0.0, 20.0), (6.0, 6.0, 6.0), 20.0)])
    hu = ph.hu.copy()
    y = ph.vgeom.axes()[1]
    slab = (y > 125.0) & (y < 135.0)                           # table 10 mm below the back, air gap in between
    hu[:, slab, 10:-10] = 200.0
    cleaned = remove_table(hu)
    assert (cleaned[:, slab, 10:-10] == -1000.0).all()
    body = ph.body_mask
    assert np.allclose(cleaned[body], hu[body])               # lungs, nodule, ribs untouched


def test_case_roundtrip(tmp_path):
    ph = make_chest_phantom(spacing=(4.0, 4.0, 4.0), n_vessels=4, seed=2,
                            nodules=[NoduleSpec((-70.0, 0.0, 20.0), (6.0, 6.0, 6.0), 20.0)])
    c = case_from_phantom(ph, "p")
    save_case(c, str(tmp_path / "p.npz"))
    d = load_case(str(tmp_path / "p.npz"))
    assert d.hu.shape == c.hu.shape and len(d.nodules) == 1
    assert np.array_equal(d.nodules[0].mask, c.nodules[0].mask)
    assert np.allclose(d.nodules[0].center_xyz, c.nodules[0].center_xyz)
