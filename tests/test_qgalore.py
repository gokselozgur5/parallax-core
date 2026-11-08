import numpy as np
from parallax.qgalore import QGaLoreProjector


def test_qgalore_projector_init():
    proj = QGaLoreProjector(rank=4, update_interval=50)
    assert proj.rank == 4
    assert proj.update_interval == 50
    assert proj.step_count == 0
    assert proj.U is None
