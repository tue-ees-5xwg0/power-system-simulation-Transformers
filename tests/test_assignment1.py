# tests/test_get_figure.py
import matplotlib
matplotlib.use("Agg")                 # run head-less (CI-friendly)
import matplotlib.pyplot as plt
from power_system_simulation.assignment1 import GraphProcessor  
import pytest
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown"
)
# ---------------------------------------------------------------------------
# single, self-contained test for GraphProcessor.get_figure
# ---------------------------------------------------------------------------
def test_get_figure_basic():
    """Smoke-test + basic content check for the visualisation helper."""

    # --- fixed toy network --------------------------------------------------
    vertex_ids            = [0, 2, 4, 6, 10]
    edge_ids              = [1, 3, 5, 7, 8, 9]
    edge_vertex_pairs     = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled          = [True, True, True, False, False, True]
    source_vertex_id      = 10

    gp = GraphProcessor(
        vertex_ids,
        edge_ids,
        edge_vertex_pairs,
        edge_enabled,
        source_vertex_id,
    )

    # --- call the method under test ----------------------------------------
    fig = gp.get_figure(seed=0, figsize=(4, 3))

    # 1) returns a Matplotlib Figure
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)

    # 2) every edge ID (enabled or disabled) appears as a text label
    found_labels = {t.get_text() for t in fig.axes[0].texts}
    assert set(map(str, edge_ids)).issubset(found_labels)

 


