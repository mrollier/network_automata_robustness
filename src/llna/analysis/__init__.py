"""Analysis toolbox for LLNA dynamics.

Split by concern (2026-07 cleanup); every public name remains importable
directly from llna.analysis, so notebook imports are unaffected.
"""

from llna.analysis._encoding import _interval_encoding
from llna.analysis.derrida import (
    calculate_derrida_coefficient,
    derrida_map_analytical,
    derrida_spline_and_roots,
    get_derrida_arrays,
)
from llna.analysis.eca import eca_has_constantJ, jacobian_ECA
from llna.analysis.ensemble import (
    init_config_with_dens,
    median_and_percentiles_over_ensemble,
    switch_life_to_higher_node_property_values,
)
from llna.analysis.lyapunov import (
    calculate_Yt,
    defect_diameter,
    jacobian,
    lyapunov_spectrum,
    lyapunov_spectrum_analytical,
)
from llna.analysis.meanfield import mean_field_dens_propagation, mean_field_slope
from llna.analysis.sensitivity import (
    average_metric_over_degrees,
    boolean_sens,
    hamming_weight,
    id_sensitivity_naive,
    nbh_sensitivity_naive,
)

__all__ = [
    "average_metric_over_degrees",
    "boolean_sens",
    "calculate_Yt",
    "calculate_derrida_coefficient",
    "defect_diameter",
    "derrida_map_analytical",
    "derrida_spline_and_roots",
    "eca_has_constantJ",
    "get_derrida_arrays",
    "hamming_weight",
    "id_sensitivity_naive",
    "init_config_with_dens",
    "jacobian",
    "jacobian_ECA",
    "lyapunov_spectrum",
    "lyapunov_spectrum_analytical",
    "mean_field_dens_propagation",
    "mean_field_slope",
    "median_and_percentiles_over_ensemble",
    "nbh_sensitivity_naive",
    "switch_life_to_higher_node_property_values",
]
