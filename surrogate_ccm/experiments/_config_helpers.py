"""Shared helpers for experiment configuration handling."""


def get_system_kwargs(config, system_name):
    """Return constructor kwargs for a named dynamical system."""
    return dict(config.get(system_name, {}))


def collect_seccm_kwargs(seccm_cfg):
    """Extract supported SECCM kwargs from an experiment config block."""
    allowed = (
        "theiler_w",
        "adaptive_rho",
        "E_method",
        "convergence_filter",
        "convergence_threshold",
        "min_rho",
        "adaptive_rho_quantile",
        "iaaft_max_iter",
        "use_gpu",
    )
    return {key: seccm_cfg[key] for key in allowed if key in seccm_cfg}
