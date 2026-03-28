"""Shared helpers for experiment configuration handling."""

import hashlib


def stable_seed(base_seed, *parts):
    """Derive a reproducible 31-bit seed from experiment identifiers."""
    payload = "::".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


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
