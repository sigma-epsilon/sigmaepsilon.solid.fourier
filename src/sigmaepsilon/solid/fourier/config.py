__all__ = ["Config"]


class Config:
    # The default method of numerical integration of load coefficients. Either
    # "quadrature" (Gauss-Legendre) or "mc" (Monte Carlo).
    INTEGRATION_METHOD: str = "quadrature"
    # Number of Gauss-Legendre nodes per panel of a composite rule.
    QUADRATURE_ORDER: int = 8
    # Number of panels per half-wavelength of the highest harmonic involved.
    QUADRATURE_PANELS_PER_HALF_WAVE: int = 1
    # Number of integration points processed at once.
    QUADRATURE_BATCH_SIZE: int = 5000
    NUM_MC_SAMPLES_PLATE: int = 100000
    MC_BATCH_SIZE_PLATE: int = 5000
    NUM_MC_SAMPLES_BEAM: int = 50000
    MC_BATCH_SIZE_BEAM: int = 5000
    # The default seed of Monte Carlo integration. None means fresh entropy.
    MC_SEED: int | None = None
