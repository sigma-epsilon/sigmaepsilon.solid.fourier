"""
Global settings of the library, validated when they are assigned.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from .enums import IntegrationMethod, PostprocMethod

__all__ = ["Config", "FourierConfig"]


class FourierConfig(BaseModel):
    """
    Global settings of the library. The settings are validated when they are
    assigned, and invalid values raise a :class:`ValueError`. Use the instance
    :data:`Config` of the library instead of creating a new one.

    Examples
    --------
    >>> from sigmaepsilon.solid.fourier.config import Config
    >>> Config.POSTPROC_METHOD = "numba"
    >>> Config.POSTPROC_METHOD
    <PostprocMethod.NUMBA: 'numba'>
    >>> Config.POSTPROC_METHOD = "blas"
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # The default method of numerical integration of load coefficients. Either
    # "quadrature" (Gauss-Legendre) or "mc" (Monte Carlo).
    INTEGRATION_METHOD: IntegrationMethod = IntegrationMethod.QUADRATURE
    # Number of Gauss-Legendre nodes per panel of a composite rule.
    QUADRATURE_ORDER: PositiveInt = 8
    # Number of panels per half-wavelength of the highest harmonic involved.
    QUADRATURE_PANELS_PER_HALF_WAVE: PositiveInt = 1
    # Number of integration points processed at once.
    QUADRATURE_BATCH_SIZE: PositiveInt = 5000
    NUM_MC_SAMPLES_PLATE: PositiveInt = 100000
    MC_BATCH_SIZE_PLATE: PositiveInt = 5000
    NUM_MC_SAMPLES_BEAM: PositiveInt = 50000
    MC_BATCH_SIZE_BEAM: PositiveInt = 5000
    # The default seed of Monte Carlo integration. None means fresh entropy.
    MC_SEED: NonNegativeInt | None = None
    # The method of postprocessing plates. Either "blas" (dense matrix products)
    # or "numba" (the reference kernels).
    POSTPROC_METHOD: PostprocMethod = PostprocMethod.BLAS
    # The number of floats in the buffers of the "blas" postprocessing.
    POSTPROC_BUFFER_SIZE: PositiveInt = 2**22

    @field_validator("INTEGRATION_METHOD", mode="before")
    @classmethod
    def _validate_integration_method(cls, value) -> IntegrationMethod:
        return IntegrationMethod.from_value(value)

    @field_validator("POSTPROC_METHOD", mode="before")
    @classmethod
    def _validate_postproc_method(cls, value) -> PostprocMethod:
        return PostprocMethod.from_value(value)


Config = FourierConfig()
"""The global settings of the library, see :class:`FourierConfig`."""
