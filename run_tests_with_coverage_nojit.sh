#!/bin/bash
export NUMBA_DISABLE_JIT=1
poetry run pytest --cov-report html --cov-config=.coveragerc_nojit --cov sigmaepsilon.solid.fourier
export NUMBA_DISABLE_JIT=0