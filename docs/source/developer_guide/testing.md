---
myst:
  html_meta:
    description: "How to run the tests of sigmaepsilon.solid.fourier with pytest and create a coverage report."
---

# Testing and coverage

The following command runs all tests and creates a html report in a folder named `htmlcov`
(the settings are governed by the `.coveragerc` file):

```shell
python -m pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.solid.fourier
```

Alternatively, you can use `uv` to test the package (requires the `test` dependency group):

```shell
uv run pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.solid.fourier
```
