import unittest
from unittest import mock

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from sigmaepsilon.solid.fourier.config import Config, FourierConfig
from sigmaepsilon.solid.fourier.enums import IntegrationMethod, PostprocMethod


class TestConfig(SigmaEpsilonTestCase):
    def test_defaults(self):
        config = FourierConfig()
        self.assertIs(config.INTEGRATION_METHOD, IntegrationMethod.QUADRATURE)
        self.assertIs(config.POSTPROC_METHOD, PostprocMethod.BLAS)
        self.assertEqual(config.QUADRATURE_ORDER, 8)
        self.assertEqual(config.QUADRATURE_PANELS_PER_HALF_WAVE, 1)
        self.assertEqual(config.QUADRATURE_BATCH_SIZE, 5000)
        self.assertEqual(config.NUM_MC_SAMPLES_PLATE, 100000)
        self.assertEqual(config.MC_BATCH_SIZE_PLATE, 5000)
        self.assertEqual(config.NUM_MC_SAMPLES_BEAM, 50000)
        self.assertEqual(config.MC_BATCH_SIZE_BEAM, 5000)
        self.assertIsNone(config.MC_SEED)
        self.assertEqual(config.POSTPROC_BUFFER_SIZE, 2**22)

    def test_methods_from_strings(self):
        config = FourierConfig()
        config.POSTPROC_METHOD = "NUMBA"
        self.assertIs(config.POSTPROC_METHOD, PostprocMethod.NUMBA)
        config.INTEGRATION_METHOD = "monte_carlo"
        self.assertIs(config.INTEGRATION_METHOD, IntegrationMethod.MONTE_CARLO)
        config.INTEGRATION_METHOD = IntegrationMethod.QUADRATURE
        self.assertIs(config.INTEGRATION_METHOD, IntegrationMethod.QUADRATURE)
        config = FourierConfig(POSTPROC_METHOD="numba", INTEGRATION_METHOD="mc")
        self.assertIs(config.POSTPROC_METHOD, PostprocMethod.NUMBA)
        self.assertIs(config.INTEGRATION_METHOD, IntegrationMethod.MONTE_CARLO)

    def test_valid_values(self):
        config = FourierConfig()
        config.MC_SEED = 0
        self.assertEqual(config.MC_SEED, 0)
        config.MC_SEED = None
        self.assertIsNone(config.MC_SEED)
        config.QUADRATURE_ORDER = 4
        self.assertEqual(config.QUADRATURE_ORDER, 4)

    def test_invalid_values(self):
        invalid = {
            "INTEGRATION_METHOD": ["simpson", 1, None],
            "POSTPROC_METHOD": ["cuda", 1, None],
            "QUADRATURE_ORDER": [0, -3, 2.5, "eight"],
            "QUADRATURE_PANELS_PER_HALF_WAVE": [0],
            "QUADRATURE_BATCH_SIZE": [0],
            "NUM_MC_SAMPLES_PLATE": [0],
            "MC_BATCH_SIZE_PLATE": [-1],
            "NUM_MC_SAMPLES_BEAM": [0],
            "MC_BATCH_SIZE_BEAM": [-1],
            "MC_SEED": [-1, 0.5],
            "POSTPROC_BUFFER_SIZE": [0, 2.5],
        }
        config = FourierConfig()
        for name, values in invalid.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    setattr(config, name, value)
        self.assertEqual(config, FourierConfig())

    def test_unknown_setting(self):
        with self.assertRaises(ValueError):
            Config.POSTPROC_METOD = "numba"
        with self.assertRaises(ValueError):
            FourierConfig(POSTPROC_METOD="numba")

    def test_patch_restores_value(self):
        with mock.patch.object(Config, "POSTPROC_METHOD", "numba"):
            self.assertIs(Config.POSTPROC_METHOD, PostprocMethod.NUMBA)
        self.assertIs(Config.POSTPROC_METHOD, PostprocMethod.BLAS)
        with (
            self.assertRaises(ValueError),
            mock.patch.object(Config, "POSTPROC_METHOD", "cuda"),
        ):
            pass  # pragma: no cover
        self.assertIs(Config.POSTPROC_METHOD, PostprocMethod.BLAS)


if __name__ == "__main__":
    unittest.main()
