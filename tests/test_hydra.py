import unittest
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class TestHydra(unittest.TestCase):
    def test_hydra(self):
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name="defaults")
            print(OmegaConf.to_yaml(cfg))

            model = instantiate(cfg.model.object)
            print(model)

            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()