import yaml
import os
import logging

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.basicConfig(level=self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration loaded")

    def get(self, key):
        return self.config.get(key)