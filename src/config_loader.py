import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Reads a YAML config file and returns a dict of parameters."""
    cfg_file = Path(config_path)
    with cfg_file.open() as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
