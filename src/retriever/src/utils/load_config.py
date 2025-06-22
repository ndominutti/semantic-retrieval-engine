import os
import yaml


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_relative_path="config.yml"):
    project_root = get_project_root()
    config_path = os.path.join(project_root, config_relative_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
