import os
import yaml


def read_config(config_file: str):
    with open(config_file, "r") as f:
        config_file = yaml.safe_load(f)

    return config_file


if __name__ == "__main__":
    sfm_config_file = "/home/minxuan/Code/colmap/config/sfm_config_default.yaml"
    config = read_config(sfm_config_file)

    print("context: {}".format(config))

    feature_model_list = config.get("feature_mode", ["sift"])

    print("feature_model_list: {}".format(feature_model_list))

    