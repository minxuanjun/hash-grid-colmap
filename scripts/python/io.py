import json
import yaml


def load_json(filename:str)->dict:
    with open(filename, "r", encoding='utf-8') as f:
        contexts = json.load(f)
        return  contexts

def save_json(filename, data):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.load(file)
    return  config

def dump_configs(config_path, data):
    with open(config_path, "w", encoding='utf-8') as f:
        yaml.dump(data, f)
