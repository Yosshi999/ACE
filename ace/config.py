from google.protobuf import text_format

from ace.config_pb2 import Config


def load(config_path):
    config = Config()
    with open(config_path) as f:
        text_format.Merge(f.read(), config)
    return config
