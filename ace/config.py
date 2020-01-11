from ast import literal_eval

from google.protobuf import text_format
from google.protobuf.descriptor_pb2 import FieldDescriptorProto

from ace.config_pb2 import Config


def load(config_path: str, overrides: [str]):
    config = Config()
    with open(config_path) as f:
        text_format.Merge(f.read(), config)
    merge_from_list(config, overrides)
    return config


def merge_from_list(self: Config, cfg_list: [str]):
    assert len(cfg_list) % 2 == 0, 'override list has odd length'
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = self
        for subkey in key_list[:-1]:
            d = getattr(d, subkey)
        subkey = key_list[-1]
        value = _decode_cfg_value(v)
        if getattr(d.__class__, subkey).DESCRIPTOR.label == FieldDescriptorProto.LABEL_REPEATED:
            getattr(d, subkey)[:] = value
        else:
            setattr(d, subkey, value)


def _decode_cfg_value(value: str):
    try:
        value = literal_eval(value)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return value
