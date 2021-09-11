import importlib
from easydict import EasyDict
import yaml


def yaml_load(file):
    with open(file, 'r') as f:
        ed = EasyDict(yaml.safe_load(f))
    return ed


def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class, codes borrowed from ESPNet.

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)
