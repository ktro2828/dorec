#!/usr/bin/env python

"""
References:
- open-mmlab/mmcv
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
- PaddlePaddle/PaddleSeg
    https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/paddleseg/cvlibs/manager.py
"""

import inspect


def build_from_cfg(cfg, manager):
    """A functio to build module from config
    Args:
        cfg (dict[str, any])
        manager (ModuleManeger)
    Returns:
        ``obj``
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict, but got {}".format(type(cfg)))

    if not isinstance(manager, ModuleManager):
        raise TypeError(
            "module must a Registry, but got {}".format(type(manager)))

    # Copy cfg not to destroy
    cfg_args = cfg.copy()

    obj_name = cfg_args.pop("name")
    if isinstance(obj_name, str):
        obj_cls = manager.get(obj_name)
    elif inspect.isclass(obj_name):
        obj_cls = obj_name
    else:
        raise TypeError(
            "``name`` must be str or valid type, but got {}".format(type(obj_name)))

    return obj_cls(**cfg_args)


class ModuleManager:
    """A manager to map strings to classes
    Args:
        name (str): Manger name
        build_func (func, optional): Build function to construct instance from registry
    """

    def __init__(self, name, build_func=None):
        self._name = name
        self._module_dict = dict()

        # Set build func
        if build_func is None:
            self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def ___repr__(self):
        format_str = self.__class__.__name__ + \
            "name={}".format(self.name) + \
            "items={}".format(self.module_dict)
        return format_str

    def keys(self, idx=None):
        if idx is None:
            return self._module_dict.keys()
        else:
            if not isinstance(idx, int):
                raise TypeError(
                    "idx must be a type of int, but got {}".format(type(idx)))
            return self._module_dict.keys()[idx]

    def __pos__(self, other):
        self._module_dict.update(other.module_dict)
        return self

    def __sub__(self, other):
        pop_keys = set(self.keys()) & set(self.keys())
        for key in pop_keys:
            self._module_dict.pop(key)
        return self

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get registered modules with key
        Args:
            key (str): The class name in string format
        Returns:
            class: The corresponding class
        """
        if key not in self._module_dict:
            raise KeyError("{} is not registed yet".format(key))
        return self._module_dict[key]

    def build(self, *args, **kwargs):
        return self.build_func(manager=self, *args, **kwargs)

    def register(self, name=None, force=False, module=None):
        """Register a module
        Args:
            name (str, optional)
            force (bool, optional)
            module (class, optional)
        """
        if name is not None and not isinstance(name, (str, list)):
            raise TypeError(
                "name must be a str or list, but got {}".format(type(name)))

        if not isinstance(force, bool):
            raise TypeError(
                "force must be a boolean, but got {}".format(type(force)))

        if module is not None:
            """Use as a normal function
                >>> x.register(module=``obj``)
            """
            self._add_module(module, name=name, force=force)
            return module

        def _register_decorator(cls):
            """Use as a decoretor
                >>> @x.register(...)
            """
            self._add_module(cls, name=name, force=force)
            return cls

        return _register_decorator

    def _add_module(self, module, name=None, force=False):
        """Add module, called in self.register()
        Args:
            module (class)
            name (str, optional)
            force (bool, optional)
        """
        if not inspect.isclass(module):
            raise TypeError(
                "module must be a class, but got {}".format(type(module)))

        name = module.__name__ if name is None else name
        name_list = [name] if isinstance(name, str) else name

        for nm in name_list:
            if not force and nm in self.module_dict:
                raise KeyError(
                    "{} is already registered in {}".format(nm, self.name))
            self._module_dict[nm] = module


MODELS = ModuleManager("models")
BACKBONES = ModuleManager("backbones")
HEADS = ModuleManager("heads")
LOSSES = ModuleManager("losses")
DATASETS = ModuleManager("datasets")
TRANSFORMS = ModuleManager("transforms")
RUNNERS = ModuleManager("runners")
