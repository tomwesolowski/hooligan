import numpy as np
import tensorflow as tf
import types


class ParamsDict(dict):
    def __init__(self, *args, **kwargs):
        super(ParamsDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def define_flags(self):
        for k, v in self.items():
            param = v.get() if isinstance(v, HParam) else v
            if isinstance(param, bool):
                tf.app.flags.DEFINE_bool(k, None, k)
            elif isinstance(param, float):
                tf.app.flags.DEFINE_float(k, None, k)
            elif isinstance(param, int):
                tf.app.flags.DEFINE_integer(k, None, k)
            elif isinstance(param, str):
                tf.app.flags.DEFINE_string(k, None, k)
            else:
                print('Could not create a flag for: %s with type: %s' % (k, type(v)))

    def initialized(self):
        params_values = dict()
        for k, v in self.items():
            if isinstance(v, HParam):
                params_values[k] = v.get()
            elif isinstance(v, types.FunctionType):
                params_values[k] = v()
            else:
                params_values[k] = v
        return params_values

    def describe(self):
        for k, v in self.items():
            print(k, '=', v)


class HParam(object):
    def __init__(self):
        super(HParam, self).__init__()

    def get(self):
        raise NotImplementedError()


class HParamSelect(HParam):
    def __init__(self, array):
        super(HParamSelect, self).__init__()
        self.array = array

    def get(self):
        return self.array[np.random.randint(0, len(self.array))]


class HParamRange(HParam):
    def __init__(self, lowerbound, upperbound, size=1, method='uniform', fn=None):
        super(HParamRange, self).__init__()
        if lowerbound >= upperbound:
            raise ValueError('Lowerbound >= Upperbound')
        self.method = method
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.size = None if size == 1 else size
        self.fn = (lambda x: x) if fn is None else fn

    def get(self):
        if self.method == 'uniform':
            value = np.random.uniform(self.lowerbound, self.upperbound, self.size)
            return value
        else:
            raise ValueError('Unknown HParamRange method: %s' % self.method)