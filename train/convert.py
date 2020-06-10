import tensorflowjs as tfjs

from sys import argv


def convert(model_fn):
    tfjs.converters.save_keras_model(model_fn, 'model.json')


if __name__ == '__main__':
    fn = argv[1]
    convert(fn)
