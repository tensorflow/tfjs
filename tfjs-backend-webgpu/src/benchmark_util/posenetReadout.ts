export const ops = [
  {
    'name': 'PadV2',
    'inputs': [[1, 15, 15, 256]],
    'args': [[[0, 0], [1, 1], [1, 1], [0, 0]]]
  },
  {
    'name': 'PadV2',
    'inputs': [[1, 29, 29, 128]],
    'args': [[[0, 0], [1, 1], [1, 1], [0, 0]]]
  },
  {
    'name': 'PadV2',
    'inputs': [[1, 57, 57, 64]],
    'args': [[[0, 0], [1, 1], [1, 1], [0, 0]]]
  },
  {
    'name': 'PadV2',
    'inputs': [[1, 113, 113, 64]],
    'args': [[[0, 0], [1, 1], [1, 1], [0, 0]]]
  },
  {
    'name': 'PadV2',
    'inputs': [[1, 225, 225, 3]],
    'args': [[[0, 0], [3, 3], [3, 3], [0, 0]]]
  },
  {
    'name': 'PadV2',
    'inputs': [[500, 600, 3]],
    'args': [[[50, 50], [0, 0], [0, 0]]]
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 256], [1, 1, 256, 1024]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 512], [1, 1, 512, 2048]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 1024], [1, 1, 1024, 2048]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 1024], [1, 1, 1024, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 2048], [1, 1, 2048, 17]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 2048], [1, 1, 2048, 32]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 2048], [1, 1, 2048, 34]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 2048], [1, 1, 2048, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 128], [1, 1, 128, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 256], [1, 1, 256, 1024]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 512], [1, 1, 512, 256]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 512], [1, 1, 512, 1024]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 1024], [1, 1, 1024, 256]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 64], [1, 1, 64, 256]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 128], [1, 1, 128, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 256], [1, 1, 256, 128]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 256], [1, 1, 256, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 512], [1, 1, 512, 128]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 57, 57, 64], [1, 1, 64, 64]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 57, 57, 64], [1, 1, 64, 256]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 57, 57, 256], [1, 1, 256, 64]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 8, 8, 512], [3, 3, 512, 512]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 15, 15, 256], [3, 3, 256, 256]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 17, 17, 256], [3, 3, 256, 256]],
    'args': [2, 'valid']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 29, 29, 128], [3, 3, 128, 128]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 31, 31, 128], [3, 3, 128, 128]],
    'args': [2, 'valid']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 57, 57, 64], [3, 3, 64, 64]],
    'args': [1, 'same']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 59, 59, 64], [3, 3, 64, 64]],
    'args': [2, 'valid']
  },
  {
    'name': 'Conv2D',
    'inputs': [[1, 231, 231, 3], [7, 7, 3, 64]],
    'args': [2, 'valid']
  },
  {
    'name': 'Relu',
    'inputs': [[1, 15, 15, 128]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 15, 15, 256]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 15, 15, 512]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 15, 15, 1024]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 29, 29, 64]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 29, 29, 128]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 29, 29, 256]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 29, 29, 512]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 57, 57, 64]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 57, 57, 256]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 8, 8, 256]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 8, 8, 512]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 8, 8, 1024]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 8, 8, 2048]],
  },
  {
    'name': 'Relu',
    'inputs': [[1, 113, 113, 64]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 17], [17]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 32], [32]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 34], [34]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 256], [256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 512], [512]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 1024], [1, 8, 8, 1024]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 1024], [1024]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 2048], [1, 8, 8, 2048]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 8, 8, 2048], [2048]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 128], [128]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 256], [256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 512], [1, 15, 15, 512]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 512], [512]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 1024], [1, 15, 15, 1024]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 15, 15, 1024], [1024]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 64], [64]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 128], [128]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 256], [1, 29, 29, 256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 256], [256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 512], [1, 29, 29, 512]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 29, 29, 512], [512]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 57, 57, 64], [64]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 57, 57, 256], [1, 57, 57, 256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 57, 57, 256], [256]],
  },
  {
    'name': 'Add',
    'inputs': [[1, 113, 113, 64], [64]],
  },
  {
    'name': 'Add',
    'inputs': [[256, 256, 3], [3]],
  },
  {'name': 'MaxPool', 'inputs': [[1, 15, 15, 1024]], 'args': [1, 2, 'same']},
  {'name': 'MaxPool', 'inputs': [[1, 29, 29, 512]], 'args': [1, 2, 'same']},
  {'name': 'MaxPool', 'inputs': [[1, 57, 57, 256]], 'args': [1, 2, 'same']},
  {'name': 'MaxPool', 'inputs': [[1, 115, 115, 64]], 'args': [3, 2, 'valid']},
];
