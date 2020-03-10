export const ops = [
  // {
  //   'name': 'FromPixels',
  //   'result': [513, 513, 3],
  //   'inputs': {'pixels': {}},
  //   'timeMs': 0.197
  // },
  // {
  //   'name': 'Cast',
  //   'result': [257, 257, 3],
  //   'inputs': [[257, 257, 3]],
  //   'timeMs': 0
  // },
  // {'name': 'Mul', 'result': [17, 2], 'inputs': [[17, 2], []], 'timeMs': 0},
  // {'name': 'Cast', 'result': [17, 2], 'inputs': [[17, 2]], 'timeMs': 0},
  // {'name': 'Add', 'result': [17, 2], 'inputs': [[17, 2], [17, 2]], 'timeMs':
  // 0},
  // {
  //   'name': 'PadV2',
  //   'result': [513, 513, 3],
  //   'inputs': [[513, 513, 3]],
  //   'timeMs': 0.312
  // },
  {
    'name': 'ResizeBilinear',
    'scope': 'image',
    'result': [1, 257, 257, 3],
    'inputs': [[1, 513, 513, 3]],
    'timeMs': 0.513
  },
  // {
  //   'name': 'Add',
  //   'result': [257, 257, 3],
  //   'inputs': [[257, 257, 3], [3]],
  //   'timeMs': 0.124
  // },
  // {
  //   'name': 'PadV2',
  //   'result': [1, 263, 263, 3],
  //   'inputs': [[1, 257, 257, 3]],
  //   'timeMs': 0.107
  // },
  // {
  //   'name': 'Conv2D',
  //   'result': [1, 129, 129, 64],
  //   'inputs': [[1, 263, 263, 3], [7, 7, 3, 64]],
  //   'timeMs': 9.573
  // },
  // {
  //   'name': 'Add',
  //   'result': [1, 129, 129, 64],
  //   'inputs': [[1, 129, 129, 64], [64]],
  //   'timeMs': 0.759
  // },
  // {
  //   'name': 'Relu',
  //   'result': [1, 129, 129, 64],
  //   'inputs': [[1, 129, 129, 64]],
  //   'timeMs': 1.459
  // },
  // {'name': 'PadV2', 'result': [1, 131, 131, 64], 'inputs': [[1, 129, 129,
  // 64]]},
  {
    'name': 'MaxPool',
    'result': [1, 65, 65, 64],
    'inputs': [[1, 131, 131, 64]],
    'args': [3, 2, 'valid']
  },
  {
    'name': 'Conv2D',
    'result': [1, 65, 65, 64],
    'inputs': [[1, 65, 65, 64], [1, 1, 64, 64]],
    'args': [1, 'same']
  }
];
