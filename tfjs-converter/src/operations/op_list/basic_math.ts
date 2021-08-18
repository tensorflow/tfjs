import {OpMapper} from '../types';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const json: OpMapper[] = [
  {
    'tfOpName': 'Abs',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Acos',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Asin',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Atan',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Atan2',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'y', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Ceil',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'ClipByValue',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'clipValueMin', 'type': 'number'},
      {'start': 2, 'name': 'clipValueMax', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Complex',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'real', 'type': 'tensor'},
      {'start': 1, 'name': 'imag', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'ComplexAbs',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Cos',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Cosh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Elu',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Exp',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Floor',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Log',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Imag',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'outputType',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Neg',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Real',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'outputType',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Prelu',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'alpha', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Relu',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Relu6',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Selu',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Sigmoid',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Sin',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Sinh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Sqrt',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Rsqrt',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Square',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Tan',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Tanh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Sign',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Round',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Expm1',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Log1p',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Reciprocal',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Softplus',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Asinh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Acosh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Atanh',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Erf',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Prod',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axes', 'type': 'number[]'},
    ],
    'attrs': [
      {
        'tfName': 'keep_dims',
        'name': 'keepDims',
        'type': 'bool',
        'notSupported': true
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'LeakyRelu',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'alpha',
        'name': 'alpha',
        'type': 'number',
        'defaultValue': 0.2
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'IsNan',
    'category': 'basic_math',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [{
      'tfName': 'T',
      'name': 'dtype',
      'type': 'dtype',
      'notSupported': true
    }]
  }
];
