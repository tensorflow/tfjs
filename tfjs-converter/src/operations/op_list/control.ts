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
import {OpMapper} from '../types';

export const json: OpMapper[] = [
  {
    'tfOpName': 'EmptyTensorList',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'elementShape', 'type': 'shape'},
      {'start': 1, 'name': 'maxNumElements', 'type': 'number'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'LoopCond',
    'category': 'control',
    'inputs': [{'start': 0, 'name': 'pred', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Switch',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'data', 'type': 'tensor'},
      {'start': 1, 'name': 'pred', 'type': 'tensor'}
    ]
  },
  {
    'tfOpName': 'Merge',
    'category': 'control',
    'inputs': [{'start': 0, 'end': 0, 'name': 'tensors', 'type': 'tensors'}]
  },
  {
    'tfOpName': 'Enter',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true},
      {'tfName': 'frame_name', 'name': 'frameName', 'type': 'string'},
      {'tfName': 'is_constant', 'name': 'isConstant', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'Exit',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'NextIteration',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'TensorArrayV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'size', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'},
      {'tfName': 'element_shape', 'name': 'elementShape', 'type': 'shape'},
      {'tfName': 'dynamic_size', 'name': 'dynamicSize', 'type': 'bool'},
      {'tfName': 'clear_after_read', 'name': 'clearAfterRead', 'type': 'bool'},
      {
        'tfName': 'identical_element_shapes',
        'name': 'identicalElementShapes',
        'type': 'bool'
      },
      {'tfName': 'tensor_array_name', 'name': 'name', 'type': 'string'}
    ]
  },
  {
    'tfOpName': 'TensorArrayWriteV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'index', 'type': 'number'},
      {'start': 2, 'name': 'tensor', 'type': 'tensor'},
      {'start': 3, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'TensorArrayReadV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'index', 'type': 'number'},
      {'start': 2, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [{
      'tfName': 'dtype',
      'name': 'dtype',
      'type': 'dtype',
      'notSupported': true
    }]
  },
  {
    'tfOpName': 'TensorArrayGatherV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'number[]'},
      {'start': 2, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'},
      {'tfName': 'element_shape', 'name': 'elementShape', 'type': 'shape'}
    ]
  },
  {
    'tfOpName': 'TensorArrayScatterV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'number[]'},
      {'start': 2, 'name': 'tensor', 'type': 'tensor'},
      {'start': 3, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorArrayConcatV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'}, {
        'tfName': 'element_shape_except0',
        'name': 'elementShapeExcept0',
        'type': 'shape',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'TensorArraySplitV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'tensor', 'type': 'tensor'},
      {'start': 2, 'name': 'lengths', 'type': 'number[]'},
      {'start': 3, 'name': 'flowIn', 'type': 'number'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorArraySizeV3',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'},
      {'start': 1, 'name': 'flowIn', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'TensorArrayCloseV3',
    'category': 'control',
    'inputs': [{'start': 0, 'name': 'tensorArrayId', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'StatelessIf',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'cond', 'type': 'tensor'},
      {'start': 1, 'end': 0, 'name': 'args', 'type': 'tensors'}
    ],
    'attrs': [
      {'tfName': 'then_branch', 'name': 'thenBranch', 'type': 'func'},
      {'tfName': 'else_branch', 'name': 'elseBranch', 'type': 'func'}
    ]
  },
  {
    'tfOpName': 'If',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'cond', 'type': 'tensor'},
      {'start': 1, 'end': 0, 'name': 'args', 'type': 'tensors'}
    ],
    'attrs': [
      {'tfName': 'then_branch', 'name': 'thenBranch', 'type': 'func'},
      {'tfName': 'else_branch', 'name': 'elseBranch', 'type': 'func'}
    ]
  },
  {
    'tfOpName': 'StatelessWhile',
    'category': 'control',
    'inputs': [
      {'start': 0, 'end': 0, 'name': 'args', 'type': 'tensors'},
    ],
    'attrs': [
      {'tfName': 'cond', 'name': 'cond', 'type': 'func'},
      {'tfName': 'body', 'name': 'body', 'type': 'func'}
    ]
  },
  {
    'tfOpName': 'While',
    'category': 'control',
    'inputs': [
      {'start': 0, 'end': 0, 'name': 'args', 'type': 'tensors'},
    ],
    'attrs': [
      {'tfName': 'cond', 'name': 'cond', 'type': 'func'},
      {'tfName': 'body', 'name': 'body', 'type': 'func'}
    ]
  },
  {
    'tfOpName': 'TensorListScatter',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'number[]'},
      {'start': 2, 'name': 'elementShape', 'type': 'shape'}
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListScatterV2',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'number[]'},
      {'start': 2, 'name': 'elementShape', 'type': 'shape'},
      {'start': 3, 'name': 'numElements', 'type': 'number'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListGather',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'number[]'},
      {'start': 2, 'name': 'elementShape', 'type': 'shape'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListGetItem',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'index', 'type': 'number'},
      {'start': 2, 'name': 'elementShape', 'type': 'shape'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListSetItem',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'index', 'type': 'number'},
      {'start': 2, 'name': 'tensor', 'type': 'tensor'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListReserve',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'elementShape', 'type': 'shape'},
      {'start': 1, 'name': 'numElements', 'type': 'number'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListFromTensor',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
      {'start': 1, 'name': 'elementShape', 'type': 'shape'}
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListStack',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'elementShape', 'type': 'shape'},
    ],
    'attrs': [
      {'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'},
      {'tfName': 'num_elements', 'name': 'numElements', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'TensorListSplit',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensor', 'type': 'tensor'},
      {'start': 1, 'name': 'elementShape', 'type': 'shape'},
      {'start': 2, 'name': 'lengths', 'type': 'number[]'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListConcat',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'element_shape', 'name': 'elementShape', 'type': 'shape'},
      {'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'TensorListPopBack',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'elementShape', 'type': 'shape'},
    ],
    'attrs':
        [{'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TensorListPushBack',
    'category': 'control',
    'inputs': [
      {'start': 0, 'name': 'tensorListId', 'type': 'tensor'},
      {'start': 1, 'name': 'tensor', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'element_dtype', 'name': 'elementDType', 'type': 'dtype'}
    ]
  }
];
