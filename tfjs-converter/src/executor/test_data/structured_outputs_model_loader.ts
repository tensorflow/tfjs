/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export const STRUCTURED_OUTPUTS_MODEL = {
  'modelTopology': {
    'node': [
      {
        'name': 'StatefulPartitionedCall/model/concatenate/concat/axis',
        'op': 'Const',
        'attr': {
          'value': {'tensor': {'dtype': 'DT_INT32', 'tensorShape': {}}},
          'dtype': {'type': 'DT_INT32'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp',
        'op': 'Const',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'value': {
            'tensor': {
              'dtype': 'DT_FLOAT',
              'tensorShape': {'dim': [{'size': '2'}, {'size': '1'}]}
            }
          }
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp',
        'op': 'Const',
        'attr': {
          'value': {
            'tensor': {
              'dtype': 'DT_FLOAT',
              'tensorShape': {'dim': [{'size': '1'}, {'size': '1'}]}
            }
          },
          'dtype': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'input1',
        'op': 'Placeholder',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}}
        }
      },
      {
        'name': 'input2',
        'op': 'Placeholder',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}}
        }
      },
      {
        'name': 'input3',
        'op': 'Placeholder',
        'attr': {
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}},
          'dtype': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/b/MatMul',
        'op': 'MatMul',
        'input':
            ['input2', 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp'],
        'device': '/device:CPU:0',
        'attr': {
          'transpose_b': {'b': false},
          'transpose_a': {'b': false},
          'T': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/concatenate/concat',
        'op': 'ConcatV2',
        'input': [
          'input1', 'input3',
          'StatefulPartitionedCall/model/concatenate/concat/axis'
        ],
        'attr': {
          'Tidx': {'type': 'DT_INT32'},
          'T': {'type': 'DT_FLOAT'},
          'N': {'i': '2'}
        }
      },
      {
        'name': 'Identity_1',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/b/MatMul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'StatefulPartitionedCall/model/a/MatMul',
        'op': 'MatMul',
        'input': [
          'StatefulPartitionedCall/model/concatenate/concat',
          'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp'
        ],
        'device': '/device:CPU:0',
        'attr': {
          'T': {'type': 'DT_FLOAT'},
          'transpose_b': {'b': false},
          'transpose_a': {'b': false}
        }
      },
      {
        'name': 'Identity',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/a/MatMul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'StatefulPartitionedCall/model/c/mul',
        'op': 'Mul',
        'input': [
          'StatefulPartitionedCall/model/a/MatMul',
          'StatefulPartitionedCall/model/b/MatMul'
        ],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'Identity_2',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/c/mul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      }
    ],
    'library': {},
    'versions': {'producer': 898}
  },
  'format': 'graph-model',
  'generatedBy': '2.7.3',
  'convertedBy': 'TensorFlow.js Converter v1.7.0',
  'weightSpecs': [
    {
      'name': 'StatefulPartitionedCall/model/concatenate/concat/axis',
      'shape': [],
      'dtype': 'int32'
    },
    {
      'name': 'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp',
      'shape': [2, 1],
      'dtype': 'float32'
    },
    {
      'name': 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp',
      'shape': [1, 1],
      'dtype': 'float32'
    }
  ],
  'weightData': new Uint8Array([
                  0x01, 0x00, 0x00, 0x00, 0x70, 0x3d, 0x72, 0x3e, 0x3d, 0xd2,
                  0x12, 0xbf, 0x0c, 0xfb, 0x94, 0x3e
                ]).buffer,
  'signature': {
    'inputs': {
      'input1:0': {
        'name': 'input1:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'input3:0': {
        'name': 'input3:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'input2:0': {
        'name': 'input2:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      }
    },
    'outputs': {
      'Identity_1:0': {
        'name': 'Identity_1:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'Identity:0': {
        'name': 'Identity:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'Identity_2:0': {
        'name': 'Identity_2:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      }
    }
  },
  'userDefinedMetadata': {'structuredOutputKeys': ['a', 'b', 'c']}
};
