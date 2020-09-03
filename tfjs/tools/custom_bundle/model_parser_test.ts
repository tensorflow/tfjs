/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// tslint:disable-next-line: no-imports-from-dist
import * as tensorflow from '@tensorflow/tfjs-converter/dist/data/compiled_api';
import {io} from '@tensorflow/tfjs-core';

import {getOps} from './model_parser';

const SIMPLE_MODEL: io.ModelArtifacts = {
  format: 'graph-model',
  generatedBy: '0.0.0',
  convertedBy: 'Test Data',
  modelTopology: {
    node: [
      {
        name: 'Input',
        op: 'Placeholder',
        attr: {
          dtype: {
            type: tensorflow.DataType.DT_INT32,
          },
          shape: {shape: {dim: [{size: -1}, {size: 1}]}}
        }
      },
      {name: 'Add1', op: 'Add', input: ['Input', 'Const'], attr: {}},
      {name: 'Sub', op: 'Sub', input: ['Add1', 'Input'], attr: {}}
    ],
    versions: {producer: 1.0, minConsumer: 3}
  }
};

const CONTROL_FLOW_V2_MODEL: io.ModelArtifacts = {
  format: 'graph-model',
  generatedBy: '0.0.0',
  convertedBy: 'Test Data',
  modelTopology: {
    node: [
      {
        name: 'image_placeholder',
        op: 'Placeholder',
        attr: {
          dtype: {
            type: tensorflow.DataType.DT_FLOAT,
          },
          shape:
              {shape: {dim: [{size: '3'}, {size: 3}, {size: '3'}, {size: 1}]}}
        }
      },
      {
        name: 'Const',
        op: 'Const',
        attr: {
          dtype: {type: tensorflow.DataType.DT_INT32},
          value: {
            tensor: {
              dtype: tensorflow.DataType.DT_INT32,
              tensorShape: {dim: [{size: 3}, {size: 3}, {size: 1}, {size: 1}]},
              intVal: [0, 0, 0, 0, 1, 0, 0, 0, 0]
            }
          }
        }
      },
      {
        name: 'Shape',
        op: 'Const',
        attr: {
          dtype: {type: tensorflow.DataType.DT_INT32},
          value: {
            tensor: {
              dtype: tensorflow.DataType.DT_INT32,
              tensorShape: {dim: [{size: 3}, {size: 1}, {size: 1}, {size: 1}]},
              intVal: [1, 1, 1]
            }
          }
        }
      },
      {
        name: 'Value',
        op: 'Const',
        attr: {dtype: {type: tensorflow.DataType.DT_INT32}, value: {i: 1}}
      },
      {name: 'Fill', op: 'Fill', input: ['Shape', 'Value'], attr: {}},
      {
        name: 'Conv2D',
        op: 'Conv2D',
        input: ['image_placeholder', 'Const'],
        attr: {
          T: {type: tensorflow.DataType.DT_FLOAT},
          dataFormat: {s: 'TkhXQw=='},
          padding: {s: 'U0FNRQ=='},
          strides: {list: {f: [], i: [1, 2, 2, 1]}},
          useCudnnOnGpu: {b: true}
        }
      },
      {
        name: 'BiasAdd',
        op: 'BiasAdd',
        input: ['Conv2D', 'Shape'],
        attr: {
          T: {type: tensorflow.DataType.DT_FLOAT},
          dataFormat: {s: 'TkhXQw=='}
        }
      },
      {
        name: 'Cast',
        op: 'Cast',
        input: ['BiasAdd'],
        attr: {DstT: {type: tensorflow.DataType.DT_INT64}}
      },
      {
        name: 'Squeeze',
        op: 'Squeeze',
        input: ['Cast'],
        attr: {squeeze_dims: {list: {i: ['1', '2']}}}
      },
      {
        name: 'Squeeze2',
        op: 'Squeeze',
        input: ['BiasAdd'],
        attr: {squeeze_dims: {list: {}}}
      },
      {
        name: 'Split',
        op: 'Split',
        input: ['image_placeholder'],
        attr: {num_split: {i: 3} as tensorflow.IAttrValue}
      },
      {name: 'LogicalNot', op: 'LogicalNot', input: ['image_placeholder']},
      {
        name: 'FusedBatchNorm',
        op: 'FusedBatchNorm',
        input: ['image_placeholder'],
        attr: {epsilon: {f: 0.0001} as tensorflow.IAttrValue}
      },
      {
        name: 'Cast2',
        op: 'Cast',
        input: ['BiasAdd'],
        attr: {DstT: {type: tensorflow.DataType.DT_UINT8}}
      },
    ],
    library: {
      function: [
        {
          signature: {
            name: '__inference_while_cond_10_49_frozen',
            inputArg: [
              {name: 'while_loop_counter', type: tensorflow.DataType.DT_INT32},
              {
                name: 'while_maximum_iterations',
                type: tensorflow.DataType.DT_INT32
              },
              {name: 'placeholder', type: tensorflow.DataType.DT_INT32},
              {name: 'less_y', type: tensorflow.DataType.DT_INT32}, {
                name: 'while_cond_10___redundant_placeholder0',
                type: tensorflow.DataType.DT_INT32
              }
            ],
            outputArg: [{name: 'identity', type: tensorflow.DataType.DT_BOOL}]
          },
          nodeDef: [{
            name: 'Less',
            op: 'Less',
            input: ['placeholder', 'less_y'],
            attr: {T: {type: tensorflow.DataType.DT_INT32}}
          }],
          ret: {identity: 'Less:z:0'}
        },
        {
          signature: {
            name: '__inference_while_body_11_40_frozen',
            inputArg: [
              {name: 'while_loop_counter', type: tensorflow.DataType.DT_INT32},
              {
                name: 'while_maximum_iterations',
                type: tensorflow.DataType.DT_INT32
              },
              {name: 'placeholder', type: tensorflow.DataType.DT_INT32},
              {name: 'y_0', type: tensorflow.DataType.DT_INT32},
              {name: 'add_1_z_0', type: tensorflow.DataType.DT_INT32}
            ],
            outputArg: [
              {name: 'identity', type: tensorflow.DataType.DT_INT32},
              {name: 'identity_1', type: tensorflow.DataType.DT_INT32},
              {name: 'identity_2', type: tensorflow.DataType.DT_INT32},
              {name: 'y', type: tensorflow.DataType.DT_INT32},
              {name: 'add_1_z', type: tensorflow.DataType.DT_INT32}
            ]
          },
          nodeDef: [
            {
              name: 'add_2/y',
              op: 'Const',
              attr: {
                dtype: {type: tensorflow.DataType.DT_INT32},
                value: {
                  tensor: {
                    dtype: tensorflow.DataType.DT_INT32,
                    tensorShape: {},
                    intVal: [1]
                  }
                }
              }
            },
            {
              name: 'add',
              op: 'AddV2',
              input: ['placeholder', 'y_0'],
              attr: {T: {type: tensorflow.DataType.DT_INT32}}
            },
            {
              name: 'add_2',
              op: 'AddV2',
              input: ['add_2/y:output:0', 'while_loop_counter'],
              attr: {T: {type: tensorflow.DataType.DT_INT32}}
            },
            {
              name: 'add_1',
              op: 'AddV2',
              input: ['add:z:0', 'add_1_z_0'],
              attr: {T: {type: tensorflow.DataType.DT_INT32}}
            }
          ],
          ret: {
            identity_1: 'while_maximum_iterations',
            identity_2: 'add_1:z:0',
            y: 'y_0',
            identity: 'add_2:z:0',
            add_1_z: 'add_1_z_0'
          }
        }
      ]
    },
    versions: {producer: 1.0}
  }
};

describe('Model parse', () => {
  it('should get ops from simple model', () => {
    const ops = getOps(SIMPLE_MODEL);
    expect(ops).toEqual(jasmine.arrayWithExactContents(['add', 'sub']));
  });

  it('should get ops from control flow v2 model', () => {
    const ops = getOps(CONTROL_FLOW_V2_MODEL);
    expect(ops).toEqual(jasmine.arrayWithExactContents([
      'cast', 'batchNorm', 'logicalNot', 'split', 'squeeze', 'add', 'conv2d',
      'fill', 'less'
    ]));
  });
});
