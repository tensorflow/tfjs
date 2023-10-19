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
export const HASH_TABLE_MODEL_V2 = {
  modelTopology: {
    node: [
      {
        name: 'unknown_0',
        op: 'Const',
        attr: {
          value: {tensor: {dtype: 'DT_INT32', tensorShape: {}}},
          dtype: {type: 'DT_INT32'}
        }
      },
      {
        name: 'input',
        op: 'Placeholder',
        attr:
            {shape: {shape: {dim: [{size: '-1'}]}}, dtype: {type: 'DT_STRING'}}
      },
      {
        name: 'unknown',
        op: 'Placeholder',
        attr: {shape: {shape: {}}, dtype: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'StatefulPartitionedCall/None_Lookup/LookupTableFindV2',
        op: 'LookupTableFindV2',
        input: ['unknown', 'input', 'unknown_0'],
        attr: {
          Tout: {type: 'DT_INT32'},
          Tin: {type: 'DT_STRING'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'Identity',
        op: 'Identity',
        input: ['StatefulPartitionedCall/None_Lookup/LookupTableFindV2'],
        attr: {T: {type: 'DT_INT32'}}
      }
    ],
    library: {},
    versions: {producer: 1240}
  },
  format: 'graph-model',
  generatedBy: '2.11.0-dev20220822',
  convertedBy: 'TensorFlow.js Converter v1.7.0',
  weightSpecs: [
    {name: 'unknown_0', shape: [], dtype: 'int32'},
    {name: '114', shape: [2], dtype: 'string'},
    {name: '116', shape: [2], dtype: 'int32'}
  ],
  'weightData':
      new Uint8Array([
        0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x61, 0x01, 0x00,
        0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00
      ]).buffer,

  signature: {
    inputs: {
      input: {
        name: 'input:0',
        dtype: 'DT_STRING',
        tensorShape: {dim: [{size: '-1'}]}
      },
      'unknown:0': {
        name: 'unknown:0',
        dtype: 'DT_RESOURCE',
        tensorShape: {},
        resourceId: 66
      }
    },
    outputs: {
      output_0: {
        name: 'Identity:0',
        dtype: 'DT_INT32',
        tensorShape: {dim: [{size: '-1'}]}
      }
    }
  },
  modelInitializer: {
    node: [
      {
        name: 'Func/StatefulPartitionedCall/input_control_node/_0',
        op: 'NoOp',
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: '114',
        op: 'Const',
        attr: {
          value:
              {tensor: {dtype: 'DT_STRING', tensorShape: {dim: [{size: '2'}]}}},
          _has_manual_control_dependencies: {b: true},
          dtype: {type: 'DT_STRING'}
        }
      },
      {
        name: '116',
        op: 'Const',
        attr: {
          _has_manual_control_dependencies: {b: true},
          dtype: {type: 'DT_INT32'},
          value:
              {tensor: {dtype: 'DT_INT32', tensorShape: {dim: [{size: '2'}]}}}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9',
        op: 'NoOp',
        input: ['^Func/StatefulPartitionedCall/input_control_node/_0'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
        op: 'HashTableV2',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9'
        ],
        attr: {
          container: {s: ''},
          use_node_name_sharing: {b: true},
          _has_manual_control_dependencies: {b: true},
          shared_name: {s: 'OTVfbG9hZF8xXzUy'},
          value_dtype: {type: 'DT_INT32'},
          key_dtype: {type: 'DT_STRING'}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11',
        op: 'NoOp',
        input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall/output_control_node/_2',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/NoOp',
        op: 'NoOp',
        input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
        attr: {
          _acd_function_control_output: {b: true},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/Identity',
        op: 'Identity',
        input: [
          'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
          '^StatefulPartitionedCall/StatefulPartitionedCall/NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
        op: 'Identity',
        input: ['StatefulPartitionedCall/StatefulPartitionedCall/Identity'],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'StatefulPartitionedCall/NoOp',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
        ],
        attr: {
          _has_manual_control_dependencies: {b: true},
          _acd_function_control_output: {b: true}
        }
      },
      {
        name: 'StatefulPartitionedCall/Identity',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
          '^StatefulPartitionedCall/NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall/output/_1',
        op: 'Identity',
        input: ['StatefulPartitionedCall/Identity'],
        attr: {
          T: {type: 'DT_RESOURCE'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input_control_node/_3',
        op: 'NoOp',
        input: ['^114', '^116', '^Func/StatefulPartitionedCall/output/_1'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_4',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/output/_1',
          '^Func/StatefulPartitionedCall_1/input_control_node/_3'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12',
        op: 'NoOp',
        input: ['^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_4',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_5',
        op: 'Identity',
        input: ['114', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {T: {type: 'DT_STRING'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_5',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_STRING'}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_6',
        op: 'Identity',
        input: ['116', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {T: {type: 'DT_INT32'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_6',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_INT32'}}
      },
      {
        name:
            'StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2',
        op: 'LookupTableImportV2',
        input: [
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15'
        ],
        attr: {
          Tout: {type: 'DT_INT32'},
          Tin: {type: 'DT_STRING'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17',
        op: 'NoOp',
        input: [
          '^StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/output_control_node/_8',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'NoOp',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/output_control_node/_2',
          '^Func/StatefulPartitionedCall_1/output_control_node/_8'
        ],
        attr: {
          _has_manual_control_dependencies: {b: true},
          _acd_function_control_output: {b: true}
        }
      },
      {
        name: 'Identity',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/output/_1',
          '^Func/StatefulPartitionedCall_1/output_control_node/_8', '^NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      }
    ],
    versions: {producer: 1240}
  },
  initializerSignature: {
    outputs: {
      'Identity:0': {
        name: 'Identity:0',
        dtype: 'DT_RESOURCE',
        tensorShape: {},
        resourceId: 66
      }
    }
  }
};
