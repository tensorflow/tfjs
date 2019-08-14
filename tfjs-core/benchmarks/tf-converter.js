/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core'], factory) :
  (global = global || self, factory(global.tf = global.tf || {}, global.tf));
}(this, function (exports, tfc) { 'use strict';

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
  const CUSTOM_OPS = {};
  /**
   * Register an Op for graph model executor. This allow you to register
   * TensorFlow custom op or override existing op.
   *
   * Here is an example of registering a new MatMul Op.
   * ```js
   * const customMatmul = (node) =>
   *    tf.matMul(
   *        node.inputs[0], node.inputs[1],
   *        node.attrs['transpose_a'], node.attrs['transpose_b']);
   *
   * tf.registerOp('MatMul', customMatmul);
   * ```
   * The inputs and attrs of the node object is based on the TensorFlow op
   * registry.
   *
   * @param name The Tensorflow Op name.
   * @param opFunc An op function which is called with the current graph node
   * during execution and needs to return a tensor or a list of tensors. The node
   * has the following attributes:
   *    - attr: A map from attribute name to its value
   *    - inputs: A list of input tensors
   */
  /** @doc {heading: 'Models', subheading: 'Op Registry'} */
  function registerOp(name, opFunc) {
      const opMapper = {
          tfOpName: name,
          category: 'custom',
          inputs: [],
          attrs: [],
          customExecutor: opFunc
      };
      CUSTOM_OPS[name] = opMapper;
  }
  /**
   * Retrieve the OpMapper object for the registered op.
   *
   * @param name The Tensorflow Op name.
   */
  /** @doc {heading: 'Models', subheading: 'Op Registry'} */
  function getRegisteredOp(name) {
      return CUSTOM_OPS[name];
  }
  /**
   * Deregister the Op for graph model executor.
   *
   * @param name The Tensorflow Op name.
   */
  /** @doc {heading: 'Models', subheading: 'Op Registry'} */
  function deregisterOp(name) {
      delete CUSTOM_OPS[name];
  }

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
   *
   * =============================================================================
   */
  /** DataType enum. */
  var DataType;
  (function (DataType) {
      DataType[DataType["DT_INVALID"] = 0] = "DT_INVALID";
      DataType[DataType["DT_FLOAT"] = 1] = "DT_FLOAT";
      DataType[DataType["DT_DOUBLE"] = 2] = "DT_DOUBLE";
      DataType[DataType["DT_INT32"] = 3] = "DT_INT32";
      DataType[DataType["DT_UINT8"] = 4] = "DT_UINT8";
      DataType[DataType["DT_INT16"] = 5] = "DT_INT16";
      DataType[DataType["DT_INT8"] = 6] = "DT_INT8";
      DataType[DataType["DT_STRING"] = 7] = "DT_STRING";
      DataType[DataType["DT_COMPLEX64"] = 8] = "DT_COMPLEX64";
      DataType[DataType["DT_INT64"] = 9] = "DT_INT64";
      DataType[DataType["DT_BOOL"] = 10] = "DT_BOOL";
      DataType[DataType["DT_QINT8"] = 11] = "DT_QINT8";
      DataType[DataType["DT_QUINT8"] = 12] = "DT_QUINT8";
      DataType[DataType["DT_QINT32"] = 13] = "DT_QINT32";
      DataType[DataType["DT_BFLOAT16"] = 14] = "DT_BFLOAT16";
      DataType[DataType["DT_FLOAT_REF"] = 101] = "DT_FLOAT_REF";
      DataType[DataType["DT_DOUBLE_REF"] = 102] = "DT_DOUBLE_REF";
      DataType[DataType["DT_INT32_REF"] = 103] = "DT_INT32_REF";
      DataType[DataType["DT_UINT8_REF"] = 104] = "DT_UINT8_REF";
      DataType[DataType["DT_INT16_REF"] = 105] = "DT_INT16_REF";
      DataType[DataType["DT_INT8_REF"] = 106] = "DT_INT8_REF";
      DataType[DataType["DT_STRING_REF"] = 107] = "DT_STRING_REF";
      DataType[DataType["DT_COMPLEX64_REF"] = 108] = "DT_COMPLEX64_REF";
      DataType[DataType["DT_INT64_REF"] = 109] = "DT_INT64_REF";
      DataType[DataType["DT_BOOL_REF"] = 110] = "DT_BOOL_REF";
      DataType[DataType["DT_QINT8_REF"] = 111] = "DT_QINT8_REF";
      DataType[DataType["DT_QUINT8_REF"] = 112] = "DT_QUINT8_REF";
      DataType[DataType["DT_QINT32_REF"] = 113] = "DT_QINT32_REF";
      DataType[DataType["DT_BFLOAT16_REF"] = 114] = "DT_BFLOAT16_REF";
  })(DataType || (DataType = {}));
  var SaverDef;
  (function (SaverDef) {
      /** CheckpointFormatVersion enum. */
      let CheckpointFormatVersion;
      (function (CheckpointFormatVersion) {
          CheckpointFormatVersion[CheckpointFormatVersion["LEGACY"] = 0] = "LEGACY";
          CheckpointFormatVersion[CheckpointFormatVersion["V1"] = 1] = "V1";
          CheckpointFormatVersion[CheckpointFormatVersion["V2"] = 2] = "V2";
      })(CheckpointFormatVersion = SaverDef.CheckpointFormatVersion || (SaverDef.CheckpointFormatVersion = {}));
  })(SaverDef || (SaverDef = {}));

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
  function getParamValue(paramName, node, tensorMap, context) {
      const inputParam = node.inputParams[paramName];
      if (inputParam && inputParam.inputIndexStart !== undefined) {
          const start = inputParam.inputIndexStart;
          const end = inputParam.inputIndexEnd === 0 ?
              undefined :
              (inputParam.inputIndexEnd === undefined ? start + 1 :
                  inputParam.inputIndexEnd);
          if (inputParam.type === 'tensor') {
              return getTensor(node.inputNames[inputParam.inputIndexStart], tensorMap, context);
          }
          if (inputParam.type === 'tensors') {
              const inputs = node.inputNames.slice(start, end);
              return inputs.map(name => getTensor(name, tensorMap, context));
          }
          const data = Array.prototype.slice.call(getTensor(node.inputNames.slice(start)[0], tensorMap, context)
              .dataSync());
          return inputParam.type === 'number' ? data[0] : data;
      }
      const attrParam = node.attrParams[paramName];
      return attrParam && attrParam.value;
  }
  /**
   * Retrieve the tensor based on input name by extracting the node name and
   * output index information.
   * @param name Node input name
   * @param tensorsMap Tensors map keyed by the node
   */
  function getTensor(name, tensorsMap, context) {
      const [nodeName, index] = parseNodeName(name);
      const contextId = context.currentContextIds.find(contextId => {
          return !!tensorsMap[getNodeNameWithContextId(nodeName, contextId)];
      });
      return contextId !== undefined ?
          tensorsMap[getNodeNameWithContextId(nodeName, contextId)][index] :
          undefined;
  }
  /**
   * Retrieve the tensors based on input name for current context.
   * @param name Node input name
   * @param tensorsMap Tensors map keyed by the node
   */
  function getTensorsForCurrentContenxt(name, tensorsMap, context) {
      return tensorsMap[getNodeNameWithContextId(name, context.currentContextId)];
  }
  /**
   * Returns the node name and index from the Node input name.
   * @param inputName The input name of the node, in format of
   * node_name:output_index, i.e. MatMul:0, if the output_index is not set, it is
   * default to 0.
   */
  function getNodeNameAndIndex(inputName, context) {
      const [nodeName, index] = parseNodeName(inputName);
      return [
          getNodeNameWithContextId(nodeName, context && context.currentContextId),
          index
      ];
  }
  function getNodeNameWithContextId(name, contextId) {
      return !!contextId ? `${name}-${contextId}` : name;
  }
  function parseNodeName(name) {
      const index = name.lastIndexOf(':');
      if (index === -1)
          return [name, 0];
      const nodeName = name.substring(0, index);
      return [nodeName, Number(name.substring(index + 1))];
  }
  function split(arr, size) {
      const res = [];
      for (let i = 0; i < arr.length; i += size) {
          res.push(arr.slice(i, i + size));
      }
      return res;
  }

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
  const json = [
      {
          'tfOpName': 'Add',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'AddV2',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'AddN',
          'category': 'arithmetic',
          'inputs': [{ 'start': 0, 'end': 0, 'name': 'tensors', 'type': 'tensors' }]
      },
      {
          'tfOpName': 'BiasAdd',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sub',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'RealDiv',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Div',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'FloorDiv',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Mul',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Maximum',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' }
          ]
      },
      {
          'tfOpName': 'Minimum',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' }
          ]
      },
      {
          'tfOpName': 'Pow',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'SquaredDifference',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Mod',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'FloorMod',
          'category': 'arithmetic',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'T',
                  'name': 'dtype',
                  'type': 'dtype',
                  'notSupported': true
              }]
      }
  ];

  var arithmetic = /*#__PURE__*/Object.freeze({
    json: json
  });

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
  const json$1 = [
      {
          'tfOpName': 'Abs',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Acos',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Asin',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Atan',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Atan2',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'y', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Ceil',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'ClipByValue',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'clip_value_min', 'name': 'clipValueMin', 'type': 'number' },
              { 'tfName': 'clip_value_max', 'name': 'clipValueMax', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'Complex',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'real', 'type': 'tensor' },
              { 'start': 1, 'name': 'imag', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'ComplexAbs',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Cos',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Cosh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Elu',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Exp',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Floor',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Log',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Neg',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Real',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }, {
                  'tfName': 'Tout',
                  'name': 'outputType',
                  'type': 'dtype',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'Relu',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Relu6',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }, {
                  'tfName': 'clipValueMin',
                  'name': 'clipValueMin',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'clipValueMax',
                  'name': 'clipValueMax',
                  'type': 'number',
                  'defaultValue': 6
              }
          ]
      },
      {
          'tfOpName': 'Selu',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sigmoid',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sin',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sinh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sqrt',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Rsqrt',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Square',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Tan',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Tanh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Sign',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Round',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Expm1',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Log1p',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Reciprocal',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Softplus',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Asinh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Acosh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Atanh',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Erf',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Prod',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axes', 'type': 'number[]' },
          ],
          'attrs': [
              {
                  'tfName': 'keep_dims',
                  'name': 'keepDims',
                  'type': 'bool',
                  'notSupported': true
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'LeakyRelu',
          'category': 'basic_math',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'alpha',
                  'name': 'alpha',
                  'type': 'number',
                  'defaultValue': 0.2
              },
              {
                  'tfName': 'T',
                  'name': 'dtype',
                  'type': 'dtype',
                  'notSupported': true
              }
          ]
      }
  ];

  var basicMath = /*#__PURE__*/Object.freeze({
    json: json$1
  });

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
  const json$2 = [
      {
          'tfOpName': 'LoopCond',
          'category': 'control',
          'inputs': [{ 'start': 0, 'name': 'pred', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'Switch',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'data', 'type': 'tensor' },
              { 'start': 1, 'name': 'pred', 'type': 'tensor' }
          ]
      },
      {
          'tfOpName': 'Merge',
          'category': 'control',
          'inputs': [{ 'start': 0, 'end': 0, 'name': 'tensors', 'type': 'tensors' }]
      },
      {
          'tfOpName': 'Enter',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensor', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true },
              { 'tfName': 'frame_name', 'name': 'frameName', 'type': 'string' },
              { 'tfName': 'is_constant', 'name': 'isConstant', 'type': 'bool' }
          ]
      },
      {
          'tfOpName': 'Exit',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensor', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'NextIteration',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensor', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'TensorArrayV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'size', 'type': 'number' },
          ],
          'attrs': [
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' },
              { 'tfName': 'element_shape', 'name': 'elementShape', 'type': 'shape' },
              { 'tfName': 'dynamic_size', 'name': 'dynamicSize', 'type': 'bool' },
              { 'tfName': 'clear_after_read', 'name': 'clearAfterRead', 'type': 'bool' },
              {
                  'tfName': 'identical_element_shapes',
                  'name': 'identicalElementShapes',
                  'type': 'bool'
              },
              { 'tfName': 'tensor_array_name', 'name': 'name', 'type': 'string' }
          ]
      },
      {
          'tfOpName': 'TensorArrayWriteV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'index', 'type': 'number' },
              { 'start': 2, 'name': 'tensor', 'type': 'tensor' },
              { 'start': 3, 'name': 'flowIn', 'type': 'number' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'TensorArrayReadV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'index', 'type': 'number' },
              { 'start': 2, 'name': 'flowIn', 'type': 'number' },
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
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'indices', 'type': 'number[]' },
              { 'start': 2, 'name': 'flowIn', 'type': 'number' },
          ],
          'attrs': [
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' },
              { 'tfName': 'element_shape', 'name': 'elementShape', 'type': 'shape' }
          ]
      },
      {
          'tfOpName': 'TensorArrayScatterV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'indices', 'type': 'number[]' },
              { 'start': 2, 'name': 'tensor', 'type': 'tensor' },
              { 'start': 3, 'name': 'flowIn', 'type': 'number' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'TensorArrayConcatV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'flowIn', 'type': 'number' },
          ],
          'attrs': [
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' }, {
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
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'tensor', 'type': 'tensor' },
              { 'start': 2, 'name': 'lengths', 'type': 'number[]' },
              { 'start': 3, 'name': 'flowIn', 'type': 'number' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'TensorArraySizeV3',
          'category': 'control',
          'inputs': [
              { 'start': 0, 'name': 'tensorArrayId', 'type': 'number' },
              { 'start': 1, 'name': 'flowIn', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'TensorArrayCloseV3',
          'category': 'control',
          'inputs': [{ 'start': 0, 'name': 'tensorArrayId', 'type': 'number' }]
      }
  ];

  var control = /*#__PURE__*/Object.freeze({
    json: json$2
  });

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
  const json$3 = [
      {
          'tfOpName': 'AvgPool',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              },
              { 'tfName': 'ksize', 'name': 'kernelSize', 'type': 'number[]' },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'MaxPool',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              },
              { 'tfName': 'ksize', 'name': 'kernelSize', 'type': 'number[]' },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Conv1D',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'filter', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'stride', 'name': 'stride', 'type': 'number' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'defaultValue': 'NWC'
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }, {
                  'tfName': 'dilation',
                  'name': 'dilation',
                  'type': 'number',
                  'defaultValue': 1
              }
          ]
      },
      {
          'tfOpName': 'Conv2D',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'filter', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true },
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' },
              { 'tfName': 'useCudnnOnGpu', 'name': 'useCudnnOnGpu', 'type': 'bool' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'defaultValue': 'NHWC'
              },
              { 'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'Conv2DBackpropInput',
          'category': 'convolution',
          'inputs': [
              { 'start': 2, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'filter', 'type': 'tensor' },
              { 'start': 0, 'name': 'outputShape', 'type': 'number[]' },
          ],
          'attrs': [
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'DepthwiseConv2d',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'input', 'type': 'tensor' },
              { 'start': 1, 'name': 'filter', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'defaultValue': 'NHWC'
              },
              { 'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'DepthwiseConv2dNative',
          'category': 'convolution',
          'inputs': [
              { 'start': 0, 'name': 'input', 'type': 'tensor' },
              { 'start': 1, 'name': 'filter', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'strides', 'name': 'strides', 'type': 'number[]' },
              { 'tfName': 'padding', 'name': 'pad', 'type': 'string' }, {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'defaultValue': 'NHWC'
              },
              { 'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]' }
          ]
      }
  ];

  var convolution = /*#__PURE__*/Object.freeze({
    json: json$3
  });

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
  const json$4 = [
      {
          'tfOpName': 'Fill',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'shape', 'type': 'number[]' },
              { 'start': 1, 'name': 'value', 'type': 'number' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'LinSpace',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'start', 'type': 'number' },
              { 'start': 1, 'name': 'stop', 'type': 'number' },
              { 'start': 2, 'name': 'num', 'type': 'number' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'OneHot',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'indices', 'type': 'tensor' },
              { 'start': 1, 'name': 'depth', 'type': 'number' },
              { 'start': 2, 'name': 'onValue', 'type': 'number', 'defaultValue': 1 },
              { 'start': 3, 'name': 'offValue', 'type': 'number', 'defaultValue': 0 },
          ],
          'attrs': [
              {
                  'tfName': 'axis',
                  'name': 'axis',
                  'type': 'number',
                  'notSupported': true
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Ones',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'shape', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'OnesLike',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [{ 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'RandomUniform',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'shape', 'type': 'number[]' },
          ],
          'attrs': [
              {
                  'tfName': 'minval',
                  'name': 'minval',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'maxval',
                  'name': 'maxval',
                  'type': 'number',
                  'defaultValue': 1
              },
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' },
              { 'tfName': 'seed', 'name': 'seed', 'type': 'number', 'defaultValue': 0 }, {
                  'tfName': 'seed2',
                  'name': 'seed2',
                  'type': 'number',
                  'defaultValue': 0,
                  'notSupported': true
              },
              { 'tfName': 'T', 'name': 'T', 'type': 'number', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Range',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'start', 'type': 'number' },
              { 'start': 1, 'name': 'stop', 'type': 'number' },
              { 'start': 2, 'name': 'step', 'type': 'number', 'defaultValue': 0 },
          ],
          'attrs': [{ 'tfName': 'Tidx', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'TruncatedNormal',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'shape', 'type': 'number[]' },
          ],
          'attrs': [
              {
                  'tfName': 'means',
                  'name': 'mean',
                  'type': 'number',
                  'defaultValue': 0.0
              },
              {
                  'tfName': 'stddev',
                  'name': 'stdDev',
                  'type': 'number',
                  'defaultValue': 1.0
              },
              { 'tfName': 'seed', 'name': 'seed', 'type': 'number' }, {
                  'tfName': 'seed2',
                  'name': 'seed2',
                  'type': 'number',
                  'defaultValue': 0,
                  'notSupported': true
              },
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' },
              { 'tfName': 'T', 'name': 'T', 'type': 'number', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Zeros',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'shape', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      },
      {
          'tfOpName': 'ZerosLike',
          'category': 'creation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [{ 'tfName': 'T', 'name': 'dtype', 'type': 'dtype' }]
      }
  ];

  var creation = /*#__PURE__*/Object.freeze({
    json: json$4
  });

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
  const json$5 = [
      {
          'tfOpName': 'NonMaxSuppressionV2',
          'category': 'dynamic',
          'inputs': [
              { 'start': 0, 'name': 'boxes', 'type': 'tensor' },
              { 'start': 1, 'name': 'scores', 'type': 'tensor' },
              { 'start': 2, 'name': 'maxOutputSize', 'type': 'number' },
              { 'start': 3, 'name': 'iouThreshold', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'NonMaxSuppressionV3',
          'category': 'dynamic',
          'inputs': [
              { 'start': 0, 'name': 'boxes', 'type': 'tensor' },
              { 'start': 1, 'name': 'scores', 'type': 'tensor' },
              { 'start': 2, 'name': 'maxOutputSize', 'type': 'number' },
              { 'start': 3, 'name': 'iouThreshold', 'type': 'number' },
              { 'start': 4, 'name': 'scoreThreshold', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'Where',
          'category': 'dynamic',
          'inputs': [
              { 'start': 0, 'name': 'condition', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'ListDiff',
          'category': 'dynamic',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'y', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'T',
                  'name': 'dtype',
                  'type': 'dtype',
                  'notSupported': true
              }]
      }
  ];

  var dynamic = /*#__PURE__*/Object.freeze({
    json: json$5
  });

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
  const json$6 = [{
          'tfOpName': 'TopKV2',
          'category': 'evaluation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'k', 'type': 'number' },
          ],
          'attrs': [{ 'tfName': 'sorted', 'name': 'sorted', 'type': 'bool' }]
      }];

  var evaluation = /*#__PURE__*/Object.freeze({
    json: json$6
  });

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
  const json$7 = [
      {
          'tfOpName': 'PlaceholderWithDefault',
          'category': 'graph',
          'inputs': [
              { 'start': 0, 'name': 'default', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'shape', 'name': 'shape', 'type': 'shape' },
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' }
          ]
      },
      {
          'tfOpName': 'Placeholder',
          'category': 'graph',
          'attrs': [
              { 'tfName': 'shape', 'name': 'shape', 'type': 'shape' },
              { 'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype' }
          ]
      },
      { 'tfOpName': 'Const', 'category': 'graph' }, {
          'tfOpName': 'Identity',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'IdentityN',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'end': 0, 'name': 'x', 'type': 'tensors' }]
      },
      {
          'tfOpName': 'Snapshot',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'Rank',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'Size',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'Shape',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'ShapeN',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'end': 0, 'name': 'x', 'type': 'tensors' }]
      },
      {
          'tfOpName': 'Print',
          'category': 'graph',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'data', 'type': 'tensors' },
          ],
          'attrs': [
              { 'tfName': 'message', 'name': 'message', 'type': 'string' }, {
                  'tfName': 'first_n',
                  'name': 'firstN',
                  'type': 'number',
                  'notSupported': true
              },
              {
                  'tfName': 'summarize',
                  'name': 'summarize',
                  'type': 'number',
                  'defaultValue': 3
              }
          ]
      },
      { 'tfOpName': 'NoOp', 'category': 'graph', 'inputs': [] }, {
          'tfOpName': 'StopGradient',
          'category': 'graph',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'FakeQuantWithMinMaxVars',
          'category': 'graph',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'min', 'name': 'min', 'type': 'number' },
              { 'tfName': 'max', 'name': 'max', 'type': 'number' }
          ]
      }
  ];

  var graph = /*#__PURE__*/Object.freeze({
    json: json$7
  });

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
  const json$8 = [
      {
          'tfOpName': 'ResizeBilinear',
          'category': 'image',
          'inputs': [
              { 'start': 0, 'name': 'images', 'type': 'tensor' },
              { 'start': 1, 'name': 'size', 'type': 'number[]' },
          ],
          'attrs': [
              { 'tfName': 'align_corners', 'name': 'alignCorners', 'type': 'bool' },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'ResizeNearestNeighbor',
          'category': 'image',
          'inputs': [
              { 'start': 0, 'name': 'images', 'type': 'tensor' },
              { 'start': 1, 'name': 'size', 'type': 'number[]' },
          ],
          'attrs': [
              { 'tfName': 'align_corners', 'name': 'alignCorners', 'type': 'bool' },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'CropAndResize',
          'category': 'image',
          'inputs': [
              { 'start': 0, 'name': 'image', 'type': 'tensor' },
              { 'start': 1, 'name': 'boxes', 'type': 'tensor' },
              { 'start': 2, 'name': 'boxInd', 'type': 'tensor' },
              { 'start': 3, 'name': 'cropSize', 'type': 'number[]' },
          ],
          'attrs': [
              { 'tfName': 'method', 'name': 'method', 'type': 'string' }, {
                  'tfName': 'extrapolation_value',
                  'name': 'extrapolationValue',
                  'type': 'number'
              }
          ]
      }
  ];

  var image = /*#__PURE__*/Object.freeze({
    json: json$8
  });

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
  const json$9 = [
      {
          'tfOpName': 'Equal',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'NotEqual',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Greater',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'GreaterEqual',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Less',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'LessEqual',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'LogicalAnd',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'LogicalNot',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'LogicalOr',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Select',
          'category': 'logical',
          'inputs': [
              { 'start': 0, 'name': 'condition', 'type': 'tensor' },
              { 'start': 1, 'name': 'a', 'type': 'tensor' },
              { 'start': 2, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'T',
                  'name': 'dtype',
                  'type': 'dtype',
                  'notSupported': true
              }]
      }
  ];

  var logical = /*#__PURE__*/Object.freeze({
    json: json$9
  });

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
  const json$a = [
      {
          'tfOpName': 'MatMul',
          'category': 'matrices',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'transpose_a',
                  'name': 'transposeA',
                  'type': 'bool',
                  'defaultValue': false
              },
              {
                  'tfName': 'transpose_b',
                  'name': 'transposeB',
                  'type': 'bool',
                  'defaultValue': false
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'BatchMatMul',
          'category': 'matrices',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'adj_x',
                  'name': 'transposeA',
                  'type': 'bool',
                  'defaultValue': false
              },
              {
                  'tfName': 'adj_y',
                  'name': 'transposeB',
                  'type': 'bool',
                  'defaultValue': false
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'BatchMatMulV2',
          'category': 'matrices',
          'inputs': [
              { 'start': 0, 'name': 'a', 'type': 'tensor' },
              { 'start': 1, 'name': 'b', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'adj_x',
                  'name': 'transposeA',
                  'type': 'bool',
                  'defaultValue': false
              },
              {
                  'tfName': 'adj_y',
                  'name': 'transposeB',
                  'type': 'bool',
                  'defaultValue': false
              },
              { 'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'Transpose',
          'category': 'matrices',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'perm', 'type': 'number[]' },
          ],
          'attrs': [{
                  'tfName': 'T',
                  'name': 'dtype',
                  'type': 'dtype',
                  'notSupported': true
              }]
      }
  ];

  var matrices = /*#__PURE__*/Object.freeze({
    json: json$a
  });

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
  const json$b = [
      {
          'tfOpName': 'FusedBatchNorm',
          'category': 'normalization',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'scale', 'type': 'tensor' },
              { 'start': 2, 'name': 'offset', 'type': 'tensor' },
              { 'start': 3, 'name': 'mean', 'type': 'tensor' },
              { 'start': 4, 'name': 'variance', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'epsilon',
                  'name': 'epsilon',
                  'type': 'number',
                  'defaultValue': 0.001
              },
              {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'FusedBatchNormV2',
          'category': 'normalization',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'scale', 'type': 'tensor' },
              { 'start': 2, 'name': 'offset', 'type': 'tensor' },
              { 'start': 3, 'name': 'mean', 'type': 'tensor' },
              { 'start': 4, 'name': 'variance', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'epsilon',
                  'name': 'epsilon',
                  'type': 'number',
                  'defaultValue': 0.001
              },
              {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'FusedBatchNormV3',
          'category': 'normalization',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'scale', 'type': 'tensor' },
              { 'start': 2, 'name': 'offset', 'type': 'tensor' },
              { 'start': 3, 'name': 'mean', 'type': 'tensor' },
              { 'start': 4, 'name': 'variance', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'epsilon',
                  'name': 'epsilon',
                  'type': 'number',
                  'defaultValue': 0.001
              },
              {
                  'tfName': 'data_format',
                  'name': 'dataFormat',
                  'type': 'string',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'LRN',
          'category': 'normalization',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'depth_radius',
                  'name': 'radius',
                  'type': 'number',
                  'defaultValue': 5
              },
              { 'tfName': 'bias', 'name': 'bias', 'type': 'number', 'defaultValue': 1.0 },
              {
                  'tfName': 'alpha',
                  'name': 'alpha',
                  'type': 'number',
                  'defaultValue': 1.0
              },
              {
                  'tfName': 'beta',
                  'name': 'beta',
                  'type': 'number',
                  'defaultValue': 0.5
              }
          ]
      },
      {
          'tfOpName': 'Softmax',
          'category': 'normalization',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'LogSoftmax',
          'category': 'normalization',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'SparseToDense',
          'category': 'normalization',
          'inputs': [
              { 'start': 0, 'name': 'sparseIndices', 'type': 'tensor' },
              { 'start': 1, 'name': 'outputShape', 'type': 'number[]' },
              { 'start': 2, 'name': 'sparseValues', 'type': 'tensor' },
              { 'start': 3, 'name': 'defaultValue', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'validate_indices',
                  'name': 'validateIndices',
                  'type': 'bool',
                  'defaultValue': true,
                  'notSupported': true
              }]
      }
  ];

  var normalization = /*#__PURE__*/Object.freeze({
    json: json$b
  });

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
  const json$c = [
      {
          'tfOpName': 'Max',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'Mean',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'Min',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'Sum',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'All',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'Any',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      },
      {
          'tfOpName': 'ArgMax',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'ArgMin',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'Prod',
          'category': 'reduction',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' },
          ],
          'attrs': [{ 'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool' }]
      }
  ];

  var reduction = /*#__PURE__*/Object.freeze({
    json: json$c
  });

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
  const json$d = [
      {
          'tfOpName': 'ConcatV2',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'end': -1, 'name': 'tensors', 'type': 'tensors' },
              { 'start': -1, 'name': 'axis', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'Concat',
          'category': 'slice_join',
          'inputs': [
              { 'start': 1, 'end': 0, 'name': 'tensors', 'type': 'tensors' },
              { 'start': 0, 'name': 'axis', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'GatherV2',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'indices', 'type': 'tensor' },
              { 'start': 2, 'name': 'axis', 'type': 'number', 'defaultValue': 0 }
          ]
      },
      {
          'tfOpName': 'Gather',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'indices', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'axis', 'name': 'axis', 'type': 'number', 'defaultValue': 0 }, {
                  'tfName': 'validate_indices',
                  'name': 'validateIndices',
                  'type': 'bool',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'Reverse',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'dims', 'type': 'bool', 'notSupported': true }
          ]
      },
      {
          'tfOpName': 'ReverseV2',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'Slice',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'begin', 'type': 'number[]' },
              { 'start': 2, 'name': 'size', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'StridedSlice',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'begin', 'type': 'number[]' },
              { 'start': 2, 'name': 'end', 'type': 'number[]' },
              { 'start': 3, 'name': 'strides', 'type': 'number[]' },
          ],
          'attrs': [
              {
                  'tfName': 'begin_mask',
                  'name': 'beginMask',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'end_mask',
                  'name': 'endMask',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'new_axis_mask',
                  'name': 'newAxisMask',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'ellipsis_mask',
                  'name': 'ellipsisMask',
                  'type': 'number',
                  'defaultValue': 0
              },
              {
                  'tfName': 'shrink_axis_mask',
                  'name': 'shrinkAxisMask',
                  'type': 'number',
                  'defaultValue': 0
              }
          ]
      },
      {
          'tfOpName': 'Pack',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'end': 0, 'name': 'tensors', 'type': 'tensors' },
          ],
          'attrs': [
              { 'tfName': 'axis', 'name': 'axis', 'type': 'number', 'defaultValue': 0 }
          ]
      },
      {
          'tfOpName': 'Unpack',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'tensor', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'axis', 'name': 'axis', 'type': 'number', 'defaultValue': 0 }, {
                  'tfName': 'num',
                  'name': 'num',
                  'type': 'number',
                  'defaultValue': 0,
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'Tile',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'reps', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'Split',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'axis', 'type': 'number', 'defaultValue': 0 },
              { 'start': 1, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'num_split',
                  'name': 'numOrSizeSplits',
                  'type': 'number',
                  'defaultValue': 1
              }]
      },
      {
          'tfOpName': 'SplitV',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'numOrSizeSplits', 'type': 'number[]' },
              { 'start': 2, 'name': 'axis', 'type': 'number', 'defaultValue': 0 }
          ]
      },
      {
          'tfOpName': 'ScatterNd',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'indices', 'type': 'tensor' },
              { 'start': 1, 'name': 'values', 'type': 'tensor' },
              { 'start': 2, 'name': 'shape', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'GatherNd',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'indices', 'type': 'tensor' }
          ]
      },
      {
          'tfOpName': 'SparseToDense',
          'category': 'slice_join',
          'inputs': [
              { 'start': 0, 'name': 'sparseIndices', 'type': 'tensor' },
              { 'start': 1, 'name': 'outputShape', 'type': 'number[]' },
              { 'start': 2, 'name': 'sparseValues', 'type': 'tensor' },
              { 'start': 3, 'name': 'defaultValue', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'validate_indices',
                  'name': 'validateIndices',
                  'type': 'bool',
                  'defaultValue': false,
                  'notSupported': true
              }]
      }
  ];

  var sliceJoin = /*#__PURE__*/Object.freeze({
    json: json$d
  });

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
  const json$e = [
      {
          'tfOpName': 'FFT',
          'category': 'spectral',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'IFFT',
          'category': 'spectral',
          'inputs': [{ 'start': 0, 'name': 'x', 'type': 'tensor' }]
      },
      {
          'tfOpName': 'RFFT',
          'category': 'spectral',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' }, {
                  'start': 1,
                  'name': 'fft_length',
                  'type': 'number',
                  'notSupported': true
              }
          ]
      },
      {
          'tfOpName': 'IRFFT',
          'category': 'spectral',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' }, {
                  'start': 1,
                  'name': 'fft_length',
                  'type': 'number',
                  'notSupported': true
              }
          ]
      }
  ];

  var spectral = /*#__PURE__*/Object.freeze({
    json: json$e
  });

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
  const json$f = [
      {
          'tfOpName': 'Cast',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              {
                  'tfName': 'SrcT',
                  'name': 'sdtype',
                  'type': 'dtype',
                  'notSupported': true
              },
              { 'tfName': 'DstT', 'name': 'dtype', 'type': 'dtype' }
          ]
      },
      {
          'tfOpName': 'ExpandDims',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'axis', 'type': 'number' }
          ]
      },
      {
          'tfOpName': 'Pad',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'padding', 'type': 'number[]' },
          ],
          'attrs': [{
                  'tfName': 'constant_value',
                  'name': 'constantValue',
                  'type': 'number',
                  'defaultValue': 0
              }]
      },
      {
          'tfOpName': 'PadV2',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'padding', 'type': 'number[]' }, {
                  'start': 2,
                  'name': 'constantValue',
                  'type': 'number',
                  'defaultValue': 0
              }
          ]
      },
      {
          'tfOpName': 'Reshape',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'shape', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'Squeeze',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [{
                  'tfName': 'axis',
                  'tfDeprecatedName': 'squeeze_dims',
                  'name': 'axis',
                  'type': 'number[]'
              }]
      },
      {
          'tfOpName': 'SpaceToBatchND',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'blockShape', 'type': 'number[]' },
              { 'start': 2, 'name': 'paddings', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'BatchToSpaceND',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
              { 'start': 1, 'name': 'blockShape', 'type': 'number[]' },
              { 'start': 2, 'name': 'crops', 'type': 'number[]' }
          ]
      },
      {
          'tfOpName': 'DepthToSpace',
          'category': 'transformation',
          'inputs': [
              { 'start': 0, 'name': 'x', 'type': 'tensor' },
          ],
          'attrs': [
              { 'tfName': 'block_size', 'name': 'blockSize', 'type': 'number' },
              { 'tfName': 'data_format', 'name': 'dataFormat', 'type': 'string' }
          ]
      }
  ];

  var transformation = /*#__PURE__*/Object.freeze({
    json: json$f
  });

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
  class OperationMapper {
      // Singleton instance for the mapper
      static get Instance() {
          return this._instance || (this._instance = new this());
      }
      // Loads the op mapping from the JSON file.
      constructor() {
          const ops = [
              arithmetic, basicMath, control, convolution, creation, dynamic,
              evaluation, logical, image, graph, matrices, normalization, reduction,
              sliceJoin, spectral, transformation
          ];
          const mappersJson = [].concat.apply([], ops.map(op => op.json));
          this.opMappers = mappersJson.reduce((map, mapper) => {
              map[mapper.tfOpName] = mapper;
              return map;
          }, {});
      }
      // Converts the model from Tensorflow GraphDef to local representation for
      // TensorFlow.js API
      transformGraph(graph) {
          const tfNodes = graph.node;
          const placeholders = [];
          const weights = [];
          const nodes = tfNodes.reduce((map, node) => {
              map[node.name] = this.mapNode(node);
              if (node.op === 'Placeholder') {
                  placeholders.push(map[node.name]);
              }
              if (node.op === 'Const') {
                  weights.push(map[node.name]);
              }
              return map;
          }, {});
          const inputs = [];
          const outputs = [];
          const allNodes = Object.keys(nodes);
          allNodes.forEach(key => {
              const node = nodes[key];
              node.inputNames.forEach(name => {
                  const [nodeName,] = getNodeNameAndIndex(name);
                  node.inputs.push(nodes[nodeName]);
                  nodes[nodeName].children.push(node);
              });
              if (node.inputs.length === 0)
                  inputs.push(node);
          });
          allNodes.forEach(key => {
              const node = nodes[key];
              if (node.children.length === 0)
                  outputs.push(node);
          });
          return { nodes, inputs, outputs, weights, placeholders };
      }
      mapNode(node) {
          // Unsupported ops will cause an error at run-time (not parse time), since
          // they may not be used by the actual execution subgraph.
          const mapper = getRegisteredOp(node.op) || this.opMappers[node.op] || {};
          if (node.attr == null) {
              node.attr = {};
          }
          const newNode = {
              name: node.name,
              op: node.op,
              category: mapper.category,
              inputNames: (node.input ||
                  []).map(input => input.startsWith('^') ? input.substr(1) : input),
              inputs: [],
              children: [],
              inputParams: {},
              attrParams: {},
              rawAttrs: node.attr
          };
          if (mapper.inputs != null) {
              newNode.inputParams =
                  mapper.inputs.reduce((map, param) => {
                      map[param.name] = {
                          type: param.type,
                          inputIndexStart: param.start,
                          inputIndexEnd: param.end
                      };
                      return map;
                  }, {});
          }
          if (mapper.attrs != null) {
              newNode.attrParams =
                  mapper.attrs.reduce((map, param) => {
                      const type = param.type;
                      let value = undefined;
                      switch (param.type) {
                          case 'string':
                              value = getStringParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getStringParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'string[]':
                              value = getStringArrayParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getStringArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'number':
                              value = getNumberParam(node.attr, param.tfName, (param.defaultValue || 0));
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getNumberParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'number[]':
                              value = getNumericArrayParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getNumericArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'bool':
                              value = getBoolParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getBoolParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'bool[]':
                              value = getBoolArrayParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getBoolArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'shape':
                              value = getTensorShapeParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getTensorShapeParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'shape[]':
                              value = getTensorShapeArrayParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getTensorShapeArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'dtype':
                              value = getDtypeParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getDtypeParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'dtype[]':
                              value = getDtypeArrayParam(node.attr, param.tfName, param.defaultValue);
                              if (value === undefined && !!param.tfDeprecatedName) {
                                  value = getDtypeArrayParam(node.attr, param.tfDeprecatedName, param.defaultValue);
                              }
                              break;
                          case 'tensor':
                          case 'tensors':
                              break;
                          default:
                              throw new Error(`Unsupported param type: ${param.type} for op: ${node.op}`);
                      }
                      map[param.name] = { value, type };
                      return map;
                  }, {});
          }
          return newNode;
      }
  }
  function decodeBase64(text) {
      // tslint:disable-next-line:no-any
      const global = tfc.ENV.global;
      if (typeof global.atob !== 'undefined') {
          return global.atob(text);
      }
      else if (typeof Buffer !== 'undefined') {
          return new Buffer(text, 'base64').toString();
      }
      else {
          throw new Error('Unable to decode base64 in this environment. ' +
              'Missing built-in atob() or Buffer()');
      }
  }
  function parseStringParam(s, keepCase) {
      const value = Array.isArray(s) ? String.fromCharCode.apply(null, s) : decodeBase64(s);
      return keepCase ? value : value.toLowerCase();
  }
  function getStringParam(attrs, name, def, keepCase = false) {
      const param = attrs[name];
      if (param != null) {
          return parseStringParam(param.s, keepCase);
      }
      return def;
  }
  function getBoolParam(attrs, name, def) {
      const param = attrs[name];
      return param ? param.b : def;
  }
  function getNumberParam(attrs, name, def) {
      const param = attrs[name] || {};
      const value = param['i'] != null ? param['i'] : (param['f'] != null ? param['f'] : def);
      return (typeof value === 'number') ? value :
          parseInt(value, 10);
  }
  function parseDtypeParam(value) {
      if (typeof (value) === 'string') {
          // tslint:disable-next-line:no-any
          value = DataType[value];
      }
      switch (value) {
          case DataType.DT_FLOAT:
              return 'float32';
          case DataType.DT_INT32:
              return 'int32';
          case DataType.DT_BOOL:
              return 'bool';
          case DataType.DT_DOUBLE:
              return 'float32';
          case DataType.DT_STRING:
              return 'string';
          default:
              // Unknown dtype error will happen at runtime (instead of parse time),
              // since these nodes might not be used by the actual subgraph execution.
              return null;
      }
  }
  function getDtypeParam(attrs, name, def) {
      const param = attrs[name];
      if (param && param.type) {
          return parseDtypeParam(param.type);
      }
      return def;
  }
  function getDtypeArrayParam(attrs, name, def) {
      const param = attrs[name];
      if (param && param.list && param.list.type) {
          return param.list.type.map(v => parseDtypeParam(v));
      }
      return def;
  }
  function parseTensorShapeParam(shape) {
      if (shape.unknownRank) {
          return undefined;
      }
      if (shape.dim != null) {
          return shape.dim.map(dim => (typeof dim.size === 'number') ?
              dim.size :
              parseInt(dim.size, 10));
      }
      return [];
  }
  function getTensorShapeParam(attrs, name, def) {
      const param = attrs[name];
      if (param && param.shape) {
          return parseTensorShapeParam(param.shape);
      }
      return def;
  }
  function getNumericArrayParam(attrs, name, def) {
      const param = attrs[name];
      if (param) {
          return ((param.list.f && param.list.f.length ? param.list.f : param.list.i))
              .map(v => (typeof v === 'number') ? v :
              parseInt(v, 10));
      }
      return def;
  }
  function getStringArrayParam(attrs, name, def, keepCase = false) {
      const param = attrs[name];
      if (param && param.list && param.list.s) {
          return param.list.s.map((v) => {
              return parseStringParam(v, keepCase);
          });
      }
      return def;
  }
  function getTensorShapeArrayParam(attrs, name, def) {
      const param = attrs[name];
      if (param && param.list && param.list.shape) {
          return param.list.shape.map((v) => {
              return parseTensorShapeParam(v);
          });
      }
      return def;
  }
  function getBoolArrayParam(attrs, name, def) {
      const param = attrs[name];
      if (param && param.list && param.list.b) {
          return param.list.b;
      }
      return def;
  }

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
  /**
   * Helper class for lookup inputs and params for nodes in the model graph.
   */
  class NodeValueImpl {
      constructor(node, tensorMap, context) {
          this.node = node;
          this.tensorMap = tensorMap;
          this.context = context;
          this.inputs = [];
          this.attrs = {};
          this.inputs = node.inputNames.map(name => this.getInput(name));
          if (node.rawAttrs != null) {
              this.attrs = Object.keys(node.rawAttrs)
                  .reduce((attrs, key) => {
                  attrs[key] = this.getAttr(key);
                  return attrs;
              }, {});
          }
      }
      /**
       * Return the value of the attribute or input param.
       * @param name String: name of attribute or input param.
       */
      getInput(name) {
          return getTensor(name, this.tensorMap, this.context);
      }
      /**
       * Return the value of the attribute or input param.
       * @param name String: name of attribute or input param.
       */
      getAttr(name, defaultValue) {
          const value = this.node.rawAttrs[name];
          if (value.tensor != null) {
              return getTensor(name, this.tensorMap, this.context);
          }
          if (value.i != null || value.f != null) {
              return getNumberParam(this.node.rawAttrs, name, defaultValue);
          }
          if (value.s != null) {
              return getStringParam(this.node.rawAttrs, name, defaultValue);
          }
          if (value.b != null) {
              return getBoolParam(this.node.rawAttrs, name, defaultValue);
          }
          if (value.shape != null) {
              return getTensorShapeParam(this.node.rawAttrs, name, defaultValue);
          }
          if (value.type != null) {
              return getDtypeParam(this.node.rawAttrs, name, defaultValue);
          }
          if (value.list != null) {
              if (value.list.i != null || value.list.f != null) {
                  return getNumericArrayParam(this.node.rawAttrs, name, defaultValue);
              }
              if (value.list.s != null) {
                  return getStringArrayParam(this.node.rawAttrs, name, defaultValue);
              }
              if (value.list.shape != null) {
                  return getTensorShapeArrayParam(this.node.rawAttrs, name, defaultValue);
              }
              if (value.list.b != null) {
                  return getBoolArrayParam(this.node.rawAttrs, name, defaultValue);
              }
              if (value.list.type != null) {
                  return getDtypeArrayParam(this.node.rawAttrs, name, defaultValue);
              }
          }
          return defaultValue;
      }
  }

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
  let executeOp = (node, tensorMap, context) => {
      switch (node.op) {
          case 'BiasAdd':
          case 'AddV2':
          case 'Add': {
              return [tfc.add(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'AddN': {
              return [tfc.addN(getParamValue('tensors', node, tensorMap, context))];
          }
          case 'FloorMod':
          case 'Mod':
              return [tfc.mod(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          case 'Mul':
              return [tfc.mul(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          case 'RealDiv':
          case 'Div': {
              return [tfc.div(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'FloorDiv': {
              return [tfc.floorDiv(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Sub': {
              return [tfc.sub(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Minimum': {
              return [tfc.minimum(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Maximum': {
              return [tfc.maximum(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Pow': {
              return [tfc.pow(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'SquaredDifference': {
              return [tfc.squaredDifference(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$1 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Abs':
          case 'ComplexAbs':
              return [tfc.abs(getParamValue('x', node, tensorMap, context))];
          case 'Acos':
              return [tfc.acos(getParamValue('x', node, tensorMap, context))];
          case 'Acosh':
              return [tfc.acosh(getParamValue('x', node, tensorMap, context))];
          case 'Asin':
              return [tfc.asin(getParamValue('x', node, tensorMap, context))];
          case 'Asinh':
              return [tfc.asinh(getParamValue('x', node, tensorMap, context))];
          case 'Atan':
              return [tfc.atan(getParamValue('x', node, tensorMap, context))];
          case 'Atan2':
              return [tfc.atan2(getParamValue('x', node, tensorMap, context), getParamValue('y', node, tensorMap, context))];
          case 'Atanh':
              return [tfc.atanh(getParamValue('x', node, tensorMap, context))];
          case 'Ceil':
              return [tfc.ceil(getParamValue('x', node, tensorMap, context))];
          case 'Complex':
              return [tfc.complex(getParamValue('real', node, tensorMap, context), getParamValue('imag', node, tensorMap, context))];
          case 'Cos':
              return [tfc.cos(getParamValue('x', node, tensorMap, context))];
          case 'Cosh':
              return [tfc.cosh(getParamValue('x', node, tensorMap, context))];
          case 'Elu':
              return [tfc.elu(getParamValue('x', node, tensorMap, context))];
          case 'Erf':
              return [tfc.erf(getParamValue('x', node, tensorMap, context))];
          case 'Exp':
              return [tfc.exp(getParamValue('x', node, tensorMap, context))];
          case 'Expm1': {
              return [tfc.expm1(getParamValue('x', node, tensorMap, context))];
          }
          case 'Floor':
              return [tfc.floor(getParamValue('x', node, tensorMap, context))];
          case 'Log':
              return [tfc.log(getParamValue('x', node, tensorMap, context))];
          case 'Log1p': {
              return [tfc.log1p(getParamValue('x', node, tensorMap, context))];
          }
          case 'Neg':
              return [tfc.neg(getParamValue('x', node, tensorMap, context))];
          case 'Reciprocal': {
              return [tfc.reciprocal(getParamValue('x', node, tensorMap, context))];
          }
          case 'Real':
              return [tfc.real(getParamValue('x', node, tensorMap, context))];
          case 'Relu':
              return [tfc.relu(getParamValue('x', node, tensorMap, context))];
          case 'Round': {
              return [tfc.round(getParamValue('x', node, tensorMap, context))];
          }
          case 'Selu':
              return [tfc.selu(getParamValue('x', node, tensorMap, context))];
          case 'Sigmoid':
              return [tfc.sigmoid(getParamValue('x', node, tensorMap, context))];
          case 'Sin':
              return [tfc.sin(getParamValue('x', node, tensorMap, context))];
          case 'Sign': {
              return [tfc.sign(getParamValue('x', node, tensorMap, context))];
          }
          case 'Sinh': {
              return [tfc.sinh(getParamValue('x', node, tensorMap, context))];
          }
          case 'Softplus': {
              return [tfc.softplus(getParamValue('x', node, tensorMap, context))];
          }
          case 'Sqrt': {
              return [tfc.sqrt(getParamValue('x', node, tensorMap, context))];
          }
          case 'Square': {
              return [tfc.square(getParamValue('x', node, tensorMap, context))];
          }
          case 'Tanh': {
              return [tfc.tanh(getParamValue('x', node, tensorMap, context))];
          }
          case 'Tan':
              return [tfc.tan(getParamValue('x', node, tensorMap, context))];
          case 'Relu6':
          case 'ClipByValue':
              return [tfc.clipByValue(getParamValue('x', node, tensorMap, context), getParamValue('clipValueMin', node, tensorMap, context), getParamValue('clipValueMax', node, tensorMap, context))];
          case 'Rsqrt':
              return [tfc.rsqrt(getTensor(node.inputNames[0], tensorMap, context))];
          case 'Prod':
              return [tfc.prod(getParamValue('x', node, tensorMap, context), getParamValue('axes', node, tensorMap, context))];
          case 'LeakyRelu':
              return [tfc.leakyRelu(getParamValue('x', node, tensorMap, context), getParamValue('alpha', node, tensorMap, context))];
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  /**
   * The TensorArray object keeps an array of Tensors.  It
   * allows reading from the array and writing to the array.
   */
  class TensorArray {
      constructor(name, dtype, maxSize, elementShape, identicalElementShapes, dynamicSize, clearAfterRead) {
          this.name = name;
          this.dtype = dtype;
          this.maxSize = maxSize;
          this.elementShape = elementShape;
          this.identicalElementShapes = identicalElementShapes;
          this.dynamicSize = dynamicSize;
          this.clearAfterRead = clearAfterRead;
          this.tensors = [];
          this.closed_ = false;
          this.id = TensorArray.nextId++;
      }
      get closed() {
          return this.closed_;
      }
      /**
       * Close the current TensorArray.
       */
      clearAndClose() {
          this.tensors.forEach(tensor => tensor.tensor.dispose());
          this.tensors = [];
          this.closed_ = true;
      }
      size() {
          return this.tensors.length;
      }
      /**
       * Read the value at location index in the TensorArray.
       * @param index Number the index to read from.
       */
      read(index) {
          if (this.closed_) {
              throw new Error(`TensorArray ${this.name} has already been closed.`);
          }
          if (index < 0 || index >= this.tensors.length) {
              throw new Error(`Tried to read from index ${index}, but array size is: ${this.tensors.length}`);
          }
          const tensorWithState = this.tensors[index];
          if (tensorWithState.cleared) {
              throw new Error(`TensorArray ${this.name}: Could not read index ${index} twice because it was cleared after a previous read ` +
                  `(perhaps try setting clear_after_read = false?).`);
          }
          if (this.clearAfterRead) {
              tensorWithState.cleared = true;
          }
          tensorWithState.read = true;
          return tensorWithState.tensor;
      }
      /**
       * Helper method to read multiple tensors from the specified indices.
       */
      readMany(indices) {
          return indices.map(index => this.read(index));
      }
      /**
       * Write value into the index of the TensorArray.
       * @param index number the index to write to.
       * @param tensor
       */
      write(index, tensor) {
          if (this.closed_) {
              throw new Error(`TensorArray ${this.name} has already been closed.`);
          }
          if (index < 0 || !this.dynamicSize && index >= this.maxSize) {
              throw new Error(`Tried to write to index ${index}, but array is not resizeable and size is: ${this.maxSize}`);
          }
          const t = this.tensors[index] || {};
          if (tensor.dtype !== this.dtype) {
              throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${index},
          because the value dtype is ${tensor.dtype}, but TensorArray dtype is ${this.dtype}.`);
          }
          // Set the shape for the first time write to unknow shape tensor array
          if (this.size() === 0 &&
              (this.elementShape == null || this.elementShape.length === 0)) {
              this.elementShape = tensor.shape;
          }
          this.assertShapesMatchAllowUndefinedSize(this.elementShape, tensor.shape, `TensorArray ${this.name}: Could not write to TensorArray index ${index}.`);
          if (t && t.read) {
              throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${index}, because it has already been read.`);
          }
          if (t && t.written) {
              throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${index}, because it has already been written.`);
          }
          t.tensor = tensor;
          t.written = true;
          this.tensors[index] = t;
      }
      /**
       * Helper method to write multiple tensors to the specified indices.
       */
      writeMany(indices, tensors) {
          if (indices.length !== tensors.length) {
              throw new Error(`TensorArray ${this.name}: could not write multiple tensors,` +
                  `because the index size: ${indices.length} is not the same as tensors size: ${tensors.length}.`);
          }
          indices.forEach((i, index) => this.write(i, tensors[index]));
      }
      /**
       * Return selected values in the TensorArray as a packed Tensor. All of
       * selected values must have been written and their shapes must all match.
       * @param [indices] number[] Optional. Taking values in [0, max_value). If the
       *    TensorArray is not dynamic, max_value=size(). If not specified returns
       *    all tensors in the original order.
       * @param [dtype]
       */
      gather(indices, dtype) {
          if (!!dtype && dtype !== this.dtype) {
              throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${dtype}`);
          }
          if (!indices) {
              indices = [];
              for (let i = 0; i < this.size(); i++) {
                  indices.push(i);
              }
          }
          if (indices.length === 0) {
              return tfc.tensor([], [0].concat(this.elementShape));
          }
          // Read all the PersistentTensors into a vector to keep track of
          // their memory.
          const tensors = this.readMany(indices);
          this.assertShapesMatchAllowUndefinedSize(this.elementShape, tensors[0].shape, 'TensorArray shape mismatch: ');
          return tfc.stack(tensors, 0);
      }
      /**
       * Return the values in the TensorArray as a concatenated Tensor.
       */
      concat(dtype) {
          if (!!dtype && dtype !== this.dtype) {
              throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${dtype}`);
          }
          if (this.size() === 0) {
              return tfc.tensor([], [0].concat(this.elementShape));
          }
          const indices = [];
          for (let i = 0; i < this.size(); i++) {
              indices.push(i);
          }
          // Collect all the tensors from the tensors array.
          const tensors = this.readMany(indices);
          this.assertShapesMatchAllowUndefinedSize(this.elementShape, tensors[0].shape, `TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${tensors[0].shape})`);
          return tfc.concat(tensors, 0);
      }
      /**
       * Scatter the values of a Tensor in specific indices of a TensorArray.
       * @param indices nummber[] values in [0, max_value). If the
       *    TensorArray is not dynamic, max_value=size().
       * @param tensor Tensor input tensor.
       */
      scatter(indices, tensor) {
          if (tensor.dtype !== this.dtype) {
              throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${tensor.dtype}`);
          }
          if (indices.length !== tensor.shape[0]) {
              throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${indices.length} vs. ${tensor.shape[0]}`);
          }
          const maxIndex = Math.max(...indices);
          if (!this.dynamicSize && maxIndex >= this.maxSize) {
              throw new Error(`Max index must be < array size (${maxIndex}  vs. ${this.maxSize})`);
          }
          this.writeMany(indices, tfc.unstack(tensor, 0));
      }
      /**
       * Split the values of a Tensor into the TensorArray.
       * @param length number[] with the lengths to use when splitting value along
       *    its first dimension.
       * @param tensor Tensor, the tensor to split.
       */
      split(length, tensor) {
          if (tensor.dtype !== this.dtype) {
              throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${tensor.dtype}`);
          }
          let totalLength = 0;
          const cumulativeLengths = length.map(len => {
              totalLength += len;
              return totalLength;
          });
          if (totalLength !== tensor.shape[0]) {
              throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
          }
          if (!this.dynamicSize && length.length !== this.maxSize) {
              throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${length.length}), ` +
                  'and the TensorArray is not marked as dynamically resizeable');
          }
          const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
          const tensors = [];
          tfc.tidy(() => {
              tensor = tensor.reshape([1, totalLength, elementPerRow]);
              for (let i = 0; i < length.length; ++i) {
                  const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
                  const indices = [0, previousLength, 0];
                  const sizes = [1, length[i], elementPerRow];
                  tensors[i] = tfc.slice(tensor, indices, sizes).reshape(this.elementShape);
              }
              return tensors;
          });
          const indices = [];
          for (let i = 0; i < length.length; i++) {
              indices[i] = i;
          }
          this.writeMany(indices, tensors);
      }
      /**
       * This differs from util.assertShapesMatch in that it allows values of
       * negative one, an undefined size of a dimensinon, in a shape to match
       * anything.
       */
      assertShapesMatchAllowUndefinedSize(shapeA, shapeB, errorMessagePrefix = '') {
          tfc.util.assert(this.shapesEqualAllowUndefinedSize(shapeA, shapeB), () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
      }
      shapesEqualAllowUndefinedSize(n1, n2) {
          if (n1.length !== n2.length) {
              return false;
          }
          for (let i = 0; i < n1.length; i++) {
              if (n1[i] !== -1 && n2[i] !== -1 && n1[i] !== n2[i]) {
                  return false;
              }
          }
          return true;
      }
  }
  TensorArray.nextId = 0;

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
  async function executeOp$2(node, tensorMap, context) {
      switch (node.op) {
          case 'LoopCond':
              return [
                  getParamValue('pred', node, tensorMap, context).clone()
              ];
          case 'Switch': {
              const pred = getParamValue('pred', node, tensorMap, context);
              const data = getParamValue('data', node, tensorMap, context);
              // Outputs nodes :0 => false, :1 => true
              return (await pred.data())[0] ? [undefined, data.clone()] :
                  [data.clone(), undefined];
          }
          case 'Merge':
              const inputName = node.inputNames.find(name => getTensor(name, tensorMap, context) !== undefined);
              return inputName ? [getTensor(inputName, tensorMap, context).clone()] :
                  undefined;
          case 'Enter':
              const frameId = getParamValue('frameName', node, tensorMap, context);
              const data = getParamValue('tensor', node, tensorMap, context);
              context.enterFrame(frameId);
              return [data.clone()];
          case 'Exit':
              const tensor = getParamValue('tensor', node, tensorMap, context);
              context.exitFrame();
              return [tensor.clone()];
          case 'NextIteration':
              const input = getParamValue('tensor', node, tensorMap, context);
              context.nextIteration();
              return [input.clone()];
          case 'TensorArrayV3':
              const size = getParamValue('size', node, tensorMap, context);
              const dtype = getParamValue('dtype', node, tensorMap, context);
              const elementShape = getParamValue('elementShape', node, tensorMap, context);
              const dynamicSize = getParamValue('dynamicSize', node, tensorMap, context);
              const clearAfterRead = getParamValue('clearAfterRead', node, tensorMap, context);
              const identicalElementShapes = getParamValue('identicalElementShapes', node, tensorMap, context);
              const name = getParamValue('name', node, tensorMap, context);
              const tensorArray = new TensorArray(name, dtype, size, elementShape, identicalElementShapes, dynamicSize, clearAfterRead);
              context.addTensorArray(tensorArray);
              return [tfc.scalar(tensorArray.id), tfc.scalar(1.0)];
          case 'TensorArrayWriteV3':
              const id = getParamValue('tensorArrayId', node, tensorMap, context);
              const index = getParamValue('index', node, tensorMap, context);
              const writeTensor = getParamValue('tensor', node, tensorMap, context);
              const writeTensorArray = context.getTensorArray(id);
              writeTensorArray.write(index, writeTensor);
              return [tfc.scalar(1.0)];
          case 'TensorArrayReadV3':
              const readId = getParamValue('tensorArrayId', node, tensorMap, context);
              const readIndex = getParamValue('index', node, tensorMap, context);
              const readTensorArray = context.getTensorArray(readId);
              return [readTensorArray.read(readIndex)];
          case 'TensorArrayGatherV3':
              const gatherId = getParamValue('tensorArrayId', node, tensorMap, context);
              const gatherIndices = getParamValue('indices', node, tensorMap, context);
              const gatherDtype = getParamValue('dtype', node, tensorMap, context);
              const gatherTensorArray = context.getTensorArray(gatherId);
              return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
          case 'TensorArrayScatterV3':
              const scatterId = getParamValue('tensorArrayId', node, tensorMap, context);
              const scatterIndices = getParamValue('indices', node, tensorMap, context);
              const scatterTensor = getParamValue('tensor', node, tensorMap, context);
              const scatterTensorArray = context.getTensorArray(scatterId);
              scatterTensorArray.scatter(scatterIndices, scatterTensor);
              return [tfc.scalar(1.0)];
          case 'TensorArrayConcatV3':
              const concatId = getParamValue('tensorArrayId', node, tensorMap, context);
              const concatTensorArray = context.getTensorArray(concatId);
              const concatDtype = getParamValue('dtype', node, tensorMap, context);
              return [concatTensorArray.concat(concatDtype)];
          case 'TensorArraySplitV3':
              const splitId = getParamValue('tensorArrayId', node, tensorMap, context);
              const splitTensor = getParamValue('tensor', node, tensorMap, context);
              const lengths = getParamValue('lengths', node, tensorMap, context);
              const splitTensorArray = context.getTensorArray(splitId);
              splitTensorArray.split(lengths, splitTensor);
              return [tfc.scalar(1.0)];
          case 'TensorArraySizeV3':
              const sizeId = getParamValue('tensorArrayId', node, tensorMap, context);
              const sizeTensorArray = context.getTensorArray(sizeId);
              return [tfc.scalar(sizeTensorArray.size(), 'int32')];
          case 'TensorArrayCloseV3':
              const closeId = getParamValue('tensorArrayId', node, tensorMap, context);
              const closeTensorArray = context.getTensorArray(closeId);
              closeTensorArray.clearAndClose();
              return [];
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  }

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
  let executeOp$3 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Conv1D': {
              const stride = getParamValue('stride', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              const dataFormat = getParamValue('dataFormat', node, tensorMap, context)
                  .toUpperCase();
              const dilation = getParamValue('dilation', node, tensorMap, context);
              return [tfc.conv1d(getParamValue('x', node, tensorMap, context), getParamValue('filter', node, tensorMap, context), stride, pad, dataFormat, dilation)];
          }
          case 'Conv2D': {
              const stride = getParamValue('strides', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              const dataFormat = getParamValue('dataFormat', node, tensorMap, context)
                  .toUpperCase();
              const dilations = getParamValue('dilations', node, tensorMap, context);
              return [tfc.conv2d(getParamValue('x', node, tensorMap, context), getParamValue('filter', node, tensorMap, context), [stride[1], stride[2]], pad, dataFormat, [dilations[1], dilations[2]])];
          }
          case 'Conv2DBackpropInput':
          case 'Conv2dTranspose': {
              const shape = getParamValue('outputShape', node, tensorMap, context);
              const stride = getParamValue('strides', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              return [tfc.conv2dTranspose(getParamValue('x', node, tensorMap, context), getParamValue('filter', node, tensorMap, context), shape, [stride[1], stride[2]], pad)];
          }
          case 'DepthwiseConv2dNative':
          case 'DepthwiseConv2d': {
              const stride = getParamValue('strides', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              const dilations = getParamValue('dilations', node, tensorMap, context);
              const dataFormat = getParamValue('dataFormat', node, tensorMap, context)
                  .toUpperCase();
              return [tfc.depthwiseConv2d(getParamValue('input', node, tensorMap, context), getParamValue('filter', node, tensorMap, context), [stride[1], stride[2]], pad, dataFormat, [dilations[1], dilations[2]])];
          }
          case 'AvgPool': {
              const stride = getParamValue('strides', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              const kernelSize = getParamValue('kernelSize', node, tensorMap, context);
              return [tfc.avgPool(getParamValue('x', node, tensorMap, context), [kernelSize[1], kernelSize[2]], [stride[1], stride[2]], pad)];
          }
          case 'MaxPool': {
              const stride = getParamValue('strides', node, tensorMap, context);
              const pad = getParamValue('pad', node, tensorMap, context);
              const kernelSize = getParamValue('kernelSize', node, tensorMap, context);
              return [tfc.maxPool(getParamValue('x', node, tensorMap, context), [kernelSize[1], kernelSize[2]], [stride[1], stride[2]], pad)];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$4 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Fill': {
              const shape = getParamValue('shape', node, tensorMap, context);
              const dtype = getParamValue('dtype', node, tensorMap, context);
              const value = getParamValue('value', node, tensorMap, context);
              return [tfc.fill(shape, value, dtype)];
          }
          case 'LinSpace': {
              const start = getParamValue('start', node, tensorMap, context);
              const stop = getParamValue('stop', node, tensorMap, context);
              const num = getParamValue('num', node, tensorMap, context);
              return [tfc.linspace(start, stop, num)];
          }
          case 'OneHot': {
              const indices = getParamValue('indices', node, tensorMap, context);
              const depth = getParamValue('depth', node, tensorMap, context);
              const onValue = getParamValue('onValue', node, tensorMap, context);
              const offValue = getParamValue('offValue', node, tensorMap, context);
              return [tfc.oneHot(indices, depth, onValue, offValue)];
          }
          case 'Ones': {
              return [tfc.ones(getParamValue('shape', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
          }
          case 'OnesLike': {
              return [tfc.onesLike(getParamValue('x', node, tensorMap, context))];
          }
          case 'RandomUniform': {
              return [tfc.randomUniform(
                  // tslint:disable-next-line:no-any
                  getParamValue('shape', node, tensorMap, context), getParamValue('minval', node, tensorMap, context), getParamValue('maxval', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
          }
          case 'Range': {
              const start = getParamValue('start', node, tensorMap, context);
              const stop = getParamValue('stop', node, tensorMap, context);
              const step = getParamValue('step', node, tensorMap, context);
              return [tfc.range(start, stop, step, getParamValue('dtype', node, tensorMap, context))];
          }
          case 'TruncatedNormal': {
              const shape = getParamValue('shape', node, tensorMap, context);
              const mean = getParamValue('mean', node, tensorMap, context);
              const stdDev = getParamValue('stdDev', node, tensorMap, context);
              const seed = getParamValue('seed', node, tensorMap, context);
              return [tfc.truncatedNormal(shape, mean, stdDev, getParamValue('dtype', node, tensorMap, context), seed)];
          }
          case 'Zeros': {
              return [tfc.zeros(getParamValue('shape', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
          }
          case 'ZerosLike': {
              return [tfc.zerosLike(getParamValue('x', node, tensorMap, context))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  async function executeOp$5(node, tensorMap, context) {
      switch (node.op) {
          case 'NonMaxSuppressionV3':
          case 'NonMaxSuppressionV2': {
              const boxes = getParamValue('boxes', node, tensorMap, context);
              const scores = getParamValue('scores', node, tensorMap, context);
              const maxOutputSize = getParamValue('maxOutputSize', node, tensorMap, context);
              const iouThreshold = getParamValue('iouThreshold', node, tensorMap, context);
              const scoreThreshold = getParamValue('scoreThreshold', node, tensorMap, context);
              return [await tfc.image.nonMaxSuppressionAsync(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold)];
          }
          case 'Where': {
              return [await tfc.whereAsync(getParamValue('condition', node, tensorMap, context))];
          }
          case 'ListDiff': {
              return await tfc.setdiff1dAsync(getParamValue('x', node, tensorMap, context), getParamValue('y', node, tensorMap, context));
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  }

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  let executeOp$6 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'TopKV2': {
              const x = getParamValue('x', node, tensorMap, context);
              const k = getParamValue('k', node, tensorMap, context);
              const sorted = getParamValue('sorted', node, tensorMap, context);
              const result = tfc.topk(x, k, sorted);
              return [result.values, result.indices];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$7 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Const': {
              return tensorMap[node.name];
          }
          case 'PlaceholderWithDefault':
              const def = getParamValue('default', node, tensorMap, context);
              return [getTensor(node.name, tensorMap, context) || def];
          case 'Placeholder':
              return [getTensor(node.name, tensorMap, context)];
          case 'Identity':
          case 'StopGradient':
          case 'FakeQuantWithMinMaxVars': // This op is currently ignored.
              return [
                  getParamValue('x', node, tensorMap, context).clone()
              ];
          case 'IdentityN':
              return getParamValue('x', node, tensorMap, context)
                  .map((t) => t.clone());
          case 'Snapshot':
              const snapshot = getParamValue('x', node, tensorMap, context);
              return [snapshot.clone()];
          case 'Shape':
              return [tfc.tensor1d(getParamValue('x', node, tensorMap, context).shape, 'int32')];
          case 'ShapeN':
              return getParamValue('x', node, tensorMap, context)
                  .map((t) => tfc.tensor1d(t.shape));
          case 'Size':
              return [tfc.scalar(getParamValue('x', node, tensorMap, context).size, 'int32')];
          case 'Rank':
              return [tfc.scalar(getParamValue('x', node, tensorMap, context).rank, 'int32')];
          case 'NoOp':
              return [];
          case 'Print':
              const input = getParamValue('x', node, tensorMap, context);
              const data = getParamValue('data', node, tensorMap, context);
              const message = getParamValue('message', node, tensorMap, context);
              const summarize = getParamValue('summarize', node, tensorMap, context);
              console.warn('The graph has a tf.print() operation,' +
                  'usually used for debugging, which slows down performance.');
              console.log(message);
              for (let i = 0; i < data.length; i++) {
                  console.log(Array.prototype.slice.call(data[i].dataSync()).slice(0, summarize));
              }
              return [input];
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

  /**
   * @license
   * Copyright 2018 Google Inc. All Rights Reserved.
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
  let executeOp$8 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'ResizeBilinear': {
              const images = getParamValue('images', node, tensorMap, context);
              const size = getParamValue('size', node, tensorMap, context);
              const alignCorners = getParamValue('alignCorners', node, tensorMap, context);
              return [tfc.image.resizeBilinear(images, [size[0], size[1]], alignCorners)];
          }
          case 'ResizeNearestNeighbor': {
              const images = getParamValue('images', node, tensorMap, context);
              const size = getParamValue('size', node, tensorMap, context);
              const alignCorners = getParamValue('alignCorners', node, tensorMap, context);
              return [tfc.image.resizeNearestNeighbor(images, [size[0], size[1]], alignCorners)];
          }
          case 'CropAndResize': {
              const image = getParamValue('image', node, tensorMap, context);
              const boxes = getParamValue('boxes', node, tensorMap, context);
              const boxInd = getParamValue('boxInd', node, tensorMap, context);
              const cropSize = getParamValue('cropSize', node, tensorMap, context);
              const method = getParamValue('method', node, tensorMap, context);
              const extrapolationValue = getParamValue('extrapolationValue', node, tensorMap, context);
              return [tfc.image.cropAndResize(image, boxes, boxInd, cropSize, method, extrapolationValue)];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$9 = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Equal': {
              return [tfc.equal(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'NotEqual': {
              return [tfc.notEqual(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Greater': {
              return [tfc.greater(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'GreaterEqual': {
              return [tfc.greaterEqual(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Less': {
              return [tfc.less(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'LessEqual': {
              return [tfc.lessEqual(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'LogicalAnd': {
              return [tfc.logicalAnd(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'LogicalNot': {
              return [tfc.logicalNot(getParamValue('a', node, tensorMap, context))];
          }
          case 'LogicalOr': {
              return [tfc.logicalOr(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          case 'Select': {
              return [tfc.where(getParamValue('condition', node, tensorMap, context), getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$a = (node, tensorMap, context) => {
      switch (node.op) {
          case 'BatchMatMul':
          case 'BatchMatMulV2':
          case 'MatMul':
              return [tfc.matMul(getParamValue('a', node, tensorMap, context), getParamValue('b', node, tensorMap, context), getParamValue('transposeA', node, tensorMap, context), getParamValue('transposeB', node, tensorMap, context))];
          case 'Transpose':
              return [tfc.transpose(getParamValue('x', node, tensorMap, context), getParamValue('perm', node, tensorMap, context))];
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$b = (node, tensorMap, context) => {
      switch (node.op) {
          case 'FusedBatchNorm':
          case 'FusedBatchNormV2': {
              return [tfc.batchNorm(getParamValue('x', node, tensorMap, context), getParamValue('mean', node, tensorMap, context), getParamValue('variance', node, tensorMap, context), getParamValue('offset', node, tensorMap, context), getParamValue('scale', node, tensorMap, context), getParamValue('epsilon', node, tensorMap, context))];
          }
          case 'FusedBatchNormV3': {
              return [tfc.batchNorm(getParamValue('x', node, tensorMap, context), getParamValue('mean', node, tensorMap, context), getParamValue('variance', node, tensorMap, context), getParamValue('offset', node, tensorMap, context), getParamValue('scale', node, tensorMap, context), getParamValue('epsilon', node, tensorMap, context))];
          }
          case 'LRN': {
              return [tfc.localResponseNormalization(getParamValue('x', node, tensorMap, context), getParamValue('radius', node, tensorMap, context), getParamValue('bias', node, tensorMap, context), getParamValue('alpha', node, tensorMap, context), getParamValue('beta', node, tensorMap, context))];
          }
          case 'Softmax': {
              return [tfc.softmax(getParamValue('x', node, tensorMap, context))];
          }
          case 'LogSoftmax': {
              return [tfc.logSoftmax(getParamValue('x', node, tensorMap, context))];
          }
          case 'SparseToDense': {
              return [tfc.sparseToDense(getParamValue('sparseIndices', node, tensorMap, context), getParamValue('outputShape', node, tensorMap, context), getParamValue('sparseValues', node, tensorMap, context), getParamValue('defaultValue', node, tensorMap, context))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$c = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Max': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.max(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'Mean': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.mean(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'Min': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.min(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'Sum': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.sum(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'All': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.all(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'Any': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.any(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          case 'ArgMax': {
              const axis = getParamValue('axis', node, tensorMap, context);
              return [tfc.argMax(getParamValue('x', node, tensorMap, context), axis)];
          }
          case 'ArgMin': {
              const axis = getParamValue('axis', node, tensorMap, context);
              return [tfc.argMin(getParamValue('x', node, tensorMap, context), axis)];
          }
          case 'Prod': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const keepDims = getParamValue('keepDims', node, tensorMap, context);
              return [tfc.prod(getParamValue('x', node, tensorMap, context), axis, keepDims)];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$d = (node, tensorMap, context) => {
      switch (node.op) {
          case 'ConcatV2':
          case 'Concat': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const inputs = getParamValue('tensors', node, tensorMap, context);
              return [tfc.concat(inputs, axis)];
          }
          case 'GatherV2':
          case 'Gather': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const input = getParamValue('x', node, tensorMap, context);
              const indices = getParamValue('indices', node, tensorMap, context);
              return [tfc.gather(input, indices.asType('int32'), axis)];
          }
          case 'ReverseV2':
          case 'Reverse': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const input = getParamValue('x', node, tensorMap, context);
              return [tfc.reverse(input, axis)];
          }
          case 'Slice': {
              // tslint:disable-next-line:no-any
              const begin = getParamValue('begin', node, tensorMap, context);
              // tslint:disable-next-line:no-any
              const size = getParamValue('size', node, tensorMap, context);
              return [tfc.slice(getParamValue('x', node, tensorMap, context), begin, size)];
          }
          case 'StridedSlice': {
              const begin = getParamValue('begin', node, tensorMap, context);
              const end = getParamValue('end', node, tensorMap, context);
              const strides = getParamValue('strides', node, tensorMap, context);
              const beginMask = getParamValue('beginMask', node, tensorMap, context);
              const endMask = getParamValue('endMask', node, tensorMap, context);
              const ellipsisMask = getParamValue('ellipsisMask', node, tensorMap, context);
              const newAxisMask = getParamValue('newAxisMask', node, tensorMap, context);
              const shrinkAxisMask = getParamValue('shrinkAxisMask', node, tensorMap, context);
              const tensor = getParamValue('x', node, tensorMap, context);
              if (begin.length === 1 && tensor.shape.length > 1) {
                  for (let i = 1; i < tensor.shape.length; i++) {
                      begin.push(0);
                      end.push(tensor.shape[i]);
                      strides.push(strides[0]);
                  }
              }
              return [tfc.stridedSlice(tensor, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask)];
          }
          case 'Pack': {
              return tfc.tidy(() => {
                  const axis = getParamValue('axis', node, tensorMap, context);
                  const tensors = getParamValue('tensors', node, tensorMap, context);
                  // Reshape the tensors to the first tensor's shape if they don't match.
                  const shape = tensors[0].shape;
                  const squeezedShape = tensors[0].squeeze().shape;
                  const mapped = tensors.map(tensor => {
                      const sameShape = tfc.util.arraysEqual(tensor.shape, shape);
                      if (!sameShape &&
                          !tfc.util.arraysEqual(tensor.squeeze().shape, squeezedShape)) {
                          throw new Error('the input tensors shape does not match');
                      }
                      return sameShape ? tensor : tensor.reshape(shape);
                  });
                  return [tfc.stack(mapped, axis)];
              });
          }
          case 'Unpack': {
              return tfc.tidy(() => {
                  const axis = getParamValue('axis', node, tensorMap, context);
                  const tensor = getParamValue('tensor', node, tensorMap, context);
                  return tfc.unstack(tensor, axis);
              });
          }
          case 'Tile': {
              const reps = getParamValue('reps', node, tensorMap, context);
              return [tfc.tile(getParamValue('x', node, tensorMap, context), reps)];
          }
          case 'Split':
          case 'SplitV': {
              const axis = getParamValue('axis', node, tensorMap, context);
              const numOrSizeSplits = getParamValue('numOrSizeSplits', node, tensorMap, context);
              return tfc.split(getParamValue('x', node, tensorMap, context), numOrSizeSplits, axis);
          }
          case 'ScatterNd': {
              const indices = getParamValue('indices', node, tensorMap, context);
              const values = getParamValue('values', node, tensorMap, context);
              const shape = getParamValue('shape', node, tensorMap, context);
              return [tfc.scatterND(indices, values, shape)];
          }
          case 'GatherNd': {
              const x = getParamValue('x', node, tensorMap, context);
              const indices = getParamValue('indices', node, tensorMap, context);
              return [tfc.gatherND(x, indices)];
          }
          case 'SparseToDense': {
              const indices = getParamValue('sparseIndices', node, tensorMap, context);
              const shape = getParamValue('outputShape', node, tensorMap, context);
              const sparseValues = getParamValue('sparseValues', node, tensorMap, context);
              const defaultValue = getParamValue('defaultValue', node, tensorMap, context);
              return [tfc.sparseToDense(indices, sparseValues, shape, sparseValues.dtype === defaultValue.dtype ?
                      defaultValue :
                      defaultValue.asType(sparseValues.dtype))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$e = (node, tensorMap, context) => {
      switch (node.op) {
          case 'FFT': {
              return [tfc.fft(getParamValue('x', node, tensorMap, context))];
          }
          case 'IFFT': {
              return [tfc.ifft(getParamValue('x', node, tensorMap, context))];
          }
          case 'RFFT': {
              return [tfc.rfft(getParamValue('x', node, tensorMap, context))];
          }
          case 'IRFFT': {
              return [tfc.irfft(getParamValue('x', node, tensorMap, context))];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  let executeOp$f = (node, tensorMap, context) => {
      switch (node.op) {
          case 'Cast': {
              return [tfc.cast(getParamValue('x', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
          }
          case 'ExpandDims': {
              const axis = getParamValue('axis', node, tensorMap, context);
              return [tfc.expandDims(getParamValue('x', node, tensorMap, context), axis)];
          }
          case 'Squeeze': {
              const axis = getParamValue('axis', node, tensorMap, context);
              return [tfc.squeeze(getParamValue('x', node, tensorMap, context), axis)];
          }
          case 'Reshape': {
              return [tfc.reshape(getParamValue('x', node, tensorMap, context), getParamValue('shape', node, tensorMap, context))];
          }
          case 'PadV2':
          case 'Pad': {
              return [tfc.pad(getParamValue('x', node, tensorMap, context), split(getParamValue('padding', node, tensorMap, context), 2), getParamValue('constantValue', node, tensorMap, context))];
          }
          case 'SpaceToBatchND': {
              const blockShape = getParamValue('blockShape', node, tensorMap, context);
              const paddings = split(getParamValue('paddings', node, tensorMap, context), 2);
              return [tfc.spaceToBatchND(getParamValue('x', node, tensorMap, context), blockShape, paddings)];
          }
          case 'BatchToSpaceND': {
              const blockShape = getParamValue('blockShape', node, tensorMap, context);
              const crops = split(getParamValue('crops', node, tensorMap, context), 2);
              return [tfc.batchToSpaceND(getParamValue('x', node, tensorMap, context), blockShape, crops)];
          }
          case 'DepthToSpace': {
              const blockSize = getParamValue('blockSize', node, tensorMap, context);
              const dataFormat = getParamValue('dataFormat', node, tensorMap, context).toUpperCase();
              return [tfc.depthToSpace(getParamValue('x', node, tensorMap, context), blockSize, dataFormat)];
          }
          default:
              throw TypeError(`Node type ${node.op} is not implemented`);
      }
  };

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
  /**
   * Executes the op defined by the node object.
   * @param node
   * @param tensorMap contains tensors for executed nodes and weights
   */
  function executeOp$g(node, tensorMap, context) {
      const value = ((node, tensorMap, context) => {
          switch (node.category) {
              case 'arithmetic':
                  return executeOp(node, tensorMap, context);
              case 'basic_math':
                  return executeOp$1(node, tensorMap, context);
              case 'control':
                  return executeOp$2(node, tensorMap, context);
              case 'convolution':
                  return executeOp$3(node, tensorMap, context);
              case 'creation':
                  return executeOp$4(node, tensorMap, context);
              case 'dynamic':
                  return executeOp$5(node, tensorMap, context);
              case 'evaluation':
                  return executeOp$6(node, tensorMap, context);
              case 'image':
                  return executeOp$8(node, tensorMap, context);
              case 'graph':
                  return executeOp$7(node, tensorMap, context);
              case 'logical':
                  return executeOp$9(node, tensorMap, context);
              case 'matrices':
                  return executeOp$a(node, tensorMap, context);
              case 'normalization':
                  return executeOp$b(node, tensorMap, context);
              case 'reduction':
                  return executeOp$c(node, tensorMap, context);
              case 'slice_join':
                  return executeOp$d(node, tensorMap, context);
              case 'spectral':
                  return executeOp$e(node, tensorMap, context);
              case 'transformation':
                  return executeOp$f(node, tensorMap, context);
              case 'custom':
                  const opMapper = getRegisteredOp(node.op);
                  if (opMapper && opMapper.customExecutor) {
                      return opMapper.customExecutor(new NodeValueImpl(node, tensorMap, context));
                  }
                  else {
                      throw TypeError(`Custom op ${node.op} is not registered.`);
                  }
              default:
                  throw TypeError(`Unknown op '${node.op}'. File an issue at ` +
                      `https://github.com/tensorflow/tfjs/issues so we can add it` +
                      `, or register a custom execution with tf.registerOp()`);
          }
      })(node, tensorMap, context);
      if (value instanceof Promise) {
          return value.then((data) => [].concat(data));
      }
      return [].concat(value);
  }

  /**
   * ExecutionContext captures the runtime environment of the node. It keeps
   * track of the current frame and iteration for the control flow ops.
   *
   * For example, typical Dynamic RNN model may contain loops, for which
   * TensorFlow will generate graphs with Enter/Exit nodes to control the
   * current execution frame, and NextIteration Nodes for iteration id increment.
   * For model with branch logic, TensorFLow will generate Switch/Merge ops.
   */
  class ExecutionContext {
      constructor(weightMap, tensorArrayMap) {
          this.weightMap = weightMap;
          this.tensorArrayMap = tensorArrayMap;
          this.rootContext = { id: 0, frameName: '', iterationId: 0 };
          this.contexts = [this.rootContext];
          this.lastId = 0;
          this.generateCurrentContextIds();
      }
      newFrame(id, frameName) {
          return { id, frameName, iterationId: 0 };
      }
      /**
       * Set the current context
       * @param contexts: ExecutionContextInfo[] the current path of execution
       * frames
       */
      set currentContext(contexts) {
          if (this.contexts !== contexts) {
              this.contexts = contexts;
              this.generateCurrentContextIds();
          }
      }
      get currentContext() {
          return this.contexts;
      }
      /**
       * Returns the current context in string format.
       */
      get currentContextId() {
          return this._currentContextIds[0];
      }
      /**
       * Returns the current context and all parent contexts in string format.
       * This allow access to the nodes in the current and parent frames.
       */
      get currentContextIds() {
          return this._currentContextIds;
      }
      generateCurrentContextIds() {
          const names = [];
          for (let i = 0; i < this.contexts.length - 1; i++) {
              const contexts = this.contexts.slice(0, this.contexts.length - i);
              names.push(this.contextIdforContexts(contexts));
          }
          names.push('');
          this._currentContextIds = names;
      }
      contextIdforContexts(contexts) {
          return contexts ?
              contexts
                  .map(context => (context.id === 0 && context.iterationId === 0) ?
                  '' :
                  `${context.frameName}-${context.iterationId}`)
                  .join('/') :
              '';
      }
      /**
       * Enter a new frame, a new context is pushed on the current context list.
       * @param frameId new frame id
       */
      enterFrame(frameId) {
          if (this.contexts) {
              this.lastId++;
              this.contexts = this.contexts.slice();
              this.contexts.push(this.newFrame(this.lastId, frameId));
              this._currentContextIds.unshift(this.contextIdforContexts(this.contexts));
          }
      }
      /**
       * Exit the current frame, the last context is removed from the current
       * context list.
       */
      exitFrame() {
          if (this.contexts && this.contexts.length > 1) {
              this.contexts = this.contexts.slice();
              this.contexts.splice(-1);
              this.currentContextIds.shift();
          }
          else {
              throw new Error('Cannot exit frame, the context is empty');
          }
      }
      /**
       * Enter the next iteration of a loop, the iteration id of last context is
       * increased.
       */
      nextIteration() {
          if (this.contexts && this.contexts.length > 0) {
              this.contexts = this.contexts.slice();
              this.lastId++;
              const context = Object.assign({}, this.contexts[this.contexts.length - 1]);
              context.iterationId += 1;
              context.id = this.lastId;
              this.contexts.splice(-1, 1, context);
              this._currentContextIds.splice(0, 1, this.contextIdforContexts(this.contexts));
          }
          else {
              throw new Error('Cannot increase frame iteration, the context is empty');
          }
      }
      getWeight(name) {
          return this.weightMap[name];
      }
      addTensorArray(tensorArray) {
          this.tensorArrayMap[tensorArray.id] = tensorArray;
      }
      getTensorArray(id) {
          return this.tensorArrayMap[id];
      }
  }

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
  /**
   * Given graph inputs and desired outputs, find the minimal set of nodes
   * to execute in order to compute the outputs. In addition return other useful
   * info such:
   * - Missing inputs needed to compute the output.
   * - Whether the subgraph contains dynamic ops (control flow, dynamic shape).
   * - Alternative inputs in order to avoid async (dynamic op) execution.
   */
  function getExecutionSubgraph(inputs, outputs, weightMap) {
      const usedNodes = new Set();
      const missingInputs = [];
      let dynamicNode = null;
      let syncInputs = null;
      // Start with the outputs, going backwards and find all the nodes that are
      // needed to compute those outputs.
      const seen = new Set();
      const frontier = [...outputs];
      while (frontier.length > 0) {
          const node = frontier.pop();
          if (isControlFlow(node) || isDynamicShape(node)) {
              if (dynamicNode == null) {
                  dynamicNode = node;
                  syncInputs = dynamicNode.children.map(child => child.name)
                      .filter(name => usedNodes.has(name));
              }
          }
          usedNodes.add(node.name);
          // Weights are dead end since we already have their values.
          if (weightMap[node.name] != null) {
              continue;
          }
          // This node is a dead end since it's one of the user-provided inputs.
          if (inputs[node.name] != null) {
              continue;
          }
          if (node.inputs.length === 0) {
              missingInputs.push(node.name);
              continue;
          }
          node.inputs.forEach(input => {
              // Don't add to the frontier if it is already there.
              if (seen.has(input.name)) {
                  return;
              }
              seen.add(input.name);
              frontier.push(input);
          });
      }
      return { inputs, outputs, usedNodes, missingInputs, dynamicNode, syncInputs };
  }
  /**
   * Given the execution info, return a list of nodes in topological order that
   * need to be executed to compute the output.
   */
  function getNodesInTopologicalOrder(graph, weightMap, executionInfo) {
      const { usedNodes, inputs } = executionInfo;
      const frontier = [];
      const inputNodes = Object.keys(inputs).map(name => graph.nodes[name]);
      inputNodes.forEach(input => {
          if (usedNodes.has(input.name)) {
              frontier.push(input);
          }
      });
      graph.weights.forEach(weight => {
          if (usedNodes.has(weight.name)) {
              frontier.push(weight);
          }
      });
      const seen = new Set();
      const orderedNodes = [];
      while (frontier.length > 0) {
          const node = frontier.pop();
          seen.add(node.name);
          if (!weightMap[node.name]) {
              orderedNodes.push(node);
          }
          node.children.forEach(child => {
              if (!seen.has(child.name) && usedNodes.has(child.name) &&
                  child.inputs.every(input => seen.has(input.name))) {
                  frontier.push(child);
              }
          });
      }
      return orderedNodes;
  }
  const CONTROL_FLOW_OPS = ['Switch', 'Merge', 'Enter', 'Exit', 'NextIteration'];
  const DYNAMIC_SHAPE_OPS = ['NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'Where'];
  function isControlFlow(node) {
      return CONTROL_FLOW_OPS.indexOf(node.op) >= 0;
  }
  function isDynamicShape(node) {
      return DYNAMIC_SHAPE_OPS.indexOf(node.op) >= 0;
  }

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
  /**
   * This graph rewrite rule tries to identify the PRelu structure generated by
   * tf.keras, and convert it to tfjs core prelu op.
   *
   * The formula of PReLU is:
   * f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
   *
   * `x` is the input, and `alpha` is a trainable tensor which can be broadcasted
   * to the shape of `x`.
   *
   * There's no native PRelu op in TensorFlow, so tf.keras generates the following
   * structure which does the equivalent calculation:
   * f(x) = Relu(x) + (-alpha * Relu(-x))
   *
   * Practically, alpha is always a constant in the inference graph.
   * Therefore, we're looking for the structure:
   *
   * f(x) = Add(Relu(x), Mul(negative_alpha, Relu(Neg(x))))
   *
   * And generate the follow sub graph:
   * f(x) = Prelu(x, neg(negative_alpha))
   *
   * @param graph Graph, model graph object
   * @param weightMap NamedTensorsMap, the weight map for the executor.
   */
  function rewritePrelu(graph, weightMap) {
      for (const key in graph.nodes) {
          const addNode = graph.nodes[key];
          if (addNode == null || addNode.op !== 'Add' && addNode.op !== 'AddV2' ||
              addNode.inputNames.length !== 2) {
              continue;
          }
          const reluNode = addNode.inputs.find(input => input.op === 'Relu');
          if (reluNode == null || reluNode.inputNames.length !== 1) {
              continue;
          }
          const mulOp = addNode.inputs.find(input => input.op === 'Mul');
          if (mulOp == null || mulOp.inputNames.length !== 2) {
              continue;
          }
          const negAlphaTensorNode = mulOp.inputs.find(input => input.op === 'Const');
          const reluNegInputNode = mulOp.inputs.find(input => input.op === 'Relu');
          if (negAlphaTensorNode == null || reluNegInputNode == null ||
              reluNegInputNode.inputNames.length !== 1) {
              continue;
          }
          // This detects a Neg op followed by a separated Relu op.
          const negInputNode = reluNegInputNode.inputs[0];
          if (negInputNode == null || negInputNode.op !== 'Neg' ||
              negInputNode.inputNames.length !== 1) {
              continue;
          }
          if (reluNode.inputNames[0] !== negInputNode.inputNames[0]) {
              continue;
          }
          const inputNode = reluNode.inputs[0];
          const outputNodes = addNode.children;
          // Construct a tensor for positive alpha (double negative).
          const alphaTensorName = negAlphaTensorNode.name + '_neg';
          const negNode = {
              name: alphaTensorName,
              inputNames: [],
              inputs: [],
              attrParams: {},
              category: 'graph',
              children: [],
              op: 'Const',
              inputParams: {},
              rawAttrs: {}
          };
          // Add the constant to weightMap
          weightMap[alphaTensorName] = [tfc.neg(weightMap[negAlphaTensorNode.name][0])];
          graph.weights.push(negNode);
          // Construct the prelu node
          const preluNode = {
              name: addNode.name + '_Prelu',
              inputNames: [inputNode.name, negNode.name],
              inputs: [inputNode, negNode],
              attrParams: {},
              category: 'custom',
              children: outputNodes,
              op: 'Prelu',
              inputParams: {
                  'x': { inputIndexStart: 0, type: 'tensor' },
                  'alpha': { inputIndexStart: 1, type: 'tensor' }
              }
          };
          negNode.children.push(preluNode);
          // Clean up the children and inputs of input/output nodes of the subgraph.
          const mulIndex = negAlphaTensorNode.children.indexOf(mulOp);
          if (mulIndex > -1) {
              negAlphaTensorNode.children.splice(mulIndex, 1);
          }
          const reluIndex = inputNode.children.indexOf(reluNode);
          if (reluIndex > -1) {
              inputNode.children.splice(reluIndex, 1);
          }
          const negIndex = inputNode.children.indexOf(negInputNode);
          if (negIndex > -1) {
              inputNode.children.splice(negIndex, 1);
          }
          inputNode.children.push(preluNode);
          outputNodes.forEach(node => {
              const addIndex = node.inputNames.indexOf(addNode.name);
              if (addIndex > -1) {
                  node.inputNames[addIndex] = preluNode.name;
                  node.inputs[addIndex] = preluNode;
              }
          });
          // The prelu node should be an output node.
          if (outputNodes.length === 0) {
              const addIndex = graph.outputs.indexOf(addNode);
              if (addIndex > -1) {
                  graph.outputs.splice(addIndex, 1);
              }
              graph.outputs.push(preluNode);
          }
          // remove the nodes for keras generated prelu subgraph.
          delete graph.nodes[addNode.name];
          delete graph.nodes[mulOp.name];
          delete graph.nodes[reluNode.name];
          delete graph.nodes[reluNegInputNode.name];
          delete graph.nodes[negInputNode.name];
          // add the newly generated nodes.
          graph.nodes[preluNode.name] = preluNode;
          graph.nodes[negNode.name] = negNode;
      }
  }

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
  class GraphExecutor {
      constructor(graph) {
          this.graph = graph;
          this.compiledMap = new Map();
          this._weightMap = {};
          this.SEPERATOR = ',';
          this.placeholders = graph.placeholders;
          this._outputs = graph.outputs;
      }
      get weightMap() {
          return this._weightMap;
      }
      set weightMap(weightMap) {
          const weightIds = Object.keys(weightMap).map(key => weightMap[key].map(tensor => tensor.id));
          this.weightIds = [].concat.apply([], weightIds);
          this._weightMap = weightMap;
      }
      get inputs() {
          return this.placeholders.map(node => {
              return {
                  name: node.name,
                  shape: node.attrParams['shape'] ?
                      node.attrParams['shape'].value :
                      undefined,
                  dtype: node.attrParams['dtype'] ?
                      node.attrParams['dtype'].value :
                      undefined
              };
          });
      }
      get outputs() {
          return this._outputs.map(node => {
              return {
                  name: node.name,
                  shape: node.attrParams['shape'] ?
                      node.attrParams['shape'].value :
                      undefined,
                  dtype: node.attrParams['dtype'] ?
                      node.attrParams['dtype'].value :
                      undefined
              };
          });
      }
      get inputNodes() {
          return this.placeholders.map(node => node.name);
      }
      get outputNodes() {
          return this.outputs.map(node => node.name);
      }
      getCompilationKey(inputs, outputs) {
          const sortedInputs = inputs.map(node => node.name).sort();
          const sortedOutputs = outputs.map(node => node.name).sort();
          return sortedInputs.join(this.SEPERATOR) + '--' +
              sortedOutputs.join(this.SEPERATOR);
      }
      /**
       * Compiles the inference graph and returns the minimal set of nodes that are
       * required for execution, in the correct execution order.
       */
      compile(inputs, outputs) {
          const executionInfo = getExecutionSubgraph(inputs, outputs, this.weightMap);
          const { missingInputs, dynamicNode, syncInputs } = executionInfo;
          if (dynamicNode != null) {
              throw new Error(`This execution contains the node '${dynamicNode.name}', which has ` +
                  `the dynamic op '${dynamicNode.op}'. Please use ` +
                  `model.executeAsync() instead. Alternatively, to avoid the ` +
                  `dynamic ops, specify the inputs [${syncInputs}]`);
          }
          if (missingInputs.length > 0) {
              const outNames = outputs.map(n => n.name);
              const inNames = Object.keys(inputs);
              throw new Error(`Cannot compute the outputs [${outNames}] from the provided inputs ` +
                  `[${inNames}]. Missing the following inputs: [${missingInputs}]`);
          }
          return getNodesInTopologicalOrder(this.graph, this.weightMap, executionInfo);
      }
      fusePrelu() {
          rewritePrelu(this.graph, this.weightMap);
      }
      /**
       * Executes the inference for given input tensors.
       * @param inputs Tensor map for the model inputs, keyed by the input node
       * names.
       * @param outputs output node name from the Tensorflow model, if no outputs
       * are specified, the default outputs of the model would be used. You can
       * inspect intermediate nodes of the model by adding them to the outputs
       * array.
       */
      async execute(inputs, outputs) {
          console.log("EXECUTE");
          const names = Object.keys(inputs).sort();
          this.checkInputs(inputs);
          this.checkInputShapeAndType(inputs);
          this.checkOutputs(outputs);
          const inputNodes = names.map(name => this.graph.nodes[name]);
          const outputNodes = outputs.map(name => this.graph.nodes[parseNodeName(name)[0]]);
          const compilationKey = this.getCompilationKey(inputNodes, outputNodes);
          // Do nothing if the compiled graph cache contains the input.
          let orderedNodes = this.compiledMap.get(compilationKey);
          if (orderedNodes == null) {
              orderedNodes = this.compile(inputs, outputNodes);
              this.compiledMap.set(compilationKey, orderedNodes);
          }
          const tensorArrayMap = {};
        //   return tfc.tidy(() => {
              const context = new ExecutionContext(this._weightMap, tensorArrayMap);
              const tensorsMap = Object.assign({}, this.weightMap);
              Object.keys(inputs).forEach(name => {
                  tensorsMap[name] = [inputs[name]];
              });
              const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
              const intermediateTensorConsumerCount = {};
              for (let i = 0; i < orderedNodes.length; i++) {
                  const node = orderedNodes[i];
                  if (!tensorsMap[node.name]) {
                      console.log("-----------------------------------");
                      console.log(node);

                      window.logCompileAndRun = true;
                      const tensors = executeOp$g(node, tensorsMap, context);
                      window.logCompileAndRun = false;
                      
                        const times = [];

                        for(let iter=0; iter<20; iter++) {
                            const start = performance.now();
                            const res = executeOp$g(node, tensorsMap, context);
                            await res[0].data();
                            times.push(performance.now() - start);
                        }
                        console.log("min:", Math.min(...times));
                        console.log("avg:", times.reduce((acc, curr) => acc + curr, 0));

                      if (tensors instanceof Promise) {
                          throw new Error(`The execution of the op '${node.op}' returned a promise. ` +
                              `Please use model.executeAsync() instead.`);
                      }
                      tensorsMap[node.name] = tensors;
                      this.checkTensorForDisposal(node.name, node, tensorsMap, context, tensorsToKeep, outputs, intermediateTensorConsumerCount);
                  }
              }
              return outputs.map(name => getTensor(name, tensorsMap, context));
        //   });
      }
      getFrozenTensorIds(tensorMap) {
          const ids = [].concat.apply([], Object.keys(tensorMap)
              .map(key => tensorMap[key])
              .map(tensors => tensors.map(tensor => tensor.id)));
          return new Set(ids);
      }
      checkTensorForDisposal(nodeName, node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount) {
          // Skip output nodes and any control flow nodes, since its dependency is
          // tricky to track correctly.
          if (node.category === 'control' || outputNames.indexOf(nodeName) !== -1) {
              return;
          }
          tensorMap[nodeName].forEach(tensor => {
              if (tensor != null) {
                  intermediateTensorConsumerCount[tensor.id] =
                      (intermediateTensorConsumerCount[tensor.id] || 0) +
                          node.children.length;
              }
          });
          node.inputs.forEach(input => {
              // Skip any control flow nodes, since its dependency is tricky to track
              // correctly.
              if (input.category !== 'control') {
                  const tensors = getTensorsForCurrentContenxt(input.name, tensorMap, context);
                  if (tensors != null) {
                      tensors.forEach(tensor => {
                          if (tensor && !tensorsToKeep.has(tensor.id)) {
                              const count = intermediateTensorConsumerCount[tensor.id];
                              if (count === 1) {
                                  tensor.dispose();
                                  delete intermediateTensorConsumerCount[tensor.id];
                              }
                              else if (count != null) {
                                  // only intermediate nodes has count set, inputs and weights are
                                  // not.
                                  intermediateTensorConsumerCount[tensor.id]--;
                              }
                          }
                      });
                  }
              }
          });
      }
      /**
       * Executes the inference for given input tensors in Async fashion.
       * @param inputs Tensor map for the model inputs, keyed by the input node
       * names.
       * @param outputs output node name from the Tensorflow model, if no outputs
       * are specified, the default outputs of the model would be used. You can
       * inspect intermediate nodes of the model by adding them to the outputs
       * array.
       */
      async executeAsync(inputs, outputs) {
          this.checkInputs(inputs);
          this.checkInputShapeAndType(inputs);
          this.checkOutputs(outputs);
          const tensorArrayMap = {};
          const context = new ExecutionContext(this._weightMap, tensorArrayMap);
          // Graph with control flow op requires runtime evaluation of the execution
          // order, while without control flow the execution order is pre-determined
          // in the compile method.
          const tensorMap = await this.executeWithControlFlow(inputs, context, outputs);
          const results = outputs.map(name => getTensor(name, tensorMap, context));
          // dispose all the intermediate tensors
          const outputIds = new Set(results.map(t => t.id));
          const inputIds = new Set(Object.keys(inputs).map(name => inputs[name].id));
          Object.keys(tensorMap).forEach(key => {
              const tensorArray = tensorMap[key];
              tensorArray.forEach(tensor => {
                  if (tensor && !tensor.isDisposed && !outputIds.has(tensor.id) &&
                      !inputIds.has(tensor.id) &&
                      this.weightIds.indexOf(tensor.id) === -1) {
                      tensor.dispose();
                  }
              });
          });
          return results;
      }
      /**
       * When there are control flow nodes in the graph, the graph execution use
       * ExecutionContext to keep track of the frames and loop iterators.
       * @param inputs placeholder tensors for the graph.
       * @param context the execution context object for current execution.
       */
      async executeWithControlFlow(inputs, context, outputNames) {
          const names = Object.keys(inputs);
          const inputNodes = names.map(name => this.graph.nodes[name]);
          const outputNodes = outputNames.map(name => this.graph.nodes[parseNodeName(name)[0]]);
          const { usedNodes, missingInputs, dynamicNode, syncInputs } = getExecutionSubgraph(inputs, outputNodes, this.weightMap);
          const stack = [...inputNodes, ...this.graph.weights].map(node => {
              return { node, contexts: context.currentContext };
          });
          const tensorsMap = Object.assign({}, this.weightMap);
          Object.keys(inputs).forEach(name => {
              tensorsMap[name] = [inputs[name]];
          });
          const intermediateTensorConsumerCount = {};
          const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
          const added = {};
          while (stack.length > 0) {
              const promises = this.processStack(inputNodes, stack, context, tensorsMap, added, tensorsToKeep, outputNames, intermediateTensorConsumerCount, usedNodes);
              await Promise.all(promises);
          }
          if (dynamicNode == null) {
              console.warn(`This model execution did not contain any nodes with control flow ` +
                  `or dynamic output shapes. You can use model.execute() instead.`);
          }
          const missingOutputs = outputNodes
              .filter(node => !isControlFlow(node) &&
              !getTensor(node.name, tensorsMap, context))
              .map(node => node.name);
          if (missingOutputs.length > 0) {
              let alternativeMsg = '';
              if (dynamicNode != null) {
                  alternativeMsg =
                      `Alternatively, to avoid the dynamic ops, use model.execute() ` +
                          `and specify the inputs [${syncInputs}]`;
              }
              throw new Error(`Cannot compute the outputs [${missingOutputs}] from the provided ` +
                  `inputs [${names}]. Consider providing the following inputs: ` +
                  `[${missingInputs}]. ${alternativeMsg}`);
          }
          return tensorsMap;
      }
      processStack(inputNodes, stack, context, tensorMap, added, tensorsToKeep, outputNames, intermediateTensorConsumerCount, usedNodes) {
          const promises = [];
          while (stack.length > 0) {
              const item = stack.pop();
              context.currentContext = item.contexts;
              let nodeName = '';
              // The tensor of the Enter op with isConstant set should be set
              // in the parent scope, so it will be available as constant for the
              // whole loop.
              if (item.node.op === 'Enter' &&
                  getParamValue('isConstant', item.node, tensorMap, context)) {
                  [nodeName] = getNodeNameAndIndex(item.node.name, context);
              }
              // only process nodes that are not provided as input nodes.
              if (inputNodes.indexOf(item.node) === -1) {
                  const tensors = executeOp$g(item.node, tensorMap, context);
                  if (!nodeName) {
                      [nodeName] = getNodeNameAndIndex(item.node.name, context);
                  }
                  const currentContext = context.currentContext;
                  if (tensors instanceof Promise) {
                      promises.push(tensors.then(t => {
                          tensorMap[nodeName] = t;
                          context.currentContext = currentContext;
                          this.checkTensorForDisposal(nodeName, item.node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount);
                          this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
                          return t;
                      }));
                  }
                  else {
                      tensorMap[nodeName] = tensors;
                      this.checkTensorForDisposal(nodeName, item.node, tensorMap, context, tensorsToKeep, outputNames, intermediateTensorConsumerCount);
                      this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
                  }
              }
              else {
                  this.processChildNodes(item.node, stack, context, tensorMap, added, usedNodes);
              }
          }
          return promises;
      }
      processChildNodes(node, stack, context, tensorMap, added, usedNodes) {
          node.children.forEach((childNode) => {
              const [nodeName,] = getNodeNameAndIndex(childNode.name, context);
              if (added[nodeName] || !usedNodes.has(childNode.name)) {
                  return;
              }
              // Merge op can be pushed if any of its inputs has value.
              if (childNode.op === 'Merge') {
                  if (childNode.inputNames.some(name => {
                      return !!getTensor(name, tensorMap, context);
                  })) {
                      added[nodeName] = true;
                      stack.push({ contexts: context.currentContext, node: childNode });
                  }
              }
              else // Otherwise all inputs must to have value.
               if (childNode.inputNames.every(name => {
                  return !!getTensor(name, tensorMap, context);
              })) {
                  added[nodeName] = true;
                  stack.push({ contexts: context.currentContext, node: childNode });
              }
          });
      }
      /**
       * Releases the memory used by the weight tensors.
       */
      dispose() {
          Object.keys(this.weightMap)
              .forEach(key => this.weightMap[key].forEach(tensor => tensor.dispose()));
      }
      checkInputShapeAndType(inputs) {
          Object.keys(inputs).forEach(name => {
              const input = inputs[name];
              const node = this.graph.nodes[name];
              if (node.attrParams['shape'] && node.attrParams['shape'].value) {
                  const shape = node.attrParams['shape'].value;
                  const match = shape.length === input.shape.length &&
                      input.shape.every((dim, index) => shape[index] === -1 || shape[index] === dim);
                  tfc.util.assert(match, () => `The shape of dict['${node.name}'] provided in ` +
                      `model.execute(dict) must be [${shape}], but was ` +
                      `[${input.shape}]`);
              }
              if (node.attrParams['dtype'] && node.attrParams['dtype'].value) {
                  tfc.util.assert(input.dtype === node.attrParams['dtype'].value, () => `The dtype of dict['${node.name}'] provided in ` +
                      `model.execute(dict) must be ` +
                      `${node.attrParams['dtype'].value}, but was ${input.dtype}`);
              }
          });
      }
      checkInputs(inputs) {
          const notInGraph = Object.keys(inputs).filter(name => !this.graph.nodes[name]);
          if (notInGraph.length > 0) {
              throw new Error(`The dict provided in model.execute(dict) has ` +
                  `keys: [${notInGraph}] that are not part of graph`);
          }
      }
      checkOutputs(outputs) {
          outputs.forEach(name => {
              const [normalizedName] = parseNodeName(name);
              if (!this.graph.nodes[normalizedName]) {
                  throw new Error(`The output '${name}' is not found in the graph`);
              }
          });
      }
  }

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
  const TFHUB_SEARCH_PARAM = '?tfjs-format=file';
  const DEFAULT_MODEL_NAME = 'model.json';
  /**
   * A `tf.GraphModel` is a directed, acyclic graph of built from
   * SavedModel GraphDef and allows inference exeuction.
   *
   * A `tf.GraphModel` can only be created by loading from a model converted from
   * a [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) using
   * the command line converter tool and loaded via `tf.loadGraphModel`.
   */
  /** @doc {heading: 'Models', subheading: 'Classes'} */
  class GraphModel {
      /**
       * @param modelUrl url for the model, or an `io.IOHandler`.
       * @param weightManifestUrl url for the weight file generated by
       * scripts/convert.py script.
       * @param requestOption options for Request, which allows to send credentials
       * and custom headers.
       * @param onProgress Optional, progress callback function, fired periodically
       * before the load is completed.
       */
      constructor(modelUrl, loadOptions = {}) {
          this.modelUrl = modelUrl;
          this.loadOptions = loadOptions;
          this.version = 'n/a';
          if (loadOptions == null) {
              this.loadOptions = {};
          }
      }
      // Returns the version information for the tensorflow model GraphDef.
      get modelVersion() {
          return this.version;
      }
      get inputNodes() {
          return this.executor.inputNodes;
      }
      get outputNodes() {
          return this.executor.outputNodes;
      }
      get inputs() {
          return this.executor.inputs;
      }
      get outputs() {
          return this.executor.outputs;
      }
      get weights() {
          return this.executor.weightMap;
      }
      /**
       * There's no native PRelu op in TensorFlow, so Keras generates the following
       * structure which does the equivalent calculation:
       * f(x) = Relu(x) + (-alpha * Relu(-x))
       * Since tfjs-core has a prelu op, this method will fuse the TensorFlow
       * generated ops into prelu op. It will also try to register a custom op that
       * supports prelu op.
       */
      fusePrelu() {
          this.executor.fusePrelu();
          if (getRegisteredOp('Prelu') == null) {
              registerOp('Prelu', (node) => {
                  const x = node.inputs[0];
                  const alpha = node.inputs[1];
                  return tfc.prelu(x, alpha);
              });
          }
      }
      findIOHandler() {
          const path = this.modelUrl;
          if (path.load != null) {
              // Path is an IO Handler.
              this.handler = path;
          }
          else if (this.loadOptions.requestInit != null) {
              this.handler = tfc.io.browserHTTPRequest(path, this.loadOptions);
          }
          else {
              const handlers = tfc.io.getLoadHandlers(path, this.loadOptions.onProgress);
              if (handlers.length === 0) {
                  // For backward compatibility: if no load handler can be found,
                  // assume it is a relative http path.
                  handlers.push(tfc.io.browserHTTPRequest(path, this.loadOptions));
              }
              else if (handlers.length > 1) {
                  throw new Error(`Found more than one (${handlers.length}) load handlers for ` +
                      `URL '${[path]}'`);
              }
              this.handler = handlers[0];
          }
      }
      /**
       * Loads the model and weight files, construct the in memory weight map and
       * compile the inference graph.
       */
      async load() {
          this.findIOHandler();
          if (this.handler.load == null) {
              throw new Error('Cannot proceed with model loading because the IOHandler provided ' +
                  'does not have the `load` method implemented.');
          }
          const artifacts = await this.handler.load();
          const graph = artifacts.modelTopology;
          this.version = `${graph.versions.producer}.${graph.versions.minConsumer}`;
          const weightMap = tfc.io.decodeWeights(artifacts.weightData, artifacts.weightSpecs);
          this.executor =
              new GraphExecutor(OperationMapper.Instance.transformGraph(graph));
          this.executor.weightMap = this.convertTensorMapToTensorsMap(weightMap);
          return true;
      }
      /**
       * Execute the inference for the input tensors.
       *
       * @param input The input tensors, when there is single input for the model,
       * inputs param should be a `tf.Tensor`. For models with mutliple inputs,
       * inputs params should be in either `tf.Tensor`[] if the input order is
       * fixed, or otherwise NamedTensorMap format.
       *
       * For model with multiple inputs, we recommend you use NamedTensorMap as the
       * input type, if you use `tf.Tensor`[], the order of the array needs to
       * follow the
       * order of inputNodes array. @see {@link GraphModel.inputNodes}
       *
       * You can also feed any intermediate nodes using the NamedTensorMap as the
       * input type. For example, given the graph
       *    InputNode => Intermediate => OutputNode,
       * you can execute the subgraph Intermediate => OutputNode by calling
       *    model.execute('IntermediateNode' : tf.tensor(...));
       *
       * This is useful for models that uses tf.dynamic_rnn, where the intermediate
       * state needs to be fed manually.
       *
       * For batch inference execution, the tensors for each input need to be
       * concatenated together. For example with mobilenet, the required input shape
       * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
       * If we are provide a batched data of 100 images, the input tensor should be
       * in the shape of [100, 244, 244, 3].
       *
       * @param config Prediction configuration for specifying the batch size and
       * output node names. Currently the batch size option is ignored for graph
       * model.
       *
       * @returns Inference result tensors. The output would be single `tf.Tensor`
       * if model has single output node, otherwise Tensor[] or NamedTensorMap[]
       * will be returned for model with multiple outputs.
       */
      /** @doc {heading: 'Models', subheading: 'Classes'} */
      predict(inputs, config) {
          return this.execute(inputs, this.outputNodes);
      }
      normalizeInputs(inputs) {
          if (!(inputs instanceof tfc.Tensor) && !Array.isArray(inputs)) {
              // The input is already a NamedTensorMap.
              return inputs;
          }
          inputs = Array.isArray(inputs) ? inputs : [inputs];
          if (inputs.length !== this.inputNodes.length) {
              throw new Error('Input tensor count mismatch,' +
                  `the graph model has ${this.inputNodes.length} placeholders, ` +
                  `while there are ${inputs.length} input tensors.`);
          }
          return this.inputNodes.reduce((map, inputName, i) => {
              map[inputName] = inputs[i];
              return map;
          }, {});
      }
      normalizeOutputs(outputs) {
          outputs = outputs || this.outputNodes;
          return !Array.isArray(outputs) ? [outputs] : outputs;
      }
      /**
       * Executes inference for the model for given input tensors.
       * @param inputs tensor, tensor array or tensor map of the inputs for the
       * model, keyed by the input node names.
       * @param outputs output node name from the Tensorflow model, if no
       * outputs are specified, the default outputs of the model would be used.
       * You can inspect intermediate nodes of the model by adding them to the
       * outputs array.
       *
       * @returns A single tensor if provided with a single output or no outputs
       * are provided and there is only one default output, otherwise return a
       * tensor array. The order of the tensor array is the same as the outputs
       * if provided, otherwise the order of outputNodes attribute of the model.
       */
      /** @doc {heading: 'Models', subheading: 'Classes'} */
      execute(inputs, outputs) {
          inputs = this.normalizeInputs(inputs);
          outputs = this.normalizeOutputs(outputs);
          const result = this.executor.execute(inputs, outputs);
          return result.length > 1 ? result : result[0];
      }
      /**
       * Executes inference for the model for given input tensors in async
       * fashion, use this method when your model contains control flow ops.
       * @param inputs tensor, tensor array or tensor map of the inputs for the
       * model, keyed by the input node names.
       * @param outputs output node name from the Tensorflow model, if no outputs
       * are specified, the default outputs of the model would be used. You can
       * inspect intermediate nodes of the model by adding them to the outputs
       * array.
       *
       * @returns A Promise of single tensor if provided with a single output or
       * no outputs are provided and there is only one default output, otherwise
       * return a tensor map.
       */
      /** @doc {heading: 'Models', subheading: 'Classes'} */
      async executeAsync(inputs, outputs) {
          inputs = this.normalizeInputs(inputs);
          outputs = this.normalizeOutputs(outputs);
          const result = await this.executor.executeAsync(inputs, outputs);
          return result.length > 1 ? result : result[0];
      }
      convertTensorMapToTensorsMap(map) {
          return Object.keys(map).reduce((newMap, key) => {
              newMap[key] = [map[key]];
              return newMap;
          }, {});
      }
      /**
       * Releases the memory used by the weight tensors.
       */
      /** @doc {heading: 'Models', subheading: 'Classes'} */
      dispose() {
          this.executor.dispose();
      }
  }
  /**
   * Load a graph model given a URL to the model definition.
   *
   * Example of loading MobileNetV2 from a URL and making a prediction with a
   * zeros input:
   *
   * ```js
   * const modelUrl =
   *    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
   * const model = await tf.loadGraphModel(modelUrl);
   * const zeros = tf.zeros([1, 224, 224, 3]);
   * model.predict(zeros).print();
   * ```
   *
   * Example of loading MobileNetV2 from a TF Hub URL and making a prediction with
   * a zeros input:
   *
   * ```js
   * const modelUrl =
   *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
   * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
   * const zeros = tf.zeros([1, 224, 224, 3]);
   * model.predict(zeros).print();
   * ```
   * @param modelUrl The url or an `io.IOHandler` that loads the model.
   * @param options Options for the HTTP request, which allows to send credentials
   *    and custom headers.
   */
  /** @doc {heading: 'Models', subheading: 'Loading'} */
  async function loadGraphModel(modelUrl, options = {}) {
      if (modelUrl == null) {
          throw new Error('modelUrl in loadGraphModel() cannot be null. Please provide a url ' +
              'or an IOHandler that loads the model');
      }
      if (options == null) {
          options = {};
      }
      if (options.fromTFHub) {
          if (modelUrl.load == null) {
              if (!modelUrl.endsWith('/')) {
                  modelUrl = modelUrl + '/';
              }
              modelUrl = `${modelUrl}${DEFAULT_MODEL_NAME}${TFHUB_SEARCH_PARAM}`;
          }
      }
      const model = new GraphModel(modelUrl, options);
      await model.load();
      return model;
  }

  /** @license See the LICENSE file. */
  // This code is auto-generated, do not modify this file!
  const version = '1.2.7';

  exports.GraphModel = GraphModel;
  exports.deregisterOp = deregisterOp;
  exports.loadGraphModel = loadGraphModel;
  exports.registerOp = registerOp;
  exports.version_converter = version;

  Object.defineProperty(exports, '__esModule', { value: true });

}));
//# sourceMappingURL=tf-converter.js.map
