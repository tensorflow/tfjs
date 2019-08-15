
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

import {OpExecutor, OpMapper} from '../types';

const CUSTOM_OPS: {[key: string]: OpMapper} = {};

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
export function registerOp(name: string, opFunc: OpExecutor) {
  const opMapper: OpMapper = {
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

export function getRegisteredOp(name: string): OpMapper {
  return CUSTOM_OPS[name];
}

/**
 * Deregister the Op for graph model executor.
 *
 * @param name The Tensorflow Op name.
 */
/** @doc {heading: 'Models', subheading: 'Op Registry'} */
export function deregisterOp(name: string) {
  delete CUSTOM_OPS[name];
}
