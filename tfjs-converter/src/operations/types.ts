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
import {Tensor} from '@tensorflow/tfjs-core';

import * as tensorflow from '../data/compiled_api';
import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';
import {ResourceManager} from '../executor/resource_manager';

export type ParamType = 'number'|'string'|'string[]'|'number[]'|'bool'|'bool[]'|
    'shape'|'shape[]'|'tensor'|'tensors'|'dtype'|'dtype[]'|'func';
export type Category = 'arithmetic'|'basic_math'|'control'|'convolution'|
    'creation'|'custom'|'dynamic'|'evaluation'|'graph'|'hash_table'|'image'|
    'logical'|'matrices'|'normalization'|'reduction'|'slice_join'|'sparse'|
    'spectral'|'string'|'transformation';

// For mapping input or attributes of NodeDef into TensorFlow.js op param.
export declare interface ParamMapper {
  // tensorflow.js name for the field, it should be in camelcase format.
  name: string;
  type: ParamType;
  defaultValue?: ValueType;
  notSupported?: boolean;
}

// For mapping the input of TensorFlow NodeDef into TensorFlow.js Op param.
export declare interface InputParamMapper extends ParamMapper {
  // The first number is the starting index of the param, the second number is
  // the length of the param. If the length value is positive number, it
  // represents the true length of the param. Otherwise, it represents a
  // variable length, the value is the index go backward from the end of the
  // array.
  // For example `[0, 5]`: this param is the array of input tensors starting at
  // index 0 and with the length of 5.
  // For example `[1, -1]`: this param is the array of input tensors starting at
  // index 1 and with the `inputs.length - 1`.
  // Zero-based index at where in the input array this param starts.
  // A negative index can be used, indicating an offset from the end of the
  // sequence. slice(-2) extracts the last two elements in the sequence.
  start: number;
  // Zero-based index before where in the input array the param ends. The
  // mapping is up to but not including end. For example, start = 1, end = 4
  // includes the second element through the fourth element (elements indexed 1,
  // 2, and 3). A negative index can be used, indicating an offset from the end
  // of the sequence. start = 2, end = -1 includes the third element through the
  // second-to-last element in the sequence. If end is omitted, end is set to
  // start + 1, the mapping only include the single element at start index. If
  // end is set to 0, the mapping is through the end of the input array
  // (arr.length). If end is greater than the length of the inputs, mapping
  // inncludes through to the end of the sequence (arr.length).
  end?: number;
}

// For mapping the attributes of TensorFlow NodeDef into TensorFlow.js op param.
export declare interface AttrParamMapper extends ParamMapper {
  // TensorFlow attribute name, this should be set if the tensorflow attribute
  // name is different form the tensorflow.js name.
  tfName?: string;
  // TensorFlow deprecated attribute name, this is used to support old models.
  tfDeprecatedName?: string;
}

export interface InternalOpExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext): Tensor
      |Tensor[];
}

export interface InternalOpAsyncExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
   resourceManager?: ResourceManager): Promise<Tensor[]>;
}

export declare interface OpMapper {
  tfOpName: string;
  category?: Category;
  inputs?: InputParamMapper[];
  attrs?: AttrParamMapper[];
  outputs?: string[];
  customExecutor?: OpExecutor;
}

export declare interface Node {
  signatureKey?: string;
  name: string;
  op: string;
  category: Category;
  inputNames: string[];
  inputs: Node[];
  inputParams: {[key: string]: InputParamValue};
  attrParams: {[key: string]: ParamValue};
  children: Node[];
  rawAttrs?: {[k: string]: tensorflow.IAttrValue};
  defaultOutput?: number;
  outputs?: string[];
}

export declare interface Graph {
  nodes: {[key: string]: Node};
  placeholders: Node[];
  inputs: Node[];
  outputs: Node[];
  weights: Node[];
  signature?: tensorflow.ISignatureDef;
  functions?: {[key: string]: Graph};
  initNodes?: Node[];
}

export type ValueType = string|string[]|number|number[]|number[][]|boolean|
    boolean[]|Tensor|Tensor[];
export declare interface ParamValue {
  value?: ValueType;
  type: ParamType;
}

export declare interface InputParamValue extends ParamValue {
  inputIndexStart?: number;
  inputIndexEnd?: number;
}

export interface OpExecutor {
  (node: GraphNode): Tensor|Tensor[]|Promise<Tensor|Tensor[]>;
}

export interface GraphNode {
  inputs: Tensor[];
  attrs: {[key: string]: ValueType};
}
