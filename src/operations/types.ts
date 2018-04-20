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
export type ParamTypes =
    'number'|'string'|'number[]'|'bool'|'shape'|'tensor'|'tensors'|'dtype';
export type Category = 'arithmetic'|'basic_math'|'control'|'convolution'|
    'image'|'creation'|'graph'|'logical'|'matrices'|'normalization'|'reduction'|
    'slice_join'|'transformation';
export interface ParamMapper {
  tfParamName?: string;
  tfParamNameDeprecated?: string;
  tfInputIndex?: number;
  tfInputParamLength?: number;
  dlParamName: string;
  type: ParamTypes;
  converter?: string;
  defaultValue?: string|string[]|number|number[]|boolean|boolean[];
  notSupported?: boolean;
}

export interface OpMapper {
  tfOpName: string;
  dlOpName: string;
  category: Category;
  params: ParamMapper[];
  unsupportedParams: string[];
}

export interface Node {
  name: string;
  op: string;
  category: Category;
  inputNames: string[];
  inputs: Node[];
  params: {[key: string]: ParamValue};
  children: Node[];
}

export interface Graph {
  nodes: {[key: string]: Node};
  placeholders: Node[];
  inputs: Node[];
  outputs: Node[];
  withControlFlow: boolean;
}

export type ValueType =
    string|string[]|number|number[]|boolean|boolean[]|Tensor|Tensor[];
export interface ParamValue {
  value?: ValueType;
  inputIndex?: number;
  inputParamLength?: number;
  type: ParamTypes;
}
