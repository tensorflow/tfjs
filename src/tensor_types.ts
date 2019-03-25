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

import {Tensor, Variable} from './tensor';
import {DataType} from './types';

/** @docalias {[name: string]: Tensor} */
export type NamedTensorMap = {
  [name: string]: Tensor;
};

export type NamedVariableMap = {
  [name: string]: Variable;
};

export type GradSaveFunc = (save: Tensor[]) => void;

/**
 * @docalias void|number|string|TypedArray|Tensor|Tensor[]|{[key:
 * string]:Tensor|number|string}
 */
export type TensorContainer =
    void|Tensor|string|number|boolean|TensorContainerObject|
    TensorContainerArray|Float32Array|Int32Array|Uint8Array;
export interface TensorContainerObject {
  [x: string]: TensorContainer;
}
export interface TensorContainerArray extends Array<TensorContainer> {}

export interface TensorInfo {
  // Name of the tensor.
  name: string;
  // Tensor shape information, Optional.
  shape?: number[];
  // Data type of the tensor.
  dtype: DataType;
}
