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

declare class Context {
  constructor();
}

declare class TensorHandle {
  constructor(shape: number[], dtype: number);
  bindBuffer(buffer: Float32Array|Int32Array|Uint8Array): void;
  data(): Float32Array|Int32Array|Uint8Array;

  shape: number[];
  dtype: number;
}

// TFE Op Attr class.
declare class TFEOpAttr {
  name: string;
  type: number;
  value: number|boolean|object;
}

// TF Types
export const TF_FLOAT: number;
export const TF_INT32: number;
export const TF_BOOL: number;

// TF OpAttrTypes
export const TF_ATTR_STRING: number;
export const TF_ATTR_INT: number;
export const TF_ATTR_BOOL: number;
export const TF_ATTR_TYPE: number;
export const TF_ATTR_SHAPE: number;
export const TF_ATTR_TENSOR: number;
export const TF_ATTR_PLACEHOLDER: number;
export const TF_ATTR_FUNC: number;

export const TF_Version: string;

export function execute(
    context: Context, op: string, op_attrs: TFEOpAttr[],
    inputs: TensorHandle[], output: TensorHandle): void;
