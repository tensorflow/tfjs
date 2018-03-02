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

declare class Context { constructor(); }

declare class TensorHandle {
  constructor();
  constructor(shape: number[], dtype: number);
  bindBuffer(buffer: Float32Array|Int32Array|Uint8Array): void;
  dataSync(): Float32Array|Int32Array|Uint8Array;

  shape: number[];
  dtype: number;
}

declare class TFEOpAttr {
  name: string;
  type: number;
  value: number|boolean|object;
}

export interface TFJSBinding {
  Context: typeof Context;
  TensorHandle: typeof TensorHandle;
  TFEOpAttr: typeof TFEOpAttr;

  // TF Types
  TF_FLOAT: number;
  TF_INT32: number;
  TF_BOOL: number;

  // TF OpAttrTypes
  TF_ATTR_STRING: number;
  TF_ATTR_INT: number;
  TF_ATTR_FLOAT: number;
  TF_ATTR_BOOL: number;
  TF_ATTR_TYPE: number;
  TF_ATTR_SHAPE: number;

  TF_Version: string;

  execute(
      context: Context, op: string, op_attrs: TFEOpAttr[],
      inputs: TensorHandle[], output: TensorHandle): void;
}
