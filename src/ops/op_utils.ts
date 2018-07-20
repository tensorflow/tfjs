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

import * as tfc from '@tensorflow/tfjs-core';
import {isArray, isNullOrUndefined} from 'util';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {TFEOpAttr} from '../tfjs_binding';

let gBackend: NodeJSKernelBackend = null;

/** Returns an instance of the Node.js backend. */
export function nodeBackend(): NodeJSKernelBackend {
  if (gBackend === null) {
    gBackend = (tfc.ENV.findBackend('tensorflow') as NodeJSKernelBackend);
  }
  return gBackend;
}

/** Returns the TF dtype for a given DataType. */
export function getTFDType(dataType: tfc.DataType): number {
  const binding = nodeBackend().binding;
  switch (dataType) {
    case 'float32':
      return binding.TF_FLOAT;
    case 'int32':
      return binding.TF_INT32;
    case 'bool':
      return binding.TF_BOOL;
    default:
      throw new Error('Unknown dtype `${dtype}`');
  }
}

/** Creates a TFEOpAttr for a 'type' OpDef attribute. */
export function createTypeOpAttr(
    attrName: string, dtype: tfc.DataType): TFEOpAttr {
  return {
    name: attrName,
    type: nodeBackend().binding.TF_ATTR_TYPE,
    value: getTFDType(dtype)
  };
}

/** Returns the dtype number for a single or list of input Tensors. */
export function getTFDTypeForInputs(tensors: tfc.Tensor|tfc.Tensor[]): number {
  if (isNullOrUndefined(tensors)) {
    throw new Error('Invalid input tensors value.');
  }
  if (isArray(tensors)) {
    for (let i = 0; i < tensors.length; i++) {
      return getTFDType(tensors[i].dtype);
    }
    return -1;
  } else {
    return getTFDType(tensors.dtype);
  }
}
