/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {NDArrayMath} from '../math/math';
import {NDArray} from '../math/ndarray';

import {Tensor} from './graph';

/**
 * TensorArrayMap is an internal map from Tensor IDs to NDArrays. Since NDArrays
 * can be backed by WebGL textures, the TensorArrayMap is only used inside of a
 * Session.
 */
export abstract class TensorArrayMapBase {
  /**
   * Returns the NDArray associated with the provided tensor. Will throw an
   * exception if the tensor is not a key in the map, or if the associated
   * NDArray is null.
   * @param tensor The tensor key.
   * @param skipChecks False by default. If true will skip all checks.
   * @return The NDArray associated with the tensor.
   */
  get(tensor: Tensor, skipChecks = false): NDArray {
    if (!skipChecks && this.dict[tensor.id] === undefined) {
      throw new Error(`tensor ${tensor.id} not in array map.`);
    }
    const nda = this.dict[tensor.id];
    if (!skipChecks && nda === null) {
      throw new Error(`tensor ${tensor.id} has null array.`);
    }
    return nda;
  }

  /**
   * Removes a tensor/NDArray pair from the map.
   * @param tensor The tensor key.
   */
  delete(tensor: Tensor) {
    delete this.dict[tensor.id];
  }

  /**
   * Nullifies a tensor pair from the map.
   * @param tensor The tensor key.
   */
  nullify(tensor: Tensor) {
    this.dict[tensor.id] = null;
  }

  disposeArray(tensor: Tensor) {
    if (this.dict[tensor.id] === undefined) {
      return;
    }
    const nda = this.dict[tensor.id];
    if (nda === null) {
      return;
    }
    nda.dispose();
    this.dict[tensor.id] = null;
  }

  /**
   * @return The number of tensor/NDArray pairs in the map.
   */
  size(): number {
    return Object.keys(this.dict).length;
  }

  /**
   * Iterate over all contained NDArray values and dispose them.
   */
  dispose() {
    Object.keys(this.dict).forEach(tensorID => {
      const nda = this.dict[+tensorID];
      if (nda) {
        nda.dispose();
      }
    });
    this.dict = {};
  }

  /**
   * Tests to see if a tensor has a null associated with it. Throws
   * if the tensor is not a key in the map.
   * @param tensor The tensor key.
   * @return True if the associated NDArray is null, else False.
   */
  hasNullArray(tensor: Tensor): boolean {
    if (this.dict[tensor.id] === undefined) {
      throw new Error(`tensor ${tensor.id} not in array map.`);
    }
    return this.dict[tensor.id] === null;
  }

  protected dict: {[tensorID: number]: NDArray | null} = {};
}

export class TensorArrayMap extends TensorArrayMapBase {
  /**
   * Add or replace an entry in the map.
   * @param tensor The tensor key.
   * @param array The NDArray value, can be null.
   */
  set(tensor: Tensor, array: NDArray|null) {
    this.dict[tensor.id] = array;
  }
}

export class SummedTensorArrayMap extends TensorArrayMapBase {
  constructor(private math: NDArrayMath) {
    super();
  }

  /**
   * Aggregate by summing to an entry in the map.
   * @param tensor The tensor key.
   * @param array The NDArray value.
   */
  add(tensor: Tensor, array: NDArray) {
    if (this.dict[tensor.id] == null) {
      this.dict[tensor.id] = this.math.keep(array);
    } else {
      const oldValue = this.get(tensor);
      const newValue = this.math.keep(this.math.addStrict(oldValue, array));
      this.dict[tensor.id] = newValue;
      oldValue.dispose();
    }
  }
}
