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

import {ENV} from '../environment';
import * as util from '../util';
import {DataType, NDArray, Rank} from './ndarray';

export class Variable<D extends DataType = DataType, R extends Rank = Rank>
    extends NDArray<D, R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling super().
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: NDArray<D, R>, public trainable = true, name?: string) {
    super(
        initialValue.shape, initialValue.dtype, null /* values */,
        initialValue.id);
    this.name = name;
    if (this.name == null) {
      this.name = Variable.nextVarId.toString();
      Variable.nextVarId++;
    }
    ENV.math.registerVariable(this);
    ENV.math.keep(this);
  }

  /**
   * Creates a new variable with the provided initial value.
   *
   * @param initialValue An ndarray.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  static variable<D extends DataType, R extends Rank>(
      initialValue: NDArray<D, R>, trainable = true, name?: string,
      dtype?: D): Variable<D, R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype);
    }
    return new Variable(initialValue, trainable, name);
  }

  /** Assign a new array to this variable. The old array will be disposed. */
  assign(newValue: NDArray<D, R>): void {
    if (newValue.dtype !== this.dtype) {
      throw new Error(
          `dtype of the new value (${newValue.dtype}) and ` +
          `previous value (${this.dtype}) must match`);
    }
    if (!util.arraysEqual(newValue.shape, this.shape)) {
      throw new Error(
          `shape of the new value (${newValue.shape}) and ` +
          `previous value (${this.shape}) must match`);
    }
    this.math.disposeData(this.id);
    this.id = newValue.id;
    ENV.math.keep(this);
  }
}

const variable = Variable.variable;
export {variable};
