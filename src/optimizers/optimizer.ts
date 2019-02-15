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

import {variableGrads} from '../globals';
import {Serializable} from '../serialization';
import {Scalar, Variable} from '../tensor';
import {NamedTensorMap} from '../tensor_types';

/** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
export abstract class Optimizer extends Serializable {
  /**
   * Executes `f()` and minimizes the scalar output of `f()` by computing
   * gradients of y with respect to the list of trainable variables provided by
   * `varList`. If no list is provided, it defaults to all trainable variables.
   *
   * @param f The function to execute and whose output to minimize.
   * @param returnCost Whether to return the scalar cost value produced by
   * executing `f()`.
   * @param varList An optional list of variables to update. If specified, only
   * the trainable variables in varList will be updated by minimize. Defaults to
   * all trainable variables.
   */
  /** @doc {heading: 'Training', subheading: 'Optimizers'} */
  minimize(f: () => Scalar, returnCost = false, varList?: Variable[]): Scalar
      |null {
    const {value, grads} = this.computeGradients(f, varList);

    this.applyGradients(grads);

    // Dispose gradients.
    const varNames = Object.keys(grads);
    varNames.forEach(varName => grads[varName].dispose());

    if (returnCost) {
      return value as Scalar;
    } else {
      value.dispose();
      return null;
    }
  }

  /**
   * Executes f() and computes the gradient of the scalar output of f() with
   * respect to the list of trainable variables provided by `varList`. If no
   * list is provided, it defaults to all trainable variables.
   *
   * @param f The function to execute and whose output to use for computing
   * gradients with respect to variables.
   * @param varList An optional list of variables to compute gradients with
   * respect to. If specified, only the trainable variables in varList will have
   * gradients computed with respect to. Defaults to all trainable variables.
   */
  computeGradients(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, grads: NamedTensorMap} {
    return variableGrads(f, varList);
  }

  /**
   * Updates variables by using the computed gradients.
   *
   * @param variableGradients A mapping of variable name to its gradient value.
   */
  abstract applyGradients(variableGradients: NamedTensorMap): void;

  /**
   * Dispose the variables (if any) owned by this optimizer instance.
   */
  dispose(): void {}
}

Object.defineProperty(Optimizer, Symbol.hasInstance, {
  value: (instance: Optimizer) => {
    return instance.minimize != null && instance.computeGradients != null &&
        instance.applyGradients != null;
  }
});
