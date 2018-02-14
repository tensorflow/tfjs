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

import {doc} from './doc';
import {CustomGradientFunc} from './engine';
import {ENV} from './environment';
import {tidy} from './globals';
import {ScopeFn, ScopeResult} from './tape_util';
import {Scalar, Tensor, Variable} from './tensor';
import {NamedTensorMap} from './types';

export class Gradients {
  /**
   * Create a new gradient scope. Similar to scope, but forces all inner scopes
   * to not clean up so that gradient operations can be used inside of this
   * scope.
   * @param nameOrScopeFn The name of the scope, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If a name is provided, and debug mode is on, the timing and the memory
   *     usage of the function will be tracked and displayed on the console
   *     using the provided name.
   * @param scopeFn The function to execute.
   */
  static gradScope<T extends ScopeResult>(
      nameOrScopeFn: string|ScopeFn<T>, scopeFn?: ScopeFn<T>): T {
    return tidy(nameOrScopeFn, scopeFn, true /* gradScope */);
  }

  /**
   * Computes the gradient of `f(x)` w.r.t. `x`.
   *
   * `f(x)` must take a single tensor `x`. Returns another function `g(x, dy?)`,
   * which when called gives `df/dx`. If `dy` is provided, the gradient of
   * `f(x).mul(dy).sum()` w.r.t. `x` is computed instead.
   *
   * If `f()` takes multiple inputs, use `grads` instead.
   *
   * @param f The function f(x), to compute gradient for.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static grad<I extends Tensor, O extends Tensor>(f: (x: I) => O):
      (x: I, dy?: O) => I {
    return (x: I, dy?: O): I => {
      const {value, grads} = ENV.engine.gradients(() => f(x), [x], dy);
      value.dispose();
      checkGrads(grads);
      return grads[0] as I;
    };
  }

  /**
   * Computes the gradients of `f(x1, x2,...)` w.r.t. each input `x1`, `x2`,...
   * Returns another function `g([x1, x2,...], dy?)`, which when called gives
   * an array of tensors: `[df/dx1, df/dx2,...]` evaluated at `[x1, x2,...]`.
   * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` w.r.t. each input
   * is computed instead.
   *
   * If `f()` takes a single input, use `grad` instead.
   *
   * @param f The function `f(x1, x2,...)` to compute gradients for.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static grads<O extends Tensor>(f: (...args: Tensor[]) => O):
      (args: Tensor[], dy?: O) => Tensor[] {
    return (args: Tensor[], dy?: O): Tensor[] => {
      const {value, grads} = ENV.engine.gradients(() => f(...args), args, dy);
      value.dispose();
      checkGrads(grads);
      return grads;
    };
  }

  /**
   * Like `dl.grad`, but returns also the value of `f()`. Useful when `f()`
   * returns a metric you want to show. The result is a rich object with
   * the following properties:
   * - grad: The gradient of `f(x)` w.r.t `x` (result of `grad`).
   * - value: The value returned by `f(x)`.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static valueAndGrad<I extends Tensor, O extends Tensor>(f: (x: I) => O):
      (x: I, dy?: O) => {
        value: O;
        grad: I;
      } {
    return (x: I, dy?: O) => {
      const {grads, value} = ENV.engine.gradients(() => f(x), [x], dy);
      checkGrads(grads);
      return {grad: grads[0] as I, value: value as O};
    };
  }

  /**
   * Like `grads`, but returns also the value of `f()`. Useful when `f()`
   * returns a metric you want to show. The result is a rich object with
   * the following properties:
   * - grads: The gradients of `f()` w.r.t each input (result of `grads`).
   * - value: The value returned by `f(x)`.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static valueAndGrads<O extends Tensor>(f: (...args: Tensor[]) => O):
      (args: Tensor[], dy?: O) => {
        grads: Tensor[];
        value: O;
      } {
    return (args: Tensor[], dy?: O) => {
      const res = ENV.engine.gradients(() => f(...args), args, dy);
      checkGrads(res.grads);
      return res;
    };
  }

  /**
   * Computes and returns the gradient of f(x) with respect to the list of
   * trainable variables provided by `varList`. If no list is provided, it
   * defaults to all trainable variables.
   * @param f The function to execute. f() should return a scalar.
   * @param varList An optional list of variables to provide gradients with
   *     respect to. Defaults to all trainable variables.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static variableGrads(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, grads: NamedTensorMap} {
    if (varList == null) {
      // Get all of the trainable variables.
      varList = [];
      for (const varName in ENV.engine.registeredVariables) {
        varList.push(ENV.engine.registeredVariables[varName]);
      }
    }
    // Prune non-trainable variables.
    varList = varList.filter(variable => variable.trainable);
    const {value, grads} = ENV.engine.gradients(f, varList);
    if (value.rank > 0) {
      throw new Error(
          `The user-provided function must return a Scalar, but it ` +
          `returned a rank-${value.rank} tensor`);
    }
    const namedGrads: NamedTensorMap = {};
    varList.forEach((v, i) => {
      if (grads[i] != null) {
        namedGrads[v.name] = grads[i];
      }
    });
    return {value, grads: namedGrads};
  }

  /**
   * Overrides the gradient computation of a function `f`.
   *
   * Takes a function `f(...inputs) => {value: Tensor,
   * gradFunc: dy => Tensor[]}` and returns another function `g(...inputs)`
   * which takes the same inputs as `f`. When called, `g` returns `f().value`.
   * In backward mode, custom gradients w.r.t. each input of `f` are computed
   * using `f().gradFunc`.
   *
   * @param f The function to evaluate in forward mode, which should return
   *     `{value: Tensor, gradFunc: (dy) => Tensor[]}`, where `gradFunc` returns
   *     the custom gradients of `f` w.r.t. its inputs.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static customGrad<T extends Tensor>(f: CustomGradientFunc<T>):
      (...args: Tensor[]) => T {
    return ENV.engine.customGrad(f);
  }
}

function checkGrads(grads: Tensor[]) {
  const numNullGradients = grads.filter(g => g == null).length;
  if (numNullGradients > 0) {
    throw new Error(
        `Cannot compute gradient: y is not a function of \`x\`s. ` +
        `Make sure the xs you are computing gradients with respect ` +
        `to are used inside the gradient function.`);
  }
}
