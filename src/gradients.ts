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
import * as util from './util';

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
   * Provided `f(x)`, returns another function `g(x, dy?), which gives the
   * gradient of `f(x)` with respect to `x`.
   *
   * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` with respect to
   * `x` is computed instead. `f(x)` must take a single tensor `x` and return a
   * single tensor `y`. If `f()` takes multiple inputs, use `grads` instead.
   *
   * @param f The function f(x), to compute gradient for.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static grad<I extends Tensor, O extends Tensor>(f: (x: I) => O):
      (x: I, dy?: O) => I {
    util.assert(
        util.isFunction(f), 'The f passed in grad(f) must be a function');
    return (x: I, dy?: O): I => {
      util.assert(
          x instanceof Tensor, 'The x passed in grad(f)(x) must be a tensor');
      util.assert(
          dy == null || dy instanceof Tensor,
          'The dy passed in grad(f)(x, dy) must be a tensor');
      const {value, grads} = ENV.engine.gradients(() => f(x), [x], dy);
      if (dy != null) {
        util.assertShapesMatch(
            value.shape, dy.shape,
            'The shape of dy passed in grad(f)(x, dy) must match the shape ' +
                'returned by f(x)');
      }
      value.dispose();
      checkGrads(grads);
      return grads[0] as I;
    };
  }

  /**
   * Provided `f(x1, x2,...)`, returns another function `g([x1, x2,...], dy?)`,
   * which gives an array of gradients of `f()` with respect to each input
   * [`x1`,`x2`,...].
   *
   * If `dy` is passed when calling `g()`, the gradient of
   * `f(x1,...).mul(dy).sum()` with respect to each input is computed instead.
   * The provided `f` must take one or more tensors and return a single tensor
   * `y`. If `f()` takes a single input, we recommend using `grad` instead.
   *
   * @param f The function `f(x1, x2,...)` to compute gradients for.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static grads<O extends Tensor>(f: (...args: Tensor[]) => O):
      (args: Tensor[], dy?: O) => Tensor[] {
    util.assert(
        util.isFunction(f), 'The f passed in grads(f) must be a function');
    return (args: Tensor[], dy?: O): Tensor[] => {
      util.assert(
          Array.isArray(args) && args.every(arg => arg instanceof Tensor),
          'The args passed in grads(f)(args) must be an array of tensors');
      util.assert(
          dy == null || dy instanceof Tensor,
          'The dy passed in grads(f)(args, dy) must be a tensor');
      const {value, grads} = ENV.engine.gradients(() => f(...args), args, dy);
      if (dy != null) {
        util.assertShapesMatch(
            value.shape, dy.shape,
            'The shape of dy passed in grads(f)([x1,...], dy) must match the ' +
                'shape returned by f([x1,...])');
      }
      value.dispose();
      checkGrads(grads);
      return grads;
    };
  }

  /**
   * Like `dl.grad`, but returns also the value of `f()`. Useful when `f()`
   * returns a metric you want to show.
   *
   * The result is a rich object with the following properties:
   * - grad: The gradient of `f(x)` w.r.t `x` (result of `grad`).
   * - value: The value returned by `f(x)`.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static valueAndGrad<I extends Tensor, O extends Tensor>(f: (x: I) => O):
      (x: I, dy?: O) => {
        value: O;
        grad: I;
      } {
    util.assert(
        util.isFunction(f),
        'The f passed in valueAndGrad(f) must be a function');
    return (x: I, dy?: O) => {
      util.assert(
          x instanceof Tensor,
          'The x passed in valueAndGrad(f)(x) must be a tensor');
      util.assert(
          dy == null || dy instanceof Tensor,
          'The dy passed in valueAndGrad(f)(x, dy) must be a tensor');
      const {grads, value} = ENV.engine.gradients(() => f(x), [x], dy);
      checkGrads(grads);
      return {grad: grads[0] as I, value: value as O};
    };
  }

  /**
   * Like `grads`, but returns also the value of `f()`. Useful when `f()`
   * returns a metric you want to show.
   *
   * The result is a rich object with the following properties:
   * - grads: The gradients of `f()` w.r.t each input (result of `grads`).
   * - value: The value returned by `f(x)`.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static valueAndGrads<O extends Tensor>(f: (...args: Tensor[]) => O):
      (args: Tensor[], dy?: O) => {
        grads: Tensor[];
        value: O;
      } {
    util.assert(
        util.isFunction(f),
        'The f passed in valueAndGrads(f) must be a function');
    return (args: Tensor[], dy?: O) => {
      util.assert(
          Array.isArray(args) && args.every(arg => arg instanceof Tensor),
          'The args passed in valueAndGrads(f)(args) must be array of tensors');
      util.assert(
          dy == null || dy instanceof Tensor,
          'The dy passed in valueAndGrads(f)(args, dy) must be a tensor');
      const res = ENV.engine.gradients(() => f(...args), args, dy);
      if (dy != null) {
        util.assertShapesMatch(
            res.value.shape, dy.shape,
            'The shape of dy passed in valueAndGrads(f)([x1,...], dy) must ' +
                'match the shape returned by f([x1,...])');
      }
      checkGrads(res.grads);
      return res;
    };
  }

  /**
   * Computes and returns the gradient of f(x) with respect to the list of
   * trainable variables provided by `varList`. If no list is provided, it
   * defaults to all trainable variables.
   *
   * @param f The function to execute. f() should return a scalar.
   * @param varList An optional list of variables to provide gradients with
   *     respect to. Defaults to all trainable variables.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static variableGrads(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, grads: NamedTensorMap} {
    util.assert(
        util.isFunction(f),
        'The f passed in variableGrads(f) must be a function');
    util.assert(
        varList == null ||
            Array.isArray(varList) && varList.every(v => v instanceof Variable),
        'The varList passed in variableGrads(f, varList) must be an array ' +
            'of variables');
    if (varList == null) {
      // Get all of the trainable variables.
      varList = [];
      for (const varName in ENV.engine.registeredVariables) {
        varList.push(ENV.engine.registeredVariables[varName]);
      }
    }
    // Prune non-trainable variables.
    varList = varList.filter(variable => variable.trainable);
    const allowNoGradients = true;
    const {value, grads} =
        ENV.engine.gradients(f, varList, null, allowNoGradients);

    util.assert(
        grads.some(g => g != null),
        'Cannot find a connection between any variable and the result of the ' +
            'loss function y=f(x). Please make sure the operations that use ' +
            'variables are inside the function f passed to minimize().');
    util.assert(
        value.rank === 0,
        `The f passed in variableGrads(f) must return a scalar, but it ` +
            `returned a rank-${value.rank} tensor`);

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
   * Takes a function
   * `f(...inputs) => {value: Tensor, gradFunc: dy => Tensor[]}` and returns
   * another function `g(...inputs)` which takes the same inputs as `f`. When
   * called, `g` returns `f().value`. In backward mode, custom gradients with
   * respect to each input of `f` are computed using `f().gradFunc`.
   *
   * @param f The function to evaluate in forward mode, which should return
   *     `{value: Tensor, gradFunc: (dy) => Tensor[]}`, where `gradFunc` returns
   *     the custom gradients of `f` with respect to its inputs.
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
        `Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`);
  }
}
