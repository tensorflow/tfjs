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

import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor} from '../tensor';
import {upcastType} from '../types';
import * as util from '../util';

import * as broadcast_util from './broadcast_util';
import {operation} from './operation';
import {neg, scalar, square} from './ops';
import {KernelBackend} from '../kernels/backend';

export class BinaryOps {
  /**
   * Adds two `Tensor`s element-wise, A + B. Supports broadcasting.
   *
   * We also expose `addStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.tensor1d([10, 20, 30, 40]);
   *
   * a.add(b).print();  // or tf.add(a, b)
   * ```
   *
   * ```js
   * // Broadcast add a with b.
   * const a = tf.scalar(5);
   * const b = tf.tensor1d([10, 20, 30, 40]);
   *
   * a.add(b).print();  // or tf.add(a, b)
   * ```
   * @param a The first `Tensor` to add.
   * @param b The second `Tensor` to add. Must have the same type as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static add<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'add');
    util.assertTypesMatch(a, b);

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: Tensor) => {
      const derA = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(a.shape);
      };
      const derB = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(b.shape);
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(backend => backend.add(a, b), {a, b}, der) as T;
  }

  /**
   * Adds two `Tensor`s element-wise, A + B.
   *
   * Inputs must be the same shape. For broadcasting support, use add() instead.
   *
   * @param a The first Tensor to add element-wise.
   * @param b The second Tensor to add element-wise.
   */
  @operation
  static addStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
    return a.add(b);
  }

  /**
   * Subtracts two `Tensor`s element-wise, A - B. Supports broadcasting.
   *
   * We also expose `subStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([10, 20, 30, 40]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.sub(b).print();  // or tf.sub(a, b)
   * ```
   *
   * ```js
   * // Broadcast subtract a with b.
   * const a = tf.tensor1d([10, 20, 30, 40]);
   * const b = tf.scalar(5);
   *
   * a.sub(b).print();  // or tf.sub(a, b)
   * ```
   * @param a The first `Tensor` to subtract from.
   * @param b The second `Tensor` to be subtracted. Must have the same dtype as
   * `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static sub<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'sub');
    util.assertTypesMatch(a, b);

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: Tensor) => {
      const derA = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(a.shape);
      };
      const derB = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.neg().reshape(b.shape);
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(
               backend => backend.subtract(a, b), {a, b}, der) as T;
  }

  /**
   * Subtracts two `Tensor`s element-wise, A - B. Inputs must
   * be the same shape.
   *
   * For broadcasting support, use sub() instead.
   *
   * @param a The first Tensor to subtract element-wise.
   * @param b The second Tensor to subtract element-wise.
   */
  @operation
  static subStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
    return a.sub(b);
  }

  /**
   * Computes the power of one `Tensor` to another. Supports broadcasting.
   *
   * Given a `Tensor` x and a `Tensor` y, this operation computes x^y for
   * corresponding elements in x and y. The result's dtype will be the upcasted
   * type of the `base` and `exp` dtypes.
   *
   * ```js
   * const a = tf.tensor([[2, 3], [4, 5]])
   * const b = tf.tensor([[1, 2], [3, 0]]).toInt();
   *
   * a.pow(b).print();  // or tf.pow(a, b)
   * ```
   *
   * ```js
   * const a = tf.tensor([[1, 2], [3, 4]])
   * const b = tf.tensor(2).toInt();
   *
   * a.pow(b).print();  // or tf.pow(a, b)
   * ```
   * We also expose `powStrict` which has the same signature as this op and
   * asserts that `base` and `exp` are the same shape (does not broadcast).
   *
   * @param base The base `Tensor` to pow element-wise.
   * @param exp The exponent `Tensor` to pow element-wise.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static pow<T extends Tensor>(base: T, exp: Tensor): T {
    util.assertArgumentsAreTensors({base, exp}, 'pow');

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(base.shape, exp.shape);
    base = base.cast(upcastType(base.dtype, exp.dtype));
    exp = exp.cast(upcastType(base.dtype, exp.dtype));
    const grad = (dy: Tensor, saved: Tensor[]) => {
      const [y] = saved;
      const derBase = () => {
        let res = dy.mul(exp.toFloat().mul(y.div(base)));
        const reduceAxes =
            broadcast_util.getReductionAxes(base.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(base.shape) as T;
      };
      const derExp = () => {
        let res = dy.mul(y.mul(base.log()).toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(exp.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(exp.shape);
      };
      return {base: derBase, exp: derExp};
    };
    return ENV.engine.runKernel(
               (backend, save) => save(backend.pow(base, exp)), {base, exp},
               grad) as T;
  }

  /**
   * Computes the power of one `Tensor` to another. Inputs must
   * be the same shape.
   *
   * For broadcasting support, use pow() instead.
   *
   * @param base The base tensor to pow element-wise.
   * @param exp The exponent tensor to pow element-wise.
   */
  @operation
  static powStrict<T extends Tensor>(base: T, exp: Tensor): T {
    util.assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
    return base.pow(exp);
  }

  /**
   * Multiplies two `Tensor`s element-wise, A * B. Supports broadcasting.
   *
   * We also expose `mulStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.tensor1d([2, 3, 4, 5]);
   *
   * a.mul(b).print();  // or tf.mul(a, b)
   * ```
   *
   * ```js
   * // Broadcast mul a with b.
   * const a = tf.tensor1d([1, 2, 3, 4]);
   * const b = tf.scalar(5);
   *
   * a.mul(b).print();  // or tf.mul(a, b)
   * ```
   * @param a The first tensor to multiply.
   * @param b The second tensor to multiply. Must have the same dtype as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static mul<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'mul');
    util.assertTypesMatch(a, b);

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: Tensor) => {
      const derA = () => {
        const res = dy.mul(b.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(a.shape);
        }
        return res;
      };
      const derB = () => {
        const res = dy.mul(a.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(b.shape);
        }
        return res;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(
               backend => backend.multiply(a, b), {a, b}, der) as T;
  }

  /**
   * Multiplies two `Tensor`s element-wise, A * B.
   *
   * Inputs must be the same shape. For broadcasting support, use mul().
   *
   * @param a The first tensor to multiply.
   * @param b The first tensor to multiply. Must have the same
   *    dtype as `a`.
   */
  @operation
  static mulStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
    return a.mul(b) as T;
  }

  /**
   * Divides two `Tensor`s element-wise, A / B. Supports broadcasting.
   *
   * We also expose `divStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 9, 16]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.div(b).print();  // or tf.div(a, b)
   * ```
   *
   * ```js
   * // Broadcast div a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(2);
   *
   * a.div(b).print();  // or tf.div(a, b)
   * ```
   *
   * @param a The first tensor as the numerator.
   * @param b The second tensor as the denominator. Must have the same dtype as
   * `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static div<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'div');
    util.assertTypesMatch(a, b);

    let forwardFunc: (backend: KernelBackend) => Tensor;
    if (a.dtype === 'int32' && b.dtype === 'int32') {
      return BinaryOps.floorDiv(a, b);
    } else {
      forwardFunc = (backend: KernelBackend) => backend.realDivide(a, b);
    }

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const derA = () => {
        const res = dy.div(b.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(a.shape);
        }
        return res;
      };
      const derB = () => {
        let res = dy.mul(a.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes).reshape(b.shape);
        }
        const tmp = b.square() as Tensor;
        return res.div(tmp.toFloat()).neg() as Tensor;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(forwardFunc, {a, b}, der) as
        T;
  }

  /**
   * Divides two `Tensor`s element-wise, A / B. Supports broadcasting.
   * The result is rounded with floor function.
   *
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 9, 16]);
   * const b = tf.tensor1d([1, 2, 3, 4]);
   *
   * a.floorDiv(b).print();  // or tf.div(a, b)
   * ```
   *
   * ```js
   * // Broadcast div a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(2);
   *
   * a.floorDiv(b).print();  // or tf.floorDiv(a, b)
   * ```
   *
   * @param a The first tensor as the numerator.
   * @param b The second tensor as the denominator. Must have the same dtype as
   * `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static floorDiv<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'floorDiv');
    util.assertTypesMatch(a, b);

    const forwardFunc = (backend: KernelBackend) => backend.floorDiv(a, b);
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const derA = () => {
        const res = dy.div(b.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(a.shape);
        }
        return res;
      };
      const derB = () => {
        let res = dy.mul(a.toFloat());
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes).reshape(b.shape);
        }
        const tmp = b.square() as Tensor;
        return res.div(tmp.toFloat()).neg() as Tensor;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(forwardFunc, {a, b}, der) as
        T;
  }

  /**
   * Divides two `Tensor`s element-wise, A / B. Inputs must
   * be the same shape.
   *
   * @param a The first tensor as the numerator for element-wise division.
   * @param b The second tensor as the denominator for element-wise division.
   */
  @operation
  static divStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
    return a.div(b) as T;
  }

  /**
   * Returns the mod of a and b element-wise.
   * `floor(x / y) * y + mod(x, y) = x`
   * Supports broadcasting.
   *
   * We also expose `modStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.mod(b).print();  // or tf.mod(a, b)
   * ```
   *
   * ```js
   * // Broadcast a mod b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.mod(b).print();  // or tf.mod(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static mod<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'mod');
    util.assertTypesMatch(a, b);

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const derA = () => {
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return dy.sum(reduceAxes).reshape(a.shape);
        }
        return dy;
      };
      const derB = () => {
        const res = dy.mul(a.div(b).floor().neg());
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(b.shape);
        }
        return res;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(backend => backend.mod(a, b), {a, b}, der) as T;
  }

  /**
   * Returns the mod of a and b (`a < b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use mod().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  @operation
  static modStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in modStrict: ');
    return a.mod(b);
  }

  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * We also expose `minimumStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.minimum(b).print();  // or tf.minimum(a, b)
   * ```
   *
   * ```js
   * // Broadcast minimum a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.minimum(b).print();  // or tf.minimum(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static minimum<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'minimum');
    util.assertTypesMatch(a, b);

    if (a.dtype === 'bool') {
      a = a.toInt();
    }
    if (b.dtype === 'bool') {
      b = b.toInt();
    }
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const derA = () => dy.mul(a.lessEqual(b).toFloat());
      const derB = () => dy.mul(a.greater(b).toFloat());
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(
               backend => backend.minimum(a, b), {a, b}, der) as T;
  }

  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use minimum().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  @operation
  static minimumStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in minimumStrict: ');
    return a.minimum(b);
  }

  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * We also expose `maximumStrict` which has the same signature as this op and
   * asserts that `a` and `b` are the same shape (does not broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.maximum(b).print();  // or tf.maximum(a, b)
   * ```
   *
   * ```js
   * // Broadcast maximum a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.maximum(b).print();  // or tf.maximum(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static maximum<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'maximum');
    util.assertTypesMatch(a, b);

    if (a.dtype === 'bool') {
      a = a.toInt();
    }
    if (b.dtype === 'bool') {
      b = b.toInt();
    }
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const derA = () => dy.mul(a.greaterEqual(b).toFloat());
      const derB = () => dy.mul(a.less(b).toFloat());
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(
               backend => backend.maximum(a, b), {a, b}, der) as T;
  }

  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use maximum().
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   */
  @operation
  static maximumStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, 'Error in minimumStrict: ');
    return a.maximum(b);
  }

  /**
   * Returns (a - b) * (a - b) element-wise.
   * Supports broadcasting.
   *
   * We also expose `squaredDifferenceStrict` which has the same signature as
   * this op and asserts that `a` and `b` are the same shape (does not
   * broadcast).
   *
   * ```js
   * const a = tf.tensor1d([1, 4, 3, 16]);
   * const b = tf.tensor1d([1, 2, 9, 4]);
   *
   * a.squaredDifference(b).print();  // or tf.squaredDifference(a, b)
   * ```
   *
   * ```js
   * // Broadcast squared difference  a with b.
   * const a = tf.tensor1d([2, 4, 6, 8]);
   * const b = tf.scalar(5);
   *
   * a.squaredDifference(b).print();  // or tf.squaredDifference(a, b)
   * ```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  @doc({heading: 'Operations', subheading: 'Arithmetic'})
  @operation
  static squaredDifference<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'squaredDifference');
    util.assertTypesMatch(a, b);

    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: Tensor) => {
      const two = scalar(2);
      const derA = () => dy.mul(a.sub(b).mul(two));
      const derB = () => dy.mul(b.sub(a).mul(two));
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(
               backend => backend.squaredDifference(a, b), {a, b}, der) as T;
  }

  /**
   * Returns (a - b) * (a - b) element-wise.
   *
   * Inputs must be the same shape. For broadcasting support, use
   * squaredDifference() instead.
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same type as `a`.
   */
  @operation
  static squaredDifferenceStrict<T extends Tensor>(a: T, b: T): T {
    util.assertShapesMatch(
        a.shape, b.shape, 'Error in squaredDifferenceStrict: ');
    return a.squaredDifference(b);
  }

  /*
   * Computes arctangent of `Tensor`s a / b element-wise: `atan2(a, b)`.
   * Supports broadcasting.
   *
   * ```js
   * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
   * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
   *
   * tf.atan2(a, b).print()
   *```
   *
   * @param a The first tensor.
   * @param b The second tensor. Must have the same dtype as `a`.
   *
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static atan2<T extends Tensor>(a: Tensor, b: Tensor): T {
    util.assertArgumentsAreTensors({a, b}, 'atan2');
    util.assertTypesMatch(a, b);

    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: Tensor) => {
      const derA = () => {
        const d = BinaryOps.add(square(a), square(b));
        let res = dy.mul(b.div(d));
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(a.shape);
      };
      const derB = () => {
        const d = BinaryOps.add(square(a), square(b)) as T;
        let res = neg(dy.mul(a.div(d)));
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(b.shape);
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.runKernel(backend => backend.atan2(a, b), {a, b}, der) as
        T;
  }
}
