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

import {ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import * as broadcast_util from './broadcast_util';
import {where} from './logical_ops';
import {op} from './operation';
import {scalar, zerosLike} from './tensor_ops';
import {neg} from './unary_ops';

/**
 * Adds two `tf.Tensor`s element-wise, A + B. Supports broadcasting.
 *
 * We also expose `tf.addStrict` which has the same signature as this op and
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
 * @param a The first `tf.Tensor` to add.
 * @param b The second `tf.Tensor` to add. Must have the same type as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function add_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'add');
  let $b = convertToTensor(b, 'b', 'add');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);

  const der = (dy: Tensor) => {
    const derA = () => {
      let res = dy;
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($a.shape);
    };
    const derB = () => {
      let res = dy;
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($b.shape);
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc(
             backend => backend.add($a, $b), {a: $a, b: $b}, der, 'Add') as T;
}

/**
 * Adds a list of `tf.Tensor`s element-wise, each with the same shape and dtype.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 *
 * tf.addN([a, b, c]).print();
 * ```
 * @param tensors A list of tensors with the same shape and dtype.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function addN_<T extends Tensor>(tensors: Array<T|TensorLike>): T {
  util.assert(
      Array.isArray(tensors),
      () => 'The argument passed to tf.addN() must be a list of tensors');
  util.assert(
      tensors.length >= 1,
      () => `Must pass at least one tensor to tf.addN(), but got ` +
          `${tensors.length}`);
  const $tensors =
      tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'addN'));
  const firstTensor = $tensors[0];
  $tensors.forEach(t => {
    if (t.dtype !== firstTensor.dtype) {
      throw new Error(
          'All tensors passed to tf.addN() must have the same dtype');
    }
  });
  $tensors.forEach(t => {
    if (!util.arraysEqual(t.shape, firstTensor.shape)) {
      throw new Error(
          'All tensors passed to tf.addN() must have the same shape');
    }
  });

  const der = (dy: T) => {
    const ders: {[key: string]: () => Tensor} = {};
    $tensors.forEach((t, i) => {
      ders[i] = () => dy.clone();
    });
    return ders;
  };
  const inputs: NamedTensorMap = $tensors as {} as NamedTensorMap;
  return ENGINE.runKernelFunc(
      backend => backend.addN($tensors), inputs, der, 'AddN');
}

/**
 * Adds two `tf.Tensor`s element-wise, A + B.
 *
 * Inputs must be the same shape. For broadcasting support, use add() instead.
 *
 * @param a The first Tensor to add element-wise.
 * @param b The second Tensor to add element-wise.
 */
function addStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'addStrict');
  const $b = convertToTensor(b, 'b', 'addStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in addStrict: ');
  return $a.add($b);
}

/**
 * Subtracts two `tf.Tensor`s element-wise, A - B. Supports broadcasting.
 *
 * We also expose `tf.subStrict` which has the same signature as this op and
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
 * @param a The first `tf.Tensor` to subtract from.
 * @param b The second `tf.Tensor` to be subtracted. Must have the same dtype as
 * `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function sub_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'sub');
  let $b = convertToTensor(b, 'b', 'sub');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);

  const der = (dy: Tensor) => {
    const derA = () => {
      let res = dy;
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($a.shape);
    };
    const derB = () => {
      let res = dy;
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.neg().reshape($b.shape);
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc(
             backend => backend.subtract($a, $b), {a: $a, b: $b}, der, 'Sub') as
      T;
}

/**
 * Subtracts two `tf.Tensor`s element-wise, A - B. Inputs must
 * be the same shape.
 *
 * For broadcasting support, use `tf.sub` instead.
 *
 * @param a The first Tensor to subtract element-wise.
 * @param b The second Tensor to subtract element-wise.
 */
function subStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'subStrict');
  const $b = convertToTensor(b, 'b', 'subStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in subStrict: ');
  return $a.sub($b);
}

/**
 * Computes the power of one `tf.Tensor` to another. Supports broadcasting.
 *
 * Given a `tf.Tensor` x and a `tf.Tensor` y, this operation computes x^y for
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
 * @param base The base `tf.Tensor` to pow element-wise.
 * @param exp The exponent `tf.Tensor` to pow element-wise.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function pow_<T extends Tensor>(
    base: Tensor|TensorLike, exp: Tensor|TensorLike): T {
  let $base = convertToTensor(base, 'base', 'pow');
  let $exp = convertToTensor(exp, 'exp', 'pow');
  [$base, $exp] = makeTypesMatch($base, $exp);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($base.shape, $exp.shape);
  const grad = (dy: Tensor, saved: Tensor[]) => {
    const [$base, $exp, y] = saved;
    const derBase = () => {
      const expFloat = $exp.toFloat();
      let res = dy.mul(expFloat.mul($base.pow(expFloat.sub(scalar(1)))));
      const reduceAxes = broadcast_util.getReductionAxes($base.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($base.shape) as T;
    };
    const derExp = () => {
      const condition = $base.greater(0);
      const logBase = $base.log().where(condition, zerosLike($base));
      let res = dy.mul(y.mul(logBase));
      const reduceAxes = broadcast_util.getReductionAxes($exp.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($exp.shape);
    };
    return {a: derBase, b: derExp};
  };

  const attrs = {};
  const inputsToSave = [$base, $exp];
  const outputsToSave = [true];
  return ENGINE.runKernelFunc((backend, save) => {
    const y = backend.pow($base, $exp);
    save([$base, $exp, y]);
    return y;
  }, {a: $base, b: $exp}, grad, 'Pow', attrs, inputsToSave, outputsToSave) as T;
}

/**
 * Computes the power of one `tf.Tensor` to another. Inputs must
 * be the same shape.
 *
 * For broadcasting support, use `tf.pow` instead.
 *
 * @param base The base tensor to pow element-wise.
 * @param exp The exponent tensor to pow element-wise.
 */
function powStrict_<T extends Tensor>(base: T, exp: Tensor): T {
  util.assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
  return base.pow(exp);
}

/**
 * Multiplies two `tf.Tensor`s element-wise, A * B. Supports broadcasting.
 *
 * We also expose `tf.mulStrict` which has the same signature as this op and
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function mul_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'mul');
  let $b = convertToTensor(b, 'b', 'mul');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);

  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const res = dy.mul($b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($a.shape);
      }
      return res;
    };
    const derB = () => {
      const res = dy.mul($a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($b.shape);
      }
      return res;
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.multiply($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'Mul') as T;
}

/**
 * Multiplies two `tf.Tensor`s element-wise, A * B.
 *
 * Inputs must be the same shape. For broadcasting support, use `tf.mul`.
 *
 * @param a The first tensor to multiply.
 * @param b The first tensor to multiply. Must have the same
 *    dtype as `a`.
 */
function mulStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'mul');
  const $b = convertToTensor(b, 'b', 'mul');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in multiplyStrict: ');
  return $a.mul($b);
}

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
 *
 * We also expose `tf.divStrict` which has the same signature as this op and
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function div_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'div');
  let $b = convertToTensor(b, 'b', 'div');
  [$a, $b] = makeTypesMatch($a, $b);

  if ($a.dtype === 'int32' && $b.dtype === 'int32') {
    return floorDiv($a, $b);
  }

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const res = dy.div($b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($a.shape);
      }
      return res;
    };
    const derB = () => {
      let res = dy.mul($a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes).reshape($b.shape);
      }
      const tmp = $b.square();
      return res.div(tmp.toFloat()).neg();
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.realDivide($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'Div') as T;
}

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting. Return 0
 * if denominator is 0.
 *
 * We also expose `tf.divStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 * const c = tf.tensor1d([0, 0, 0, 0]);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 * const c = tf.scalar(0);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function divNoNan_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'div');
  let $b = convertToTensor(b, 'b', 'div');
  [$a, $b] = makeTypesMatch($a, $b);

  const divResult = div($a, $b);
  const zeros = zerosLike(divResult);
  const bEqualsZero = $b.equal(zeros);
  return where(bEqualsZero, zeros, divResult) as T;
}

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function floorDiv_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'floorDiv');
  let $b = convertToTensor(b, 'b', 'floorDiv');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const res = dy.div($b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($a.shape);
      }
      return res;
    };
    const derB = () => {
      let res = dy.mul($a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes).reshape($b.shape);
      }
      const tmp = $b.square();
      return res.div(tmp.toFloat()).neg();
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.floorDiv($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'FloorDiv') as T;
}

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Inputs must
 * be the same shape.
 *
 * @param a The first tensor as the numerator for element-wise division.
 * @param b The second tensor as the denominator for element-wise division.
 */
function divStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'div');
  const $b = convertToTensor(b, 'b', 'div');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in divideStrict: ');
  return $a.div($b);
}

/**
 * Returns the mod of a and b element-wise.
 * `floor(x / y) * y + mod(x, y) = x`
 * Supports broadcasting.
 *
 * We also expose `tf.modStrict` which has the same signature as this op and
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function mod_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'mod');
  let $b = convertToTensor(b, 'b', 'mod');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return dy.sum(reduceAxes).reshape($a.shape);
      }
      return dy;
    };
    const derB = () => {
      const res = dy.mul($a.div($b).floor().neg());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($b.shape);
      }
      return res;
    };
    return {$a: derA, $b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.mod($a, $b);
    save([$a, $b]);
    return res;
  }, {$a, $b}, der) as T;
}

/**
 * Returns the mod of a and b (`a < b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use mod().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function modStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'modStrict');
  const $b = convertToTensor(b, 'b', 'modStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in modStrict: ');
  return $a.mod($b);
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function minimum_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'minimum');
  let $b = convertToTensor(b, 'b', 'minimum');
  [$a, $b] = makeTypesMatch($a, $b);

  if ($a.dtype === 'bool') {
    $a = $a.toInt();
    $b = $b.toInt();
  }

  broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => dy.mul($a.lessEqual($b).toFloat());
    const derB = () => dy.mul($a.greater($b).toFloat());
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.minimum($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'Minimum') as T;
}

/**
 * Returns the min of a and b (`a < b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use minimum().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function minimumStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'minimumStrict');
  const $b = convertToTensor(b, 'b', 'minimumStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in minimumStrict: ');
  return $a.minimum($b);
}

/**
 * Returns the max of a and b (`a > b ? a : b`) element-wise.
 * Supports broadcasting.
 *
 * We also expose `tf.maximumStrict` which has the same signature as this op and
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
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function maximum_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'maximum');
  let $b = convertToTensor(b, 'b', 'maximum');
  [$a, $b] = makeTypesMatch($a, $b);

  if ($a.dtype === 'bool') {
    $a = $a.toInt();
    $b = $b.toInt();
  }

  broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => dy.mul($a.greaterEqual($b).toFloat());
    const derB = () => dy.mul($a.less($b).toFloat());
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.maximum($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'Maximum') as T;
}

/**
 * Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use maximum().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function maximumStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'maximumStrict');
  const $b = convertToTensor(b, 'b', 'maximumStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in maximumStrict: ');
  return $a.maximum($b);
}

/**
 * Returns (a - b) * (a - b) element-wise.
 *
 * Inputs must be the same shape. For broadcasting support, use
 * `tf.squaredDifference` instead.
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 */
function squaredDifferenceStrict_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'squaredDifferenceStrict');
  const $b = convertToTensor(b, 'b', 'squaredDifferenceStrict');
  util.assertShapesMatch(
      $a.shape, $b.shape, 'Error in squaredDifferenceStrict: ');
  return $a.squaredDifference($b);
}

/**
 * Computes arctangent of `tf.Tensor`s a / b element-wise: `atan2(a, b)`.
 * Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
 * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
 *
 * tf.atan2(a, b).print()
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 *
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function atan2_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'atan2');
  let $b = convertToTensor(b, 'b', 'atan2');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);

  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const d = add($a.square(), $b.square());
      let res = dy.mul($b.div(d));
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($a.shape);
    };
    const derB = () => {
      const d = add($a.square(), $b.square());
      let res = neg(dy.mul($a.div(d)));
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = res.sum(reduceAxes);
      }
      return res.reshape($b.shape);
    };
    return {$a: derA, $b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.atan2($a, $b);
    save([$a, $b]);
    return res;
  }, {$a, $b}, der) as T;
}

export const add = op({add_});
export const addN = op({addN_});
export const addStrict = op({addStrict_});
export const atan2 = op({atan2_});
export const div = op({div_});
export const divNoNan = op({divNoNan_});
export const divStrict = op({divStrict_});
export const floorDiv = op({floorDiv_});
export const maximum = op({maximum_});
export const maximumStrict = op({maximumStrict_});
export const minimum = op({minimum_});
export const minimumStrict = op({minimumStrict_});
export const mod = op({mod_});
export const modStrict = op({modStrict_});
export const mul = op({mul_});
export const mulStrict = op({mulStrict_});
export const pow = op({pow_});
export const powStrict = op({powStrict_});
export const squaredDifferenceStrict = op({squaredDifferenceStrict_});
export const sub = op({sub_});
export const subStrict = op({subStrict_});
