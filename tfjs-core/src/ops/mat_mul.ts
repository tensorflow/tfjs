/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {ENGINE, ForwardFunc} from '../engine';
import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes the dot product of two matrices, A * B. These must be matrices.
 *
 * ```js
 * const a = tf.tensor2d([1, 2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.matMul(b).print();  // or tf.matMul(a, b)
 * ```
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function matMul_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike, transposeA = false,
    transposeB = false): T {
  let $a = convertToTensor(a, 'a', 'matMul');
  let $b = convertToTensor(b, 'b', 'matMul');
  [$a, $b] = makeTypesMatch($a, $b);

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    save([$a, $b]);

    const innerShapeA =
        transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
    const innerShapeB =
        transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];

    const outerShapeA =
        transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
    const outerShapeB =
        transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];

    const outerDimsA = $a.shape.slice(0, -2);
    const outerDimsB = $b.shape.slice(0, -2);
    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);

    const batchDimsCompatible =
        batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;

    util.assert(
        $a.rank >= 2 && $b.rank >= 2 && batchDimsCompatible,
        () =>
            `Error in matMul: the input batch dimensions must either be the ` +
            `same or at least one input batch dimension must be 1. Got input ` +
            `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);

    util.assert(
        innerShapeA === innerShapeB,
        () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of Tensors with shapes ${$a.shape} and ` +
            `${$b.shape} and transposeA=${transposeA}` +
            ` and transposeB=${transposeB} must match.`);

    const outShapeOuterDims = batchDimA > batchDimB ? outerDimsA : outerDimsB;
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

    const a3D = transposeA ?
        reshape($a, [batchDimA, innerShapeA, outerShapeA]) :
        reshape($a, [batchDimA, outerShapeA, innerShapeA]);
    const b3D = transposeB ?
        reshape($b, [batchDimB, outerShapeB, innerShapeB]) :
        reshape($b, [batchDimB, innerShapeB, outerShapeB]);

    const res3d = backend.batchMatMul(
        a3D as Tensor3D, b3D as Tensor3D, transposeA, transposeB);
    return reshape(res3d, outShape);
  };

  const inputs: BatchMatMulInputs = {a: $a, b: $b};
  const attrs: BatchMatMulAttrs = {transposeA, transposeB};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             BatchMatMul, attrs as {} as NamedAttrMap) as T;
}

export const matMul = op({matMul_});
