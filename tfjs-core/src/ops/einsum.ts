/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Multiply, Reshape, Sum, Transpose} from '../kernel_names';
import {Tensor} from '../tensor';
import {assert} from '../util_base';
import {op} from './operation';

/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Some special cases include:
 *
 * Matrix multiplication:
 * ```js
 * const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tensor2d([[0, 1], [2, 3], [4, 5]]);
 * x.print();
 * y.print():
 * tf.einsum('ij,jk->ik', x, y).print();
 * ```
 *
 * Dot product:
 * ```js
 * const x = tensor1d([1, 2, 3]);
 * const y = tensor1d([0, 1, 2]);
 * x.print():
 * y.print();
 * tf.einsum('i,i->', x, y).print();
 * ```
 *
 * Batch dot product:
 * ```js
 * const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tensor2d([[0, 1, 2], [3, 4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('bi,bi->b', x, y).print();
 * ```
 *
 * Outer prouduct:
 * ```js
 * const x = tensor1d([1, 3, 5]);
 * const y = tensor1d([2, 4, 6]);
 * x.print();
 * y.print();
 * tf.einsum('i,j->ij', x, y).print();
 * ```
 *
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not suppoted.
 * - For two or more input tensors, up to only one summation axis is supported.
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 * [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
export function einsum_(equation: string, ...tensors: Tensor[]): Tensor {
  equation = equation.replace(/\s/g, '');  // Remove witespace in equation.
  const indexArrow = equation.indexOf('->');
  if (indexArrow === -1) {
    throw new Error('Equations without an arrow is not supported');
  }
  const [inputString, outputString] = equation.split('->');
  const inputTerms = inputString.split(',');
  const numInputs = inputTerms.length;
  if (tensors.length !== numInputs) {
    throw new Error(
        `Expected ${numInputs} input tensors, received ${tensors.length}`);
  }
  if (numInputs > 2) {
    throw new Error(
        'Support for more than 2 input tensors is not implemented yet.');
  }

  const allDims: string[] = [];
  for (let i = 0; i < outputString.length; ++i) {
    const dimName = outputString[i];
    if (!inputTerms.some(inputTerm => inputTerm.indexOf(dimName) !== -1)) {
      throw new Error(
          `Output subscripts contain the label ${dimName} ` +
          `not present in the input subscripts.`);
    }
    if (allDims.indexOf(dimName) === -1) {
      allDims.push(dimName);
    }
  }
  for (let i = 0; i < inputString.length; ++i) {
    const dimName = inputString[i];
    if (allDims.indexOf(dimName) === -1 && dimName !== ',') {
      allDims.push(dimName);
    }
  }

  const idDims: number[][] = new Array<number[]>(inputTerms.length);
  for (let i = 0; i < numInputs; ++i) {
    if (new Set(inputTerms[i].split('')).size !== inputTerms[i].length) {
      throw new Error(
          `Found duplicate axes in input component ${inputTerms[i]}. ` +
          `Support for duplicate axes in input is not implemented yet.`);
    }
    if (inputTerms.indexOf('...') !== -1) {
      throw new Error('The notation "..." is not supported yet.');
    }
    idDims[i] = [];
    for (let j = 0; j < inputTerms[i].length; ++j) {
      idDims[i].push(allDims.indexOf(inputTerms[i][j]));
    }
  }

  const numDims = allDims.length;          // Number of unique dimensions.
  const numOutDims = outputString.length;  // Number of output dimensions.
  const summedDims: number[] = [];         // Dimensions being summed over.
  for (let i = numOutDims; i < numDims; ++i) {
    summedDims.push(i);
  }

  if (numInputs > 1 && summedDims.length > 1) {
    throw new Error(
        'Summation over >1 axes is not implemented for ' +
        '>1 input tensors yet.');
  }

  checkDimSizes(allDims.length, idDims, tensors);
  const nSteps = summedDims.length + 1;

  const {path, steps} = getComputePath(summedDims, idDims);

  let out: Tensor|null = null;
  let numDimsRemaining = allDims.length;
  for (let i = 0; i < nSteps; ++i) {
    for (const idTerm of steps[i]) {
      const {permutationIndices, expandDims: dimsToExpand} =
          getPermutation(numDimsRemaining, idDims[idTerm]);
      let x = ENGINE.runKernel(
                  Transpose, {x: tensors[idTerm]},
                  {perm: permutationIndices}) as Tensor;
      const targetShape: number[] = x.shape;
      for (let k = 0; k < dimsToExpand.length; ++k) {
        targetShape.splice(dimsToExpand[k], 0, 1);
      }

      x = ENGINE.runKernel(Reshape, {x}, {shape: targetShape});
      out = out === null ? x : ENGINE.runKernel(Multiply, {a: out, b: x});
    }
    if (i < nSteps - 1) {
      if (path[i] >= 0) {
        out = ENGINE.runKernel(
            Sum, {x: out},
            {axis: path[i] < out.shape.length ? path[i] : undefined});
      }
      numDimsRemaining--;
    }
  }
  return out;
}

function getPermutation(nDims: number, idDims: number[]):
    {permutationIndices: number[], expandDims: number[]} {
  let permutationIndices: number[] = new Array<number>(nDims);
  permutationIndices.fill(-1);
  for (let i = 0; i < idDims.length; ++i) {
    permutationIndices[idDims[i]] = i;
  }
  const expandDims: number[] = [];
  for (let i = 0; i < nDims; ++i) {
    if (permutationIndices[i] === -1) {
      expandDims.push(i);
    }
  }
  permutationIndices = permutationIndices.filter(d => d !== -1);
  return {permutationIndices, expandDims};
}

/**
 * Checks that the dimension sizes from different input tensors match the
 * equation.
 */
function checkDimSizes(nDims: number, idDims: number[][], tensors: Tensor[]) {
  const dimSizes: number[] = new Array<number>(nDims);
  for (let i = 0; i < tensors.length; ++i) {
    const shape: number[] = tensors[i].shape;
    for (let j = 0; j < idDims[i].length; ++j) {
      if (dimSizes[idDims[i][j]] === undefined) {
        dimSizes[idDims[i][j]] = shape[j];
      } else {
        assert(
            dimSizes[idDims[i][j]] === shape[j],
            () => `Expected dimension ${dimSizes[idDims[i][j]]} at axis ${j} ` +
                `of input shaped ${JSON.stringify(shape)}, ` +
                `but got dimension ${shape[j]}`);
      }
    }
  }
}

/**
 * Gets path of computation.
 *
 * @param summedDims indices to the dimensions being summed over.
 * @param idDims A look up table for the dimensions present in each input
 *     tensor. Each consituent array contains indices for the dimensions in the
 *     corresponding input tensor.
 *
 * @return A map with two fields:
 *   - path: The path of computation, with each element indicating the dimension
 *     being summed over after the element-wise multiplication in that step.
 *   - steps: With the same length as `path`. Each element contains the indices
 *     to the input tensors being used for element-wise multiplication in the
 *     corresponding step.
 */
function getComputePath(summedDims: number[], idDims: number[][]):
    {path: number[], steps: number[][]} {
  const path: number[] = summedDims;
  const steps: number[][] = [];
  let nSteps = 0;
  if (summedDims.length === 0) {
    // Einsum that involes no summing: e.g., transpose and outer product.
    path.push(-1);
    nSteps = 1;
  } else {
    nSteps = summedDims.length + 1;
  }
  for (let i = 0; i < nSteps; ++i) {
    steps.push([]);
  }
  const computedTermIndices: number[] = [];
  for (let i = 0; i < path.length; ++i) {
    const summedDim = path[i];
    const termIndices = findTermsWithDim(idDims, summedDim);
    for (const termIndex of termIndices) {
      if (computedTermIndices.indexOf(termIndex) === -1) {
        steps[i].push(termIndex);
        computedTermIndices.push(termIndex);
      }
    }
  }
  return {path, steps};
}

function findTermsWithDim(idDims: number[][], dim: number): number[] {
  const termIndices: number[] = [];
  for (let i = 0; i < idDims.length; ++i) {
    if (idDims[i].indexOf(dim) !== -1 || dim === -1) {
      termIndices.push(i);
    }
  }
  return termIndices;
}

export const einsum = op({einsum_});
