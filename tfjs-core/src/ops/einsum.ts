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

import {ENGINE} from '../engine';
import {Multiply, Reshape, Sum, Transpose} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {assert} from '../util_base';
import {op} from './operation';

/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Limitations:
 * This implementation of einsum has the following limitations:
 *
 * - For two or more input tensors, up to only one summation axes is supported.
 * - Does not support >2 input tensors
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 *     [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
export function einsum_(equation: string, ...tensors: Tensor[]): Tensor {
  equation = equation.replace(/\s/g, '');
  const indexArrow = equation.indexOf('->');
  if (indexArrow === -1) {
    throw new Error('Equations without an arrow is not supported');
  }
  const [inputString, outputString] = equation.split('->');
  console.log(
      `inputString = ${inputString}; outputString = ${outputString}`);  // DEBUG
  const inputTerms = inputString.split(',');
  const numInputs = inputTerms.length;
  if (tensors.length != numInputs) {
    throw new Error(
        `Expected ${numInputs} input tensors, received ${tensors.length}`);
  }
  if (numInputs > 2) {
    throw new Error(
        'Support for more than 2 input tensors is not implemented yet.')
  }

  const allDims: string[] = [];
  for (let i = 0; i < outputString.length; ++i) {
    const dimName = outputString[i];
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
  console.log(`allDims = ${JSON.stringify(allDims)}`);  // DEBUG

  const idDims: number[][] = new Array<number[]>(inputTerms.length);
  for (let i = 0; i < numInputs; ++i) {
    if (new Set(inputTerms[i].split('')).size != inputTerms[i].length) {
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
  console.log(`idDims = ${JSON.stringify(idDims)}`);  // DEBUG

  // TOOD(cais): Throw error for duplicate dimensions in the same term.

  const numDims = allDims.length;          // Number of unique dimensions.
  const numOutDims = outputString.length;  // Number of output dimensions.
  const summedDims: number[] = [];         // Summed dimensions.
  for (let i = numOutDims; i < numDims; ++i) {
    summedDims.push(i);
  }

  if (numInputs > 1 && summedDims.length > 1) {
    throw new Error(
        'Summation over >1 axes is not implemented for ' +
        '>1 input tensors yet.');
  }

  console.log(`summedDims = ${JSON.stringify(summedDims)}`);  // DEBUG
  const dimSizes = computeDimSizes(allDims.length, idDims, tensors);
  console.log(`dimSizes = ${JSON.stringify(dimSizes)}`);  // DEBUG
  const nSteps = summedDims.length + 1;

  const {path, ops} = getComputePath(summedDims, idDims);
  console.log(`path = ${JSON.stringify(path)}`);  // DEBUG
  console.log(`ops = ${JSON.stringify(ops)}`);    // DEBUG

  let out: Tensor|null = null;
  let numDimsRemaining = allDims.length;
  for (let i = 0; i < nSteps; ++i) {
    for (const idTerm of ops[i]) {
      const {permutationIndices, expandDims: dimsToExpand} =
          getPermutation(numDimsRemaining, idDims[idTerm]);
      console.log(
          `permIndices = ${JSON.stringify(permutationIndices)}`);  // DEBUG
      let x = ENGINE.runKernel(
                  Transpose, {x: tensors[idTerm]} as {} as NamedTensorMap,
                  {perm: permutationIndices} as {} as NamedAttrMap) as Tensor;
      console.log(`expandDims = ${JSON.stringify(dimsToExpand)}`);  // DEBUG
      const targetShape: number[] = x.shape;
      console.log(`A targetShape = ${JSON.stringify(targetShape)}`);  // DEBUG
      for (let k = 0; k < dimsToExpand.length; ++k) {
        // if (dimsToExpand[k] < targetShape.length) {
        targetShape.splice(dimsToExpand[k], 0, 1);
        // }
      }
      console.log(`B targetShape = ${JSON.stringify(targetShape)}`);  // DEBUG

      x = ENGINE.runKernel(Reshape, {x}, {shape: targetShape});
      if (out === null) {
        out = x;
      } else {
        out = ENGINE.runKernel(Multiply, {a: out, b: x});
      }
    }
    if (i < nSteps - 1) {
      console.log(`Before sum(): path = ${path[i]}`);     // DEBUG
      console.log(`Before sum(): shape = ${out.shape}`);  // DEBUG
      if (path[i] >= 0) {
        out = ENGINE.runKernel(
            Sum, {x: out},
            {axis: path[i] < out.shape.length ? path[i] : undefined});
      }
      console.log(`After sum(): shape = ${out.shape}`);  // DEBUG
      numDimsRemaining--;
    }
  }
  return out;  // TOOD(cais): Fill in.
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

/** Computes the size of all dimensions. */
function computeDimSizes(
    nDims: number, idDims: number[][], tensors: Tensor[]): number[] {
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
  return dimSizes;
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
 *   - ops: With the same length as `path`. Each element contains the indices
 *     to the input tensors being used for element-wise multiplication in the
 *     corresponding step.
 */
function getComputePath(summedDims: number[], idDims: number[][]):
    {path: number[], ops: number[][]} {
  const path: number[] = summedDims;
  const ops: number[][] = [];
  let nOps: number = 0;
  if (summedDims.length === 0) {
    // Einsum that involes no summing: e.g., transpose and outer product.
    path.push(-1);
    nOps = 1;
  } else {
    nOps = summedDims.length + 1;
  }
  for (let i = 0; i < nOps; ++i) {
    ops.push([]);
  }
  const computedTermIndices: number[] = [];
  for (let i = 0; i < path.length; ++i) {
    const summedDim = path[i];
    const termIndices = findTermsWithDim(idDims, summedDim);
    for (const termIndex of termIndices) {
      if (computedTermIndices.indexOf(termIndex) === -1) {
        ops[i].push(termIndex);
        computedTermIndices.push(termIndex);
      }
    }
  }
  return {path, ops};
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
