/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {concat, DataType, keep, reshape, scalar, slice, stack, Tensor, tensor, tidy, unstack} from '@tensorflow/tfjs-core';

import {assertShapesMatchAllowUndefinedSize} from './tensor_utils';

export interface TensorWithState {
  tensor?: Tensor;
  written?: boolean;
  read?: boolean;
  cleared?: boolean;
}
/**
 * The TensorArray object keeps an array of Tensors.  It
 * allows reading from the array and writing to the array.
 */
export class TensorArray {
  private tensors: TensorWithState[] = [];
  private closed_ = false;
  readonly idTensor: Tensor;
  constructor(
      readonly name: string, readonly dtype: DataType, private maxSize: number,
      private elementShape: number[], readonly identicalElementShapes: boolean,
      readonly dynamicSize: boolean, readonly clearAfterRead: boolean) {
    this.idTensor = scalar(0);
    keep(this.idTensor);
  }

  get id() {
    return this.idTensor.id;
  }

  get closed() {
    return this.closed_;
  }

  /**
   * Dispose the tensors and idTensor and mark the TensoryArray as closed.
   */
  clearAndClose(keepIds?: Set<number>) {
    this.tensors.forEach(tensor => {
      if (keepIds == null || !keepIds.has(tensor.tensor.id)) {
        tensor.tensor.dispose();
      }
    });
    this.tensors = [];
    this.closed_ = true;
    this.idTensor.dispose();
  }

  size(): number {
    return this.tensors.length;
  }

  /**
   * Read the value at location index in the TensorArray.
   * @param index Number the index to read from.
   */
  read(index: number): Tensor {
    if (this.closed_) {
      throw new Error(`TensorArray ${this.name} has already been closed.`);
    }

    if (index < 0 || index >= this.size()) {
      throw new Error(`Tried to read from index ${index}, but array size is: ${
          this.size()}`);
    }

    const tensorWithState = this.tensors[index];
    if (tensorWithState.cleared) {
      throw new Error(
          `TensorArray ${this.name}: Could not read index ${
              index} twice because it was cleared after a previous read ` +
          `(perhaps try setting clear_after_read = false?).`);
    }

    if (this.clearAfterRead) {
      tensorWithState.cleared = true;
    }

    tensorWithState.read = true;
    return tensorWithState.tensor;
  }

  /**
   * Helper method to read multiple tensors from the specified indices.
   */
  readMany(indices: number[]): Tensor[] {
    return indices.map(index => this.read(index));
  }

  /**
   * Write value into the index of the TensorArray.
   * @param index number the index to write to.
   * @param tensor
   */
  write(index: number, tensor: Tensor) {
    if (this.closed_) {
      throw new Error(`TensorArray ${this.name} has already been closed.`);
    }

    if (index < 0 || !this.dynamicSize && index >= this.maxSize) {
      throw new Error(`Tried to write to index ${
          index}, but array is not resizeable and size is: ${this.maxSize}`);
    }

    const t = this.tensors[index] || {};

    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray ${
          this.name}: Could not write to TensorArray index ${index},
          because the value dtype is ${
          tensor.dtype}, but TensorArray dtype is ${this.dtype}.`);
    }

    // Set the shape for the first time write to unknow shape tensor array
    if (this.size() === 0 &&
        (this.elementShape == null || this.elementShape.length === 0)) {
      this.elementShape = tensor.shape;
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensor.shape,
        `TensorArray ${this.name}: Could not write to TensorArray index ${
            index}.`);

    if (t.read) {
      throw new Error(
          `TensorArray ${this.name}: Could not write to TensorArray index ${
              index}, because it has already been read.`);
    }

    if (t.written) {
      throw new Error(
          `TensorArray ${this.name}: Could not write to TensorArray index ${
              index}, because it has already been written.`);
    }

    t.tensor = tensor;
    keep(tensor);
    t.written = true;

    this.tensors[index] = t;
  }

  /**
   * Helper method to write multiple tensors to the specified indices.
   */
  writeMany(indices: number[], tensors: Tensor[]) {
    if (indices.length !== tensors.length) {
      throw new Error(
          `TensorArray ${this.name}: could not write multiple tensors,` +
          `because the index size: ${
              indices.length} is not the same as tensors size: ${
              tensors.length}.`);
    }

    indices.forEach((i, index) => this.write(i, tensors[index]));
  }

  /**
   * Return selected values in the TensorArray as a packed Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param [indices] number[] Optional. Taking values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size(). If not specified returns
   *    all tensors in the original order.
   * @param [dtype]
   */
  gather(indices?: number[], dtype?: DataType): Tensor {
    if (!!dtype && dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but gather requested dtype ${dtype}`);
    }

    if (!indices) {
      indices = [];
      for (let i = 0; i < this.size(); i++) {
        indices.push(i);
      }
    } else {
      indices = indices.slice(0, this.size());
    }

    if (indices.length === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    // Read all the PersistentTensors into a vector to keep track of
    // their memory.
    const tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensors[0].shape, 'TensorArray shape mismatch: ');

    return stack(tensors, 0);
  }

  /**
   * Return the values in the TensorArray as a concatenated Tensor.
   */
  concat(dtype?: DataType): Tensor {
    if (!!dtype && dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but concat requested dtype ${dtype}`);
    }

    if (this.size() === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    const indices = [];
    for (let i = 0; i < this.size(); i++) {
      indices.push(i);
    }
    // Collect all the tensors from the tensors array.
    const tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensors[0].shape,
        `TensorArray shape mismatch: tensor array shape (${
            this.elementShape}) vs first tensor shape (${tensors[0].shape})`);

    return concat(tensors, 0);
  }

  /**
   * Scatter the values of a Tensor in specific indices of a TensorArray.
   * @param indices nummber[] values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size().
   * @param tensor Tensor input tensor.
   */
  scatter(indices: number[], tensor: Tensor) {
    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but tensor has dtype ${tensor.dtype}`);
    }

    if (indices.length !== tensor.shape[0]) {
      throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${
          indices.length} vs. ${tensor.shape[0]}`);
    }

    const maxIndex = Math.max(...indices);

    if (!this.dynamicSize && maxIndex >= this.maxSize) {
      throw new Error(
          `Max index must be < array size (${maxIndex}  vs. ${this.maxSize})`);
    }

    this.writeMany(indices, unstack(tensor, 0));
  }

  /**
   * Split the values of a Tensor into the TensorArray.
   * @param length number[] with the lengths to use when splitting value along
   *    its first dimension.
   * @param tensor Tensor, the tensor to split.
   */
  split(length: number[], tensor: Tensor) {
    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but tensor has dtype ${tensor.dtype}`);
    }
    let totalLength = 0;
    const cumulativeLengths = length.map(len => {
      totalLength += len;
      return totalLength;
    });

    if (totalLength !== tensor.shape[0]) {
      throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
    }

    if (!this.dynamicSize && length.length !== this.maxSize) {
      throw new Error(
          `TensorArray's size is not equal to the size of lengths (${
              this.maxSize} vs. ${length.length}), ` +
          'and the TensorArray is not marked as dynamically resizeable');
    }

    const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
    const tensors: Tensor[] = [];
    tidy(() => {
      tensor = reshape(tensor, [1, totalLength, elementPerRow]);
      for (let i = 0; i < length.length; ++i) {
        const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
        const indices = [0, previousLength, 0];
        const sizes = [1, length[i], elementPerRow];
        tensors[i] = reshape(slice(tensor, indices, sizes), this.elementShape);
      }
      return tensors;
    });
    const indices = [];
    for (let i = 0; i < length.length; i++) {
      indices[i] = i;
    }
    this.writeMany(indices, tensors);
  }
}
