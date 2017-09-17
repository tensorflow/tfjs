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

import {NDArrayMath} from '../math/math';
import {NDArray} from '../math/ndarray';
import * as util from '../util';

/**
 * The interface for input providers.
 */
export interface InputProvider {
  /**
   * Get the next input as a copy. This is important because the data might
   * get uploaded to the GPU and modify the original data.
   * @param math NDArrayMath
   */
  getNextCopy(math: NDArrayMath): NDArray;
  /**
   * Dispose the input copy.
   * @param math NDArrayMath
   * @param copy The copy provided from getNextCopy
   */
  disposeCopy(math: NDArrayMath, copy: NDArray): void;
}

/**
 * A common interface for shuffled input provider builders. This returns
 * InputProviders that are synchronized.
 * @hidden
 */
export interface ShuffledInputProviderBuilder {
  getInputProviders(): InputProvider[];
}

/**
 * @hidden
 */
export abstract class InMemoryShuffledInputProviderBuilder implements
    ShuffledInputProviderBuilder {
  protected shuffledIndices: Uint32Array;
  protected numInputs: number;

  protected idx = 0;
  // Counter for how many times the current index has been called. Resets to 0
  // when it reaches the number of inputs.
  protected inputCounter = 0;
  protected epoch = 0;

  /**
   * Constructs an `InMemoryShuffledInputProvider`. All of the inputs must be
   * in memory.
   * @param inputs All of the inputs, size: [number of inputs][number of
   * examples].
   */
  constructor(protected inputs: NDArray[][]) {
    this.shuffledIndices = util.createShuffledIndices(inputs[0].length);
    this.numInputs = inputs.length;

    // Make sure the number of examples in each input matches.
    const numExamples = this.inputs[0].length;
    for (let i = 0; i < this.numInputs; i++) {
      util.assert(
          this.inputs[i].length === numExamples,
          'Number of examples must match across different inputs.');
    }

    // Make sure the shapes within inputs all match.
    for (let i = 0; i < this.numInputs; i++) {
      const inputShape = this.inputs[i][0].shape;
      for (let j = 0; j < this.inputs[i].length; j++) {
        util.assertShapesMatch(inputShape, this.inputs[i][j].shape);
      }
    }
  }

  protected getCurrentExampleIndex(): number {
    const returnIdx = this.idx;

    this.inputCounter++;
    if (this.inputCounter >= this.numInputs) {
      this.idx++;
      this.inputCounter = 0;

      if (this.idx >= this.inputs[0].length) {
        this.idx = 0;
        this.epoch++;
      }
    }
    return returnIdx;
  }

  protected getNextInput(inputId: number): NDArray {
    const currentExampleIndex = this.getCurrentExampleIndex();

    return this.inputs[inputId][this.shuffledIndices[currentExampleIndex]];
  }

  getEpoch() {
    return this.epoch;
  }

  /**
   * Returns input providers which shuffle the inputs and stay in sync.
   */
  getInputProviders(): InputProvider[] {
    const inputProviders: InputProvider[] = [];

    for (let i = 0; i < this.numInputs; i++) {
      inputProviders.push(this.getInputProvider(i));
    }
    return inputProviders;
  }

  abstract getInputProvider(inputId: number): InputProvider;
}

/**
 * An in CPU memory ShuffledInputProviderBuilder that shuffles NDArrays on the
 * CPU and keeps them mutually in sync.
 */
export class InCPUMemoryShuffledInputProviderBuilder extends
    InMemoryShuffledInputProviderBuilder {
  getInputProvider(inputId: number) {
    const shuffledInputProvider = this;

    return {
      getNextCopy(math: NDArrayMath): NDArray {
        return NDArray.like(shuffledInputProvider.getNextInput(inputId));
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }
}

/**
 * An in GPU memory ShuffledInputProviderBuilder that shuffles NDArrays on the
 * GPU and keeps them mutually in sync. This is more performant than the CPU
 * version as textures will stay in memory, however this is more GPU memory
 * intensive as it keeps textures resident in GPU memory.
 */
export class InGPUMemoryShuffledInputProviderBuilder extends
    InMemoryShuffledInputProviderBuilder {
  getInputProvider(inputId: number) {
    const shuffledInputProvider = this;

    return {
      getNextCopy(math: NDArrayMath): NDArray {
        return math.clone(shuffledInputProvider.getNextInput(inputId));
      },
      disposeCopy(math: NDArrayMath, copy: NDArray) {
        copy.dispose();
      }
    };
  }
}
