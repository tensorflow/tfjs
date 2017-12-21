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

import {ENV} from '../environment';
import {NDArrayMath} from '../math/math';
import {Array1D, Scalar} from '../math/ndarray';
import {InCPUMemoryShuffledInputProviderBuilder} from './input_provider';

describe('InCPUMemoryShuffledInputProviderBuilder', () => {
  let math: NDArrayMath;

  beforeEach(() => {
    const safeMode = false;
    math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);
  });

  afterEach(() => {
    ENV.reset();
  });

  it('ensure inputs stay in sync', () => {
    const x1s = [Scalar.new(1), Scalar.new(2), Scalar.new(3)];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    const shuffledInputProvider =
        new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);

    const [x1provider, x2provider] = shuffledInputProvider.getInputProviders();

    const seenNumbers: {[key: number]: boolean} = {};
    for (let i = 0; i < x1s.length; i++) {
      const x1 = x1provider.getNextCopy(math);
      const x2 = x2provider.getNextCopy(math);

      expect(x1.get() * 10).toEqual(x2.get());

      seenNumbers[x1.get()] = true;
      seenNumbers[x2.get()] = true;
    }

    // Values are shuffled, make sure we've seen everything.
    const expectedSeenNumbers = [1, 2, 3, 10, 20, 30];
    for (let i = 0; i < expectedSeenNumbers.length; i++) {
      expect(seenNumbers[expectedSeenNumbers[i]]).toEqual(true);
    }
  });

  it('different number of examples', () => {
    const x1s = [Scalar.new(1), Scalar.new(2)];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();
  });

  it('different shapes within input', () => {
    const x1s = [Scalar.new(1), Array1D.new([1, 2])];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();
  });
});

describe('InGPUMemoryShuffledInputProviderBuilder', () => {
  let math: NDArrayMath;

  beforeEach(() => {
    const safeMode = false;
    math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);
  });

  afterEach(() => {
    ENV.reset();
  });

  it('ensure inputs stay in sync', () => {
    const x1s = [Scalar.new(1), Scalar.new(2), Scalar.new(3)];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    const shuffledInputProvider =
        new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);

    const [x1provider, x2provider] = shuffledInputProvider.getInputProviders();

    const seenNumbers: {[key: number]: boolean} = {};
    for (let i = 0; i < x1s.length; i++) {
      const x1 = x1provider.getNextCopy(math);
      const x2 = x2provider.getNextCopy(math);

      expect(x1.get() * 10).toEqual(x2.get());

      seenNumbers[x1.get()] = true;
      seenNumbers[x2.get()] = true;

      x1provider.disposeCopy(math, x1);
      x2provider.disposeCopy(math, x1);
    }

    // Values are shuffled, make sure we've seen everything.
    const expectedSeenNumbers = [1, 2, 3, 10, 20, 30];
    for (let i = 0; i < expectedSeenNumbers.length; i++) {
      expect(seenNumbers[expectedSeenNumbers[i]]).toEqual(true);
    }
  });

  it('different number of examples', () => {
    const x1s = [Scalar.new(1), Scalar.new(2)];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();

    x1s.forEach(x1 => {
      x1.dispose();
    });
    x2s.forEach(x2 => {
      x2.dispose();
    });
  });

  it('different shapes within input', () => {
    const x1s = [Scalar.new(1), Array1D.new([1, 2])];
    const x2s = [Scalar.new(10), Scalar.new(20), Scalar.new(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();

    x1s.forEach(x1 => {
      x1.dispose();
    });
    x2s.forEach(x2 => {
      x2.dispose();
    });
  });
});
