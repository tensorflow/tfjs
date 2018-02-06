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
import * as dl from '../index';
import {InCPUMemoryShuffledInputProviderBuilder} from './input_provider';

describe('InCPUMemoryShuffledInputProviderBuilder', () => {
  beforeEach(() => {
    dl.setBackend('cpu');
  });

  afterEach(() => {
    ENV.reset();
  });

  it('ensure inputs stay in sync', () => {
    const x1s = [dl.scalar(1), dl.scalar(2), dl.scalar(3)];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

    const shuffledInputProvider =
        new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);

    const [x1provider, x2provider] = shuffledInputProvider.getInputProviders();

    const seenNumbers: {[key: number]: boolean} = {};
    for (let i = 0; i < x1s.length; i++) {
      const x1 = x1provider.getNextCopy();
      const x2 = x2provider.getNextCopy();

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
    const x1s = [dl.scalar(1), dl.scalar(2)];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();
  });

  it('different shapes within input', () => {
    const x1s = [dl.scalar(1), dl.tensor1d([1, 2])];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

    expect(() => new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]))
        .toThrowError();
  });
});

describe('InGPUMemoryShuffledInputProviderBuilder', () => {
  beforeEach(() => {
    dl.setBackend('webgl');
  });

  afterEach(() => {
    ENV.reset();
  });

  it('ensure inputs stay in sync', () => {
    const x1s = [dl.scalar(1), dl.scalar(2), dl.scalar(3)];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

    const shuffledInputProvider =
        new InCPUMemoryShuffledInputProviderBuilder([x1s, x2s]);

    const [x1provider, x2provider] = shuffledInputProvider.getInputProviders();

    const seenNumbers: {[key: number]: boolean} = {};
    for (let i = 0; i < x1s.length; i++) {
      const x1 = x1provider.getNextCopy();
      const x2 = x2provider.getNextCopy();

      expect(x1.get() * 10).toEqual(x2.get());

      seenNumbers[x1.get()] = true;
      seenNumbers[x2.get()] = true;

      x1provider.disposeCopy(x1);
      x2provider.disposeCopy(x1);
    }

    // Values are shuffled, make sure we've seen everything.
    const expectedSeenNumbers = [1, 2, 3, 10, 20, 30];
    for (let i = 0; i < expectedSeenNumbers.length; i++) {
      expect(seenNumbers[expectedSeenNumbers[i]]).toEqual(true);
    }
  });

  it('different number of examples', () => {
    const x1s = [dl.scalar(1), dl.scalar(2)];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

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
    const x1s = [dl.scalar(1), dl.tensor1d([1, 2])];
    const x2s = [dl.scalar(10), dl.scalar(20), dl.scalar(30)];

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
