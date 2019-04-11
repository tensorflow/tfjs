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

import {KernelBackend} from './backends/backend';
import {ENGINE} from './engine';
import * as tf from './index';
import {ALL_ENVS, describeWithFlags, TestKernelBackend} from './jasmine_util';
import {Tensor} from './tensor';
import {expectArraysClose} from './test_util';

describe('Backend registration', () => {
  beforeAll(() => {
    // Silences backend registration warnings.
    spyOn(console, 'warn');
  });

  beforeEach(() => {
    ENGINE.reset();
  });

  it('removeBackend disposes the backend and removes the factory', () => {
    let backend: KernelBackend;
    const factory = () => {
      const newBackend = new TestKernelBackend();
      if (backend == null) {
        backend = newBackend;
        spyOn(backend, 'dispose').and.callThrough();
      }
      return newBackend;
    };

    tf.registerBackend('test-backend', factory);

    expect(tf.findBackend('test-backend') != null).toBe(true);
    expect(tf.findBackend('test-backend')).toBe(backend);
    expect(tf.findBackendFactory('test-backend')).toBe(factory);

    tf.removeBackend('test-backend');

    expect(tf.findBackend('test-backend') == null).toBe(true);
    expect(tf.findBackend('test-backend')).toBe(null);
    expect((backend.dispose as jasmine.Spy).calls.count()).toBe(1);
    expect(tf.findBackendFactory('test-backend')).toBe(null);
  });

  it('findBackend initializes the backend', () => {
    let backend: KernelBackend;
    const factory = () => {
      const newBackend = new TestKernelBackend();
      if (backend == null) {
        backend = newBackend;
      }
      return newBackend;
    };
    tf.registerBackend('custom-cpu', factory);

    expect(tf.findBackend('custom-cpu') != null).toBe(true);
    expect(tf.findBackend('custom-cpu')).toBe(backend);
    expect(tf.findBackendFactory('custom-cpu')).toBe(factory);

    tf.removeBackend('custom-cpu');
  });

  it('custom backend registration', () => {
    let backend: KernelBackend;
    const priority = 103;
    tf.registerBackend('custom-cpu', () => {
      const newBackend = new TestKernelBackend();
      if (backend == null) {
        backend = newBackend;
      }
      return newBackend;
    }, priority);

    expect(tf.backend() != null).toBe(true);
    expect(tf.backend()).toBe(backend);

    tf.removeBackend('custom-cpu');
  });

  it('high priority backend registration fails, falls back', () => {
    let lowPriorityBackend: KernelBackend;
    const lowPriority = 103;
    const highPriority = 104;
    tf.registerBackend('custom-low-priority', () => {
      lowPriorityBackend = new TestKernelBackend();
      return lowPriorityBackend;
    }, lowPriority);
    tf.registerBackend('custom-high-priority', () => {
      throw new Error(`High priority backend fails`);
    }, highPriority);

    expect(tf.backend() != null).toBe(true);
    expect(tf.backend()).toBe(lowPriorityBackend);
    expect(tf.getBackend()).toBe('custom-low-priority');

    tf.removeBackend('custom-low-priority');
    tf.removeBackend('custom-high-priority');
  });

  it('low priority and high priority backends, setBackend low priority', () => {
    let lowPriorityBackend: KernelBackend;
    let highPriorityBackend: KernelBackend;
    const lowPriority = 103;
    const highPriority = 104;
    tf.registerBackend('custom-low-priority', () => {
      lowPriorityBackend = new TestKernelBackend();
      return lowPriorityBackend;
    }, lowPriority);
    tf.registerBackend('custom-high-priority', () => {
      highPriorityBackend = new TestKernelBackend();
      return highPriorityBackend;
    }, highPriority);

    expect(tf.backend() != null).toBe(true);
    expect(tf.backend()).toBe(highPriorityBackend);
    expect(tf.getBackend()).toBe('custom-high-priority');

    tf.setBackend('custom-low-priority');

    expect(tf.backend() != null).toBe(true);
    expect(tf.backend()).toBe(lowPriorityBackend);
    expect(tf.getBackend()).toBe('custom-low-priority');

    tf.removeBackend('custom-low-priority');
    tf.removeBackend('custom-high-priority');
  });

  it('default custom background null', () => {
    expect(tf.findBackend('custom')).toBeNull();
  });

  it('allow custom backend', () => {
    const backend = new TestKernelBackend();
    const success = tf.registerBackend('custom', () => backend);
    expect(success).toBeTruthy();
    expect(tf.findBackend('custom')).toEqual(backend);
    tf.removeBackend('custom');
  });
});

describeWithFlags('memory', ALL_ENVS, () => {
  it('Sum(float)', () => {
    expect(tf.memory().numTensors).toBe(0);
    expect(tf.memory().numBytes).toBe(0);
    const sum = tf.tidy(() => {
      const a = tf.tensor1d([1, 2, 3, 4]);
      expect(tf.memory().numTensors).toBe(1);
      expect(tf.memory().numBytes).toBe(4 * 4);
      return a.sum();
    });
    expect(tf.memory().numTensors).toBe(1);
    expect(tf.memory().numBytes).toBe(4);
    expectArraysClose(sum, [1 + 2 + 3 + 4]);
  });

  it('Sum(bool)', () => {
    const sum = tf.tidy(() => {
      const a = tf.tensor1d([true, true, false, true], 'bool');
      expect(tf.memory().numTensors).toBe(1);
      expect(tf.memory().numBytes).toBe(4);
      return a.sum();
    });
    expect(tf.memory().numTensors).toBe(1);
    expect(tf.memory().numBytes).toBe(4);
    expect(sum.dtype).toBe('int32');
    expectArraysClose(sum, [1 + 1 + 0 + 1]);
  });

  it('Sum(int32)', () => {
    const sum = tf.tidy(() => {
      const a = tf.tensor1d([1, 1, 0, 1], 'int32');
      expect(tf.memory().numTensors).toBe(1);
      expect(tf.memory().numBytes).toBe(4 * 4);
      return a.sum();
    });
    expect(tf.memory().numTensors).toBe(1);
    expect(tf.memory().numBytes).toBe(4);
    expect(sum.dtype).toBe('int32');
    expectArraysClose(sum, [1 + 1 + 0 + 1]);
  });

  it('string tensor', () => {
    const a = tf.tensor([['a', 'bb'], ['c', 'd']]);

    expect(tf.memory().numTensors).toBe(1);
    expect(tf.memory().numBytes).toBe(10);  // 5 letters, each 2 bytes.

    a.dispose();

    expect(tf.memory().numTensors).toBe(0);
    expect(tf.memory().numBytes).toBe(0);
  });

  it('unreliable is true for string tensors', () => {
    tf.tensor('a');
    const mem = tf.memory();
    expect(mem.unreliable).toBe(true);
    const expectedReason = 'Memory usage by string tensors is approximate ' +
        '(2 bytes per character)';
    expect(mem.reasons.indexOf(expectedReason) >= 0).toBe(true);
  });
});

describeWithFlags('profile', ALL_ENVS, () => {
  it('squaring', async () => {
    const profile = await tf.profile(() => {
      const x = tf.tensor1d([1, 2, 3]);
      let x2 = x.square();
      x2.dispose();
      x2 = x.square();
      x2.dispose();
      return x;
    });

    const result = profile.result as Tensor;

    expect(profile.newBytes).toBe(12);
    expect(profile.peakBytes).toBe(24);
    expect(profile.newTensors).toBe(1);
    expectArraysClose(result, [1, 2, 3]);
    expect(profile.kernels).toEqual([
      {
        'name': 'square',
        'bytesAdded': 12,
        'totalBytesSnapshot': 24,
        'tensorsAdded': 1,
        'totalTensorsSnapshot': 2,
        'inputShapes': [[3]],
        'outputShape': [3]
      },
      {
        'name': 'square',
        'bytesAdded': 12,
        'totalBytesSnapshot': 24,
        'tensorsAdded': 1,
        'totalTensorsSnapshot': 2,
        'inputShapes': [[3]],
        'outputShape': [3]
      }
    ]);
  });

  it('squaring without disposing', async () => {
    const profile = await tf.profile(() => {
      const x = tf.tensor1d([1, 2, 3]);
      const x2 = x.square();
      return x2;
    });

    const result = profile.result as Tensor;

    expect(profile.newBytes).toBe(24);
    expect(profile.peakBytes).toBe(24);
    expect(profile.newTensors).toBe(2);
    expectArraysClose(result, [1, 4, 9]);
    expect(profile.kernels).toEqual([{
      'name': 'square',
      'bytesAdded': 12,
      'totalBytesSnapshot': 24,
      'tensorsAdded': 1,
      'totalTensorsSnapshot': 2,
      'inputShapes': [[3]],
      'outputShape': [3]
    }]);
  });
});

describeWithFlags('disposeVariables', ALL_ENVS, () => {
  it('reuse same name variable', () => {
    tf.tensor1d([1, 2, 3]).variable(true, 'v1');
    tf.tensor1d([1, 2, 3]).variable(true, 'v2');
    expect(() => {
      tf.tensor1d([1, 2, 3]).variable(true, 'v1');
    }).toThrowError();
    tf.disposeVariables();
    tf.tensor1d([1, 2, 3]).variable(true, 'v1');
    tf.tensor1d([1, 2, 3]).variable(true, 'v2');
  });
});

/**
 * The following test constraints to the CPU environment because it needs a
 * concrete backend to exist. This test will work for any backend, but currently
 * this is the simplest backend to test against.
 */
describeWithFlags('Switching cpu backends', {activeBackend: 'cpu'}, () => {
  beforeEach(() => {
    tf.registerBackend('cpu1', tf.findBackendFactory('cpu'));
    tf.registerBackend('cpu2', tf.findBackendFactory('cpu'));
  });

  afterEach(() => {
    tf.removeBackend('cpu1');
    tf.removeBackend('cpu2');
  });

  it('Move data from cpu1 to cpu2 backend', () => {
    tf.setBackend('cpu1');
    // This scalar lives in cpu1.
    const a = tf.scalar(5);

    tf.setBackend('cpu2');
    // This scalar lives in cpu2.
    const b = tf.scalar(3);

    expect(tf.memory().numDataBuffers).toBe(2);
    expect(tf.memory().numTensors).toBe(2);
    expect(tf.memory().numBytes).toBe(8);

    // Make sure you can read both tensors.
    expectArraysClose(a, [5]);
    expectArraysClose(b, [3]);

    // Switch back to cpu1.
    tf.setBackend('cpu1');
    // Again make sure you can read both tensors.
    expectArraysClose(a, [5]);
    expectArraysClose(b, [3]);

    tf.dispose([a, b]);

    expect(tf.memory().numDataBuffers).toBe(0);
    expect(tf.memory().numTensors).toBe(0);
    expect(tf.memory().numBytes).toBe(0);
  });

  it('can execute op with data from mixed backends', () => {
    tf.setBackend('cpu1');
    // This scalar lives in cpu1.
    const a = tf.scalar(5);

    tf.setBackend('cpu2');
    // This scalar lives in cpu2.
    const b = tf.scalar(3);

    // Verify that ops can execute with mixed backend data.
    tf.tidy(() => {
      tf.setBackend('cpu1');
      expectArraysClose(tf.add(a, b), [8]);

      tf.setBackend('cpu2');
      expectArraysClose(tf.add(a, b), [8]);
    });
    expect(tf.memory().numTensors).toBe(2);
    expect(tf.memory().numDataBuffers).toBe(2);

    tf.dispose([a, b]);

    expect(tf.memory().numTensors).toBe(0);
    expect(tf.memory().numDataBuffers).toBe(0);
  });
});

/**
 * The following unit test is a special integration-style test that assumes
 * things about CPU & WebGL backends being registered. This tests doesn't live
 * in the backend directory because it is testing engine rather than
 * backend-specific details but needs a real backend to exist. This test will
 * fail if the CPU backends is not registered. This is intentional, we should
 * have coverage for when these backends are enabled and ensure they work with
 * the engine.
 */
describeWithFlags(
    'Switching WebGL + CPU backends',
    {activeBackend: 'webgl', registeredBackends: ['webgl', 'cpu']}, () => {
      beforeEach(() => {
        tf.registerBackend('webgl1', tf.findBackendFactory('webgl'));
        tf.registerBackend('webgl2', tf.findBackendFactory('webgl'));
        tf.registerBackend('cpu1', tf.findBackendFactory('cpu'));
      });

      afterEach(() => {
        tf.removeBackend('webgl1');
        tf.removeBackend('webgl2');
        tf.removeBackend('cpu1');
      });

      it('can execute op with data from mixed backends', () => {
        tf.setBackend('webgl1');
        const a = tf.scalar(5);

        tf.setBackend('webgl2');
        const b = tf.scalar(3);

        tf.setBackend('cpu1');
        const c = tf.scalar(2);

        // Verify that ops can execute with mixed backend data.
        tf.tidy(() => {
          tf.setBackend('webgl1');
          expectArraysClose(tf.addN([a, b, c]), [10]);

          tf.setBackend('webgl2');
          expectArraysClose(tf.addN([a, b, c]), [10]);

          tf.setBackend('cpu1');
          expectArraysClose(tf.addN([a, b, c]), [10]);
        });

        expect(tf.memory().numTensors).toBe(3);
        expect(tf.memory().numDataBuffers).toBe(3);

        tf.dispose([a, b, c]);

        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numDataBuffers).toBe(0);
      });

      it('fromPixels with mixed backends works', () => {
        tf.setBackend('webgl1');
        const a = tf.browser.fromPixels(
            new ImageData(new Uint8ClampedArray([1, 2, 3, 4]), 1, 1));

        tf.setBackend('webgl2');
        const b = tf.browser.fromPixels(
            new ImageData(new Uint8ClampedArray([5, 6, 7, 8]), 1, 1));

        expectArraysClose(tf.add(a, b), [6, 8, 10]);
      });

      it('single tidy multiple backends', () => {
        expect(tf.memory().numTensors).toBe(0);

        tf.tidy(() => {
          tf.setBackend('webgl1');
          const a = tf.scalar(1);
          a.square();  // Uploads to GPU.

          tf.setBackend('webgl2');
          const b = tf.scalar(1);
          b.square();  // Uploads to GPU.

          expect(tf.memory().numTensors).toBe(4);
        });
        expect(tf.memory().numTensors).toBe(0);
      });
    });

// NOTE: This describe is purposefully not a describeWithFlags so that we test
// tensor allocation where no scopes have been created. The backend here must
// be set to CPU because we cannot allocate GPU tensors outside a
// describeWithFlags because the default webgl backend and the test backends
// share a WebGLContext. When backends get registered, global WebGL state is
// initialized, which causes the two backends to step on each other and get in
// a bad state.
describe('Memory allocation outside a test scope', () => {
  it('constructing a tensor works', () => {
    tf.setBackend('cpu');
    const a = tf.tensor1d([1, 2, 3]);
    expectArraysClose(a, [1, 2, 3]);
    a.dispose();
  });
});
