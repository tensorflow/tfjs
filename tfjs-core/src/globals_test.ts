/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from './index';
import {ALL_ENVS, describeWithFlags, NODE_ENVS} from './jasmine_util';
import {expectArraysClose} from './test_util';

describe('deprecation warnings', () => {
  beforeEach(() => {
    spyOn(console, 'warn').and.callFake((msg: string): void => null);
  });

  it('deprecationWarn warns', () => {
    // flags_test.ts verifies deprecation warnings are on by default.
    const deprecationVal = tf.env().get('DEPRECATION_WARNINGS_ENABLED');
    tf.env().set('DEPRECATION_WARNINGS_ENABLED', true);
    tf.deprecationWarn('xyz is deprecated.');
    tf.env().set('DEPRECATION_WARNINGS_ENABLED', deprecationVal);
    expect(console.warn).toHaveBeenCalledTimes(1);
    expect(console.warn)
        .toHaveBeenCalledWith(
            'xyz is deprecated. You can disable deprecation warnings with ' +
            'tf.disableDeprecationWarnings().');
  });

  it('disableDeprecationWarnings called, deprecationWarn doesnt warn', () => {
    tf.disableDeprecationWarnings();
    expect(console.warn).toHaveBeenCalledTimes(1);
    expect(console.warn)
        .toHaveBeenCalledWith(
            'TensorFlow.js deprecation warnings have been disabled.');

    // deprecationWarn no longer warns.
    tf.deprecationWarn('xyz is deprecated.');
    expect(console.warn).toHaveBeenCalledTimes(1);
  });
});

describe('Flag flipping methods', () => {
  beforeEach(() => {
    tf.env().reset();
  });

  afterEach(() => {
    tf.env().reset();
  });

  it('tf.enableProdMode', () => {
    tf.enableProdMode();
    expect(tf.env().getBool('PROD')).toBe(true);
  });

  it('tf.enableDebugMode', () => {
    // Silence debug warnings.
    spyOn(console, 'warn');
    tf.enableDebugMode();
    expect(tf.env().getBool('DEBUG')).toBe(true);
  });
});

describeWithFlags('time cpu', NODE_ENVS, () => {
  it('simple upload', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square());
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});

describeWithFlags('tidy', ALL_ENVS, () => {
  it('returns Tensor', async () => {
    tf.tidy(() => {
      const a = tf.tensor1d([1, 2, 3]);
      let b = tf.tensor1d([0, 0, 0]);

      expect(tf.memory().numTensors).toBe(2);
      tf.tidy(() => {
        const result = tf.tidy(() => {
          b = tf.add(a, b);
          b = tf.add(a, b);
          b = tf.add(a, b);
          return tf.add(a, b);
        });

        // result is new. All intermediates should be disposed.
        expect(tf.memory().numTensors).toBe(2 + 1);
        expect(result.shape).toEqual([3]);
        expect(result.isDisposed).toBe(false);
      });

      // a, b are still here, result should be disposed.
      expect(tf.memory().numTensors).toBe(2);
    });

    expect(tf.memory().numTensors).toBe(0);
  });

  it('multiple disposes does not affect num arrays', () => {
    expect(tf.memory().numTensors).toBe(0);
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([1, 2, 3]);
    expect(tf.memory().numTensors).toBe(2);
    a.dispose();
    a.dispose();
    expect(tf.memory().numTensors).toBe(1);
    b.dispose();
    expect(tf.memory().numTensors).toBe(0);
  });

  it('allows primitive types', () => {
    const a = tf.tidy(() => 5);
    expect(a).toBe(5);

    const b = tf.tidy(() => 'hello');
    expect(b).toBe('hello');
  });

  it('allows complex types', async () => {
    const res = tf.tidy(() => {
      return {a: tf.scalar(1), b: 'hello', c: [tf.scalar(2), 'world']};
    });
    expectArraysClose(await res.a.data(), [1]);
    expectArraysClose(await (res.c[0] as tf.Tensor).data(), [2]);
  });

  it('returns Tensor[]', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([0, -1, 1]);
    expect(tf.memory().numTensors).toBe(2);

    tf.tidy(() => {
      const result = tf.tidy(() => {
        tf.add(a, b);
        return [tf.add(a, b), tf.sub(a, b)];
      });

      // the 2 results are new. All intermediates should be disposed.
      expect(tf.memory().numTensors).toBe(4);
      expect(result[0].isDisposed).toBe(false);
      expect(result[0].shape).toEqual([3]);
      expect(result[1].isDisposed).toBe(false);
      expect(result[1].shape).toEqual([3]);
      expect(tf.memory().numTensors).toBe(4);
    });

    // the 2 results should be disposed.
    expect(tf.memory().numTensors).toBe(2);
    a.dispose();
    b.dispose();
    expect(tf.memory().numTensors).toBe(0);
  });

  it('basic usage without return', () => {
    const a = tf.tensor1d([1, 2, 3]);
    let b = tf.tensor1d([0, 0, 0]);

    expect(tf.memory().numTensors).toBe(2);

    tf.tidy(() => {
      b = tf.add(a, b);
      b = tf.add(a, b);
      b = tf.add(a, b);
      tf.add(a, b);
    });

    // all intermediates should be disposed.
    expect(tf.memory().numTensors).toBe(2);
  });

  it('nested usage', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    let b = tf.tensor1d([0, 0, 0]);

    expect(tf.memory().numTensors).toBe(2);

    tf.tidy(() => {
      const result = tf.tidy(() => {
        b = tf.add(a, b);
        b = tf.tidy(() => {
          b = tf.tidy(() => {
            return tf.add(a, b);
          });
          // original a, b, and two intermediates.
          expect(tf.memory().numTensors).toBe(4);

          tf.tidy(() => {
            tf.add(a, b);
          });
          // All the intermediates should be cleaned up.
          expect(tf.memory().numTensors).toBe(4);

          return tf.add(a, b);
        });
        expect(tf.memory().numTensors).toBe(4);

        return tf.add(a, b);
      });

      expect(tf.memory().numTensors).toBe(3);
      expect(result.isDisposed).toBe(false);
      expect(result.shape).toEqual([3]);
    });
    expect(tf.memory().numTensors).toBe(2);
  });

  it('nested usage returns tensor created from outside scope', () => {
    const x = tf.scalar(1);

    tf.tidy(() => {
      tf.tidy(() => {
        return x;
      });
    });

    expect(x.isDisposed).toBe(false);
  });

  it('nested usage with keep works', () => {
    let b: tf.Tensor;
    tf.tidy(() => {
      const a = tf.scalar(1);
      tf.tidy(() => {
        b = tf.keep(a);
      });
    });

    expect(b.isDisposed).toBe(false);
    b.dispose();
  });

  it('single argument', () => {
    let hasRan = false;
    tf.tidy(() => {
      hasRan = true;
    });
    expect(hasRan).toBe(true);
  });

  it('single argument, but not a function throws error', () => {
    expect(() => {
      tf.tidy('asdf');
    }).toThrowError();
  });

  it('2 arguments, first is string', () => {
    let hasRan = false;
    tf.tidy('name', () => {
      hasRan = true;
    });
    expect(hasRan).toBe(true);
  });

  it('2 arguments, but first is not string throws error', () => {
    expect(() => {
      // tslint:disable-next-line:no-any
      tf.tidy(4 as any, () => {});
    }).toThrowError();
  });

  it('2 arguments, but second is not a function throws error', () => {
    expect(() => {
      // tslint:disable-next-line:no-any
      tf.tidy('name', 'another name' as any);
    }).toThrowError();
  });

  it('works with arbitrary depth of result', async () => {
    tf.tidy(() => {
      const res = tf.tidy(() => {
        return [tf.scalar(1), [[tf.scalar(2)]], {list: [tf.scalar(3)]}];
      });
      expect((res[0] as tf.Tensor).isDisposed).toBe(false);
      // tslint:disable-next-line:no-any
      expect((res[1] as any)[0][0].isDisposed).toBe(false);
      // tslint:disable-next-line:no-any
      expect((res[2] as any).list[0].isDisposed).toBe(false);
      expect(tf.memory().numTensors).toBe(3);
      return res[0];
    });
    // Everything but scalar(1) got disposed.
    expect(tf.memory().numTensors).toBe(1);
  });
});
