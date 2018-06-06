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

import * as tf from './index';
import {describeWithFlags} from './jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, CPU_ENVS, expectArraysClose, expectArraysEqual, WEBGL_ENVS} from './test_util';

describeWithFlags('time webgl', WEBGL_ENVS, () => {
  it('upload + compute', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square()) as tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + dataSync', async () => {
    const a = tf.zeros([10, 10]);
    const time =
        await tf.time(() => a.square().dataSync()) as tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(async () => await a.square().data()) as
        tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('preupload (not included) + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    // Pre-upload a on gpu.
    a.square();
    const time = await tf.time(() => a.sqrt()) as tf.webgl.WebGLTimingInfo;
    // The tensor was already on gpu.
    expect(time.uploadWaitMs === 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});

describeWithFlags('time cpu', CPU_ENVS, () => {
  it('simple upload', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square());
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});

describeWithFlags('tidy', ALL_ENVS, () => {
  it('returns Tensor', () => {
    tf.tidy(() => {
      const a = tf.tensor1d([1, 2, 3]);
      let b = tf.tensor1d([0, 0, 0]);

      expect(tf.memory().numTensors).toBe(2);
      tf.tidy(() => {
        const result = tf.tidy(() => {
          b = tf.addStrict(a, b);
          b = tf.addStrict(a, b);
          b = tf.addStrict(a, b);
          return tf.add(a, b);
        });

        // result is new. All intermediates should be disposed.
        expect(tf.memory().numTensors).toBe(2 + 1);
        expectArraysClose(result, [4, 8, 12]);
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

  it('allows complex types', () => {
    const res = tf.tidy(() => {
      return {a: tf.scalar(1), b: 'hello', c: [tf.scalar(2), 'world']};
    });
    expectArraysClose(res.a, [1]);
    expectArraysClose(res.c[0] as tf.Scalar, [2]);
  });

  it('returns Tensor[]', () => {
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
      expectArraysClose(result[0], [1, 1, 4]);
      expectArraysClose(result[1], [1, 3, 2]);
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
      b = tf.addStrict(a, b);
      b = tf.addStrict(a, b);
      b = tf.addStrict(a, b);
      tf.add(a, b);
    });

    // all intermediates should be disposed.
    expect(tf.memory().numTensors).toBe(2);
  });

  it('nested usage', () => {
    const a = tf.tensor1d([1, 2, 3]);
    let b = tf.tensor1d([0, 0, 0]);

    expect(tf.memory().numTensors).toBe(2);

    tf.tidy(() => {
      const result = tf.tidy(() => {
        b = tf.addStrict(a, b);
        b = tf.tidy(() => {
          b = tf.tidy(() => {
            return tf.addStrict(a, b);
          });
          // original a, b, and two intermediates.
          expect(tf.memory().numTensors).toBe(4);

          tf.tidy(() => {
            tf.addStrict(a, b);
          });
          // All the intermediates should be cleaned up.
          expect(tf.memory().numTensors).toBe(4);

          return tf.addStrict(a, b);
        });
        expect(tf.memory().numTensors).toBe(4);

        return tf.addStrict(a, b);
      });

      expect(tf.memory().numTensors).toBe(3);
      expectArraysClose(result, [4, 8, 12]);
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

  it('works with arbitrary depth of result', () => {
    tf.tidy(() => {
      const res = tf.tidy(() => {
        return [tf.scalar(1), [[tf.scalar(2)]], {list: [tf.scalar(3)]}];
      });
      expectArraysEqual(res[0] as tf.Tensor, [1]);
      // tslint:disable-next-line:no-any
      expectArraysEqual((res[1] as any)[0][0], [2]);
      // tslint:disable-next-line:no-any
      expectArraysEqual((res[2] as any).list[0], [3]);
      expect(tf.memory().numTensors).toBe(3);
      return res[0];
    });
    // Everything but scalar(1) got disposed.
    expect(tf.memory().numTensors).toBe(1);
  });
});
