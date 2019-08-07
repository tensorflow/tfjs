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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {Tensor2D} from '../../tensor';
import {expectArraysClose, expectArraysEqual} from '../../test_util';
import {Rank} from '../../types';

import {WebGLMemoryInfo} from './backend_webgl';
import {PACKED_ENVS, WEBGL_ENVS} from './backend_webgl_test_registry';

describeWithFlags('fromPixels + regular math op', WEBGL_ENVS, () => {
  it('fromPixels + add', async () => {
    const pixels = new ImageData(2, 2);
    for (let i = 0; i < 8; i++) {
      pixels.data[i] = 100;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = 250;
    }

    const a = tf.browser.fromPixels(pixels, 4);
    const b = tf.scalar(20, 'int32');

    const res = tf.add(a, b);

    expectArraysEqual(await res.data(), [
      120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270, 270,
      270
    ]);
  });
});

describeWithFlags('toPixels', WEBGL_ENVS, () => {
  it('draws a rank-2 float32 tensor, canvas', done => {
    const x = tf.tensor2d([.15, .2], [2, 1], 'float32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
        255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
        255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-2 int32 tensor, canvas', done => {
    const x = tf.tensor2d([10, 20], [2, 1], 'int32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 1 channel, canvas', done => {
    const x = tf.tensor3d([.15, .2], [2, 1, 1], 'float32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
        255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
        255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 1 channel, canvas', done => {
    const x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 3 channel, canvas', done => {
    const x =
        tf.tensor3d([.05, .1001, .15, .20, .25, .3001], [2, 1, 3], 'float32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
        255, Math.round(.2 * 255), Math.round(.25 * 255),
        Math.round(.3001 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 3 channel, canvas', done => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);
      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 float32 tensor, 4 channel, canvas', done => {
    // ImageData roundtrips are lossy because of pre-multiplied alphas, so we
    // use an alpha = 1 to avoid losing precision on r, g, b channels in these
    // tests https://www.w3.org/TR/2dcontext/
    const x = tf.tensor3d(
        [.05, .1001, .15, 1, .20, .25, .3001, 1], [2, 1, 4], 'float32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected = new Uint8ClampedArray([
        Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
        255, Math.round(.20 * 255), Math.round(.25 * 255),
        Math.round(.3001 * 255), 255
      ]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('draws a rank-3 int32 tensor, 4 channel, canvas', done => {
    // ImageData roundtrips are lossy because of pre-multiplied alphas, so we
    // use an alpha = 1 to avoid losing precision on r, g, b channels in these
    // tests https://www.w3.org/TR/2dcontext/
    const x =
        tf.tensor3d([10, 20, 30, 255, 50, 60, 70, 255], [2, 1, 4], 'int32');
    const canvas = document.createElement('canvas');

    tf.browser.toPixels(x, canvas).then(data => {
      const expected =
          new Uint8ClampedArray([10, 20, 30, 255, 50, 60, 70, 255]);
      expect(data).toEqual(expected);

      const ctx = canvas.getContext('2d');
      const imgData = ctx.getImageData(0, 0, 1, 2);

      expect(imgData.data).toEqual(expected);
      done();
    });
  });

  it('accepts a tensor-like object', async () => {
    const x = [[127], [100]];  // 2x1;
    const canvas = document.createElement('canvas');

    const data = await tf.browser.toPixels(x, canvas);
    const expected =
        new Uint8ClampedArray([127, 127, 127, 255, 100, 100, 100, 255]);
    expect(data).toEqual(expected);

    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, 1, 2);

    expect(imgData.data).toEqual(expected);
  });
});

describeWithFlags('depthToSpace', WEBGL_ENVS, () => {
  it('tensor4d, input shape=[1, 4, 1, 1], blockSize=2, format=NCHW',
     async () => {
       const t = tf.tensor4d([1, 2, 3, 4], [1, 4, 1, 1]);
       const blockSize = 2;
       const dataFormat = 'NCHW';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 1, 2, 2]);
       expectArraysClose(await res.data(), [1, 2, 3, 4]);
     });

  it('tensor4d, input shape=[1, 12, 1, 1], blockSize=2, format=NCHW',
     async () => {
       const t =
           tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 12, 1, 1]);
       const blockSize = 2;
       const dataFormat = 'NCHW';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 3, 2, 2]);
       expectArraysClose(
           await res.data(), [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]);
     });

  it('tensor4d, input shape=[1, 4, 2, 2], blockSize=2, format=NCHW',
     async () => {
       const t = tf.tensor4d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
           [1, 4, 2, 2]);
       const blockSize = 2;
       const dataFormat = 'NCHW';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 1, 4, 4]);
       expectArraysClose(
           await res.data(),
           [1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16]);
     });

  it('tensor4d, input shape=[1, 8, 2, 2], blockSize=2, format=NCHW',
     async () => {
       const t = tf.tensor4d(
           [
             1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
           ],
           [1, 8, 2, 2]);
       const blockSize = 2;
       const dataFormat = 'NCHW';

       const res = tf.depthToSpace(t, blockSize, dataFormat);
       expect(res.shape).toEqual([1, 2, 4, 4]);
       expectArraysClose(await res.data(), [
         1, 9,  2, 10, 17, 25, 18, 26, 3, 11, 4, 12, 19, 27, 20, 28,
         5, 13, 6, 14, 21, 29, 22, 30, 7, 15, 8, 16, 23, 31, 24, 32
       ]);
     });
});

describeWithFlags('maximum', WEBGL_ENVS, () => {
  it('works with squarification for large dimension', async () => {
    const maxTextureSize = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);
    const a =
        tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [2, 7]);
    const b =
        tf.tensor2d([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 7]);

    const result = tf.maximum(a, b);
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);
    expectArraysClose(
        await result.data(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
  });
});

describeWithFlags('div', PACKED_ENVS, () => {
  it('works when unused channels are divided', async () => {
    // Tests that the 0's in unused channels for input textures do not corrupt
    // the result when swizzled with 3 / 3.
    const a = tf.tensor2d([1], [1, 1]);
    const b = tf.tensor2d([1], [1, 1]);

    const c = tf.add(a, b).div(a);
    const d = tf.add(a, b).div(a);

    const result = c.matMul(d);
    expectArraysClose(await result.data(), [4]);
  });

  it('works when unused channels in tensors with size > 1 are divided',
     async () => {
       const a = tf.tensor2d([1, 2, 3], [3, 1]);
       const b = tf.tensor2d([1, 2, 3], [3, 1]);
       const c = a.div(b);

       const d = tf.tensor1d([1, 2, 3]);
       const e = tf.tensor1d([1, 2, 3]);
       const f = d.div(e).reshape([1, 3]);

       const result = c.matMul(f);
       expectArraysClose(await result.data(), [1, 1, 1, 1, 1, 1, 1, 1, 1]);
     });
});

describeWithFlags('conv2d webgl', WEBGL_ENVS, () => {
  it('packed input x=[2,1,2] f=[1,1,2,2] s=1 d=1 p=0', async () => {
    const inputShape: [number, number, number] = [2, 1, 2];
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor4d([1, 2, 3, 4], [fSize, fSize, 2, 2]);

    const webglLazilyUnpackFlagSaved = tf.ENV.getBool('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

    // First conv2D tests conv2D with non-packed input |x|, and the second uses
    // packed input |result|.
    const result = tf.conv2d(x, w, stride, pad);
    const result1 = tf.conv2d(result, w, stride, pad);

    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expectArraysClose(await result.data(), [7, 10, 15, 22]);
    expectArraysClose(await result1.data(), [37, 54, 81, 118]);
  });

  it('tf.memory() packed input x=[1,1,1,2] f=[1,1,2,2] s=1 d=1 p=0',
     async () => {
       const inputShape: [number, number, number, number] = [1, 1, 1, 2];
       const fSize = 1;
       const pad = 0;
       const stride = 1;

       const xInit = tf.tensor4d([0, 1], inputShape);
       const w = tf.tensor4d([1, 2, 3, 4], [fSize, fSize, 2, 2]);

       const webglLazilyUnpackFlagSaved = tf.ENV.getBool('WEBGL_LAZILY_UNPACK');
       tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
       const webglPackBinaryOperationsFlagSaved =
           tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
       tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

       const x = xInit.add<tf.Tensor4D>(1);
       const result = tf.conv2d(x, w, stride, pad);

       tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
       tf.ENV.set(
           'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

       expectArraysClose(await result.data(), [7, 10]);
       result.dispose();
       x.dispose();
       xInit.dispose();
       w.dispose();
       expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU).toBe(0);
       expect(tf.memory().numBytes).toBe(0);
     });
});

describeWithFlags('conv to matmul', PACKED_ENVS, () => {
  it('im2col should not leak memory', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        tf.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);

    const startNumBytes = tf.memory().numBytes;
    tf.conv2d(x, w, stride, pad, dataFormat, dilation);
    const endNumBytes = tf.memory().numBytes;

    expect(endNumBytes - startNumBytes).toEqual(4);
  });

  it('pointwise conv should work when matmul is unpacked', () => {
    const inputDepth =
        1001;  // this number must be greater than MATMUL_SHARED_DIM_THRESHOLD
               // for matmul to be unpacked
    const inputShape: [number, number, number] = [3, 3, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride: [number, number] = [1, 1];

    let x = tf.randomNormal(inputShape) as tf.Tensor3D;
    x = x.add(1);  // this packs x so we can test the case where we mistakenly
                   // want to avoid expensive reshape in pointwise conv2d even
                   // though matmul is unpacked
    const w =
        tf.randomNormal([fSize, fSize, inputDepth, outputDepth]) as tf.Tensor4D;

    expect(() => tf.conv2d(x, w, stride, pad)).not.toThrow();
  });
});

// For operations on non-trivial matrix sizes, we skip the CPU-only ENV and use
// only WebGL ENVs.
describeWithFlags('gramSchmidt-non-tiny', WEBGL_ENVS, () => {
  it('8x16', async () => {
    // Part of this test's point is that operation on a matrix of this size
    // can complete in the timeout limit of the unit test.
    const xs = tf.randomUniform([8, 16]) as Tensor2D;
    const y = tf.linalg.gramSchmidt(xs) as Tensor2D;
    expectArraysClose(
        await y.matMul(y.transpose()).data(), await tf.eye(8).data());
  });
});

describeWithFlags('matmul webgl-only', WEBGL_ENVS, () => {
  it('Matrix times vector, large matrix', async () => {
    const maxTexSize = 16000;
    const sharedDim = maxTexSize + 4;
    const matrix = tf.buffer<Rank.R2>([2, sharedDim], 'float32');
    matrix.set(1, 0, sharedDim - 3);
    matrix.set(1, 0, sharedDim - 2);

    const v = tf.buffer<Rank.R1>([sharedDim], 'float32');
    v.set(1, sharedDim - 3);
    v.set(1, sharedDim - 2);

    const result = tf.dot(matrix.toTensor(), v.toTensor());
    const expected = [2, 0];
    expectArraysClose(await result.data(), expected);
  });
});

describeWithFlags('matmul', PACKED_ENVS, () => {
  it('should not leak memory', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const b = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);

    const startNumBytes = tf.memory().numBytes;
    tf.matMul(a, b);
    const endNumBytes = tf.memory().numBytes;

    expect(endNumBytes - startNumBytes).toEqual(60);
  });

  it('should work when input matrix dimensions are not divisible by 2',
     async () => {
       const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
       const b = tf.tensor2d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);

       const c = tf.matMul(a, b);

       expect(c.shape).toEqual([3, 5]);
       expectArraysClose(await c.data(), [
         46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250
       ]);
     });

  it('should work when output texture shape != physical shape', async () => {
    const sharedDim = 16000;
    const a = tf.buffer<Rank.R2>([2, sharedDim], 'float32');
    const b = tf.buffer<Rank.R2>([sharedDim, 2], 'float32');

    a.set(1, 0, sharedDim - 1);
    a.set(1, 0, sharedDim - 2);
    a.set(1, 1, sharedDim - 1);
    b.set(1, sharedDim - 1, 0);
    b.set(1, sharedDim - 2, 0);

    const c = tf.matMul(a.toTensor(), b.toTensor());
    const expected = [2, 0, 1, 0];
    expectArraysClose(await c.data(), expected);
  });

  it('should work when input texture shapes != physical shape', async () => {
    const maxTextureSize = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);
    const a = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12]);
    const b = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1]);

    const c = tf.matMul(a, b);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);

    expectArraysClose(await c.data(), [572]);
  });

  it('should work when squarification results in zero padding', async () => {
    const maxTextureSize = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 3);
    const a = tf.tensor2d([1, 2], [1, 2]);
    const b = tf.tensor2d(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17]]);

    const c = tf.matMul(a, b);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);

    expectArraysClose(await c.data(), [18, 21, 24, 27, 30, 33, 36, 39, 42]);
  });

  it('A x B', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [0, 8, -3, 20]);
  });

  it('A x B^t', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = false;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [7, 10, 16, 31];
    expectArraysClose(await c.data(), expected);
  });

  it('A^t x B', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = false;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
    expectArraysClose(await c.data(), expected);
  });

  it('A^t x B^t', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const b = tf.tensor2d([1, 0, 2, 4, 3, 0], [2, 3]);

    const transposeA = true;
    const transposeB = true;
    const c = tf.matMul(a, b, transposeA, transposeB);

    const expected = [11, 13, 14, 20];
    expectArraysClose(await c.data(), expected);
  });

  it('works when followed by an op that requires unpacked inputs', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    const webglPackBinarySaved = tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', false);
    const d = tf.add(c, 1);
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', webglPackBinarySaved);

    expectArraysClose(await d.data(), [1, 9, -2, 21]);
  });

  // tslint:disable-next-line:max-line-length
  it('works when followed by a packed reshape that changes texture layout, and then an unpacked op',
     async () => {
       const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 1]);
       const b = tf.tensor2d([1], [1, 1]);
       const c = tf.matMul(a, b);

       const d = tf.reshape(c, [1, 3, 3, 1]);

       const webglPackBinarySaved =
           tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
       tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', false);
       const e = tf.add(d, 1);
       tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', webglPackBinarySaved);

       expectArraysClose(await e.data(), [2, 3, 4, 5, 6, 7, 8, 9, 10]);
     });

  it('works when preceded by an op that requires packed inputs', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.add(a, 1);
    const d = tf.matMul(b, c);

    expectArraysClose(await d.data(), [5, 6, 7, 4, 3, 2, 9, 12, 15]);
  });
});

describeWithFlags('Reduction: webgl packed input', WEBGL_ENVS, () => {
  it('argmax 3D, odd number of rows, axis = -1', async () => {
    const webglLazilyUnpackFlagSaved = tf.ENV.getBool('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

    const a = tf.tensor3d([3, 2, 5, 100, -7, 2], [2, 1, 3]).add(1);
    const r = tf.argMax(a, -1);
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [2, 0]);
  });

  it('argmin 4D, odd number of rows, axis = -1', async () => {
    const webglLazilyUnpackFlagSaved = tf.ENV.getBool('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

    const a =
        tf.tensor4d(
              [3, 2, 5, 100, -7, 2, 8, 7, -5, 101, 7, -2, 100, -7, 2, 8, 7, -5],
              [1, 2, 3, 3])
            .add(1);
    const r = tf.argMin(a, -1);
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 1, 2, 2, 1, 2]);
  });

  it('should not leak memory when called after unpacked op', async () => {
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', false);

    const a =
        tf.tensor5d(
              [3, 2, 5, 100, -7, 2, 8, 7, -5, 101, 7, -2, 100, -7, 2, 8, 7, -5],
              [1, 2, 3, 1, 3])
            .add(1);
    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const r = tf.argMin(a, -1);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    expect(endNumBytes - startNumBytes).toEqual(24);
    expect(endNumTensors - startNumTensors).toEqual(1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 1, 2, 2, 1, 2]);
  });
});

describeWithFlags('slice and memory usage', WEBGL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('WEBGL_CPU_FORWARD', false);
    tf.ENV.set('WEBGL_SIZE_UPLOAD_UNIFORM', 0);
  });

  it('slice a tensor, read it and check memory', async () => {
    const getMem = () => tf.memory() as WebGLMemoryInfo;
    expect(getMem().numBytesInGPU).toBe(0);

    // Lazy upload won't increase gpu memory.
    const a = tf.tensor([2, 3]);
    expect(getMem().numBytesInGPU).toBe(0);

    // Upload a to the GPU by running an op.
    a.square().dispose();
    expect(getMem().numBytesInGPU).toBe(8);

    // Slicing does not allocate new memory.
    const b = a.slice(0);
    expect(getMem().numBytesInGPU).toBe(8);

    // Download a to the CPU but the texture remains on GPU
    // since b points to it.
    await a.data();
    expect(getMem().numBytesInGPU).toBe(8);

    // Dispose a, but the texture should still remain on the GPU
    // since b points to it.
    a.dispose();
    expect(getMem().numBytesInGPU).toBe(8);

    // Dispose b and expect 0 memory on GPU.
    b.dispose();
    expect(getMem().numBytesInGPU).toBe(0);
  });
});

describeWithFlags('slice a packed texture', WEBGL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('WEBGL_PACK', true);
  });

  it('slice after a matmul', async () => {
    const a = [[1, 2], [3, 4]];
    const b = [[5, 6], [7, 8]];
    // Matmul gives a packed tensor in webgl.
    //  [19, 22]
    //  [43, 50]
    const c = tf.matMul(a, b);
    expectArraysClose(await c.slice([0, 0]).data(), [19, 22, 43, 50]);
    expectArraysClose(await c.slice([0, 1]).data(), [22, 50]);
    expectArraysClose(await c.slice([1, 0]).data(), [43, 50]);
    expectArraysClose(await c.slice([1, 1]).data(), [50]);
  });
});

describeWithFlags('relu', WEBGL_ENVS, () => {
  it('works with squarification for prime number length vector', async () => {
    const maxTextureSize = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);
    const a = tf.tensor1d([1, -2, 5, -3, -1, 4, 7]);
    const result = tf.relu(a);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);
    expectArraysClose(await result.data(), [1, 0, 5, 0, 0, 4, 7]);
  });
});

describeWithFlags('packed clip', PACKED_ENVS, () => {
  it('should not leak memory', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = -1;
    const max = 50;

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    tf.clipByValue(a, min, max);
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;

    expect(endNumBytes - startNumBytes).toEqual(24);
    expect(endNumTensors - startNumTensors).toEqual(1);
  });

  it('basic', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(await result.data(), [3, -1, 0, 50, -1, 2]);
  });

  it('using extreme values', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    let result =
        tf.clipByValue(a, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY);
    expectArraysClose(await result.data(), [3, -1, 0, 100, -7, 2]);

    result = tf.clipByValue(a, Number.MIN_VALUE, Number.MAX_VALUE);
    expectArraysClose(
        await result.data(),
        [3, Number.MIN_VALUE, Number.MIN_VALUE, 100, Number.MIN_VALUE, 2]);
  });

  it('should work for scalars', async () => {
    const a = tf.scalar(-4);
    const min = -1;
    const max = 50;

    const result = tf.clipByValue(a, min, max);

    expectArraysClose(await result.data(), [min]);
  });

  it('derivative: 1D tensor with max or min value', async () => {
    const min = -1;
    const max = 2;
    const x = tf.tensor1d([-1, 1, 2, 3]);
    const dy = tf.tensor1d([1, 10, 100, 1000]);
    const gradients = tf.grad(x => x.clipByValue(min, max))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [1, 10, 100, 0]);
  });
});

describeWithFlags('depthwiseConv2d packed', PACKED_ENVS, () => {
  it('should not leak memory', () => {
    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, 1]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [2, 2, 1, 1],
    );

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;

    tf.depthwiseConv2d(x, w, 1, 'valid');

    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;

    expect(endNumBytes - startNumBytes).toEqual(16);
    expect(endNumTensors - startNumTensors).toEqual(1);
  });
});
