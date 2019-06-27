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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../../test_util';
import {decodeString, encodeString} from '../../util';

import {MathBackendWebGL, WebGLMemoryInfo} from './backend_webgl';
import {WEBGL_ENVS} from './backend_webgl_test_registry';

function encodeStrings(a: string[]): Uint8Array[] {
  return a.map(s => encodeString(s));
}

function decodeStrings(bytes: Uint8Array[]): string[] {
  return bytes.map(b => decodeString(b));
}

describeWithFlags('lazy packing and unpacking', WEBGL_ENVS, () => {
  let webglLazilyUnpackFlagSaved: boolean;
  let webglCpuForwardFlagSaved: boolean;

  beforeAll(() => {
    webglLazilyUnpackFlagSaved =
        tf.ENV.getBool('WEBGL_LAZILY_UNPACK') as boolean;
    webglCpuForwardFlagSaved = tf.ENV.getBool('WEBGL_CPU_FORWARD') as boolean;
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    tf.ENV.set('WEBGL_CPU_FORWARD', false);
  });

  afterAll(() => {
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set('WEBGL_CPU_FORWARD', webglCpuForwardFlagSaved);
  });

  it('should not leak memory when lazily unpacking', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    // c is packed to 1x1 RGBA texture.
    const c = tf.matMul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU =
        (tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU;

    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', false);
    // Add will unpack c before the operation to 2
    tf.add(c, 1);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(tf.memory().numBytes - startNumBytes).toEqual(16);
    expect(tf.memory().numTensors - startNumTensors).toEqual(1);
    // result is unpacked 2x2 R texture.
    expect(
        (tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU -
        startNumBytesInGPU)
        .toEqual(4 * tf.util.bytesPerElement(a.dtype));
  });

  it('should not leak memory when lazily packing', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.add(a, 1);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU =
        (tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU;

    tf.matMul(b, c);

    expect(tf.memory().numBytes - startNumBytes).toEqual(36);
    expect(tf.memory().numTensors - startNumTensors).toEqual(1);
    // result [3, 3] is packed to four RGBA pixel texture b is packed to two
    // RGBA texels texture: total 6 * 4 = 24 components.
    expect(
        (tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU -
        startNumBytesInGPU)
        .toEqual(24 * tf.util.bytesPerElement(a.dtype));
  });

  it('should work when the same input must be represented by' +
         'different textures',
     async () => {
       const a = tf.tensor1d([1, 2]);
       const res = tf.dot(a, a);
       expectArraysClose(await res.data(), [5]);
     });
});

describeWithFlags('backendWebGL', WEBGL_ENVS, () => {
  let prevBackend: string;

  beforeAll(() => {
    prevBackend = tf.getBackend();
  });

  afterEach(() => {
    tf.setBackend(prevBackend);
    tf.removeBackend('test-storage');
  });

  it('register empty string tensor', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const t = tf.Tensor.make([3], {}, 'string');
    expect(backend.readSync(t.dataId) == null).toBe(true);
  });

  it('register empty string tensor and write', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const t = tf.Tensor.make([3], {}, 'string');
    backend.write(t.dataId, encodeStrings(['c', 'a', 'b']));
    expectArraysEqual(
        decodeStrings(backend.readSync(t.dataId) as Uint8Array[]),
        ['c', 'a', 'b']);
  });

  it('register string tensor with values', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const t = tf.Tensor.make([3], {values: ['a', 'b', 'c']}, 'string');
    expectArraysEqual(
        decodeStrings(backend.readSync(t.dataId) as Uint8Array[]),
        ['a', 'b', 'c']);
  });

  it('register string tensor with values and overwrite', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const t = tf.Tensor.make([3], {values: ['a', 'b', 'c']}, 'string');
    backend.write(t.dataId, encodeStrings(['c', 'a', 'b']));
    expectArraysEqual(
        decodeStrings(backend.readSync(t.dataId) as Uint8Array[]),
        ['c', 'a', 'b']);
  });

  it('register string tensor with values and wrong shape throws error', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');
    expect(() => tf.tensor(['a', 'b', 'c'], [4], 'string')).toThrowError();
  });

  it('reading', () => {
    const backend = new MathBackendWebGL(null);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const texManager = backend.getTextureManager();
    const t = tf.Tensor.make([3], {}, 'float32');
    backend.write(t.dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(
        backend.readSync(t.dataId) as Float32Array,
        new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('read packed and then use by an unpacked op', async () => {
    const backend = new MathBackendWebGL(null);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const webglPackFlagSaved = tf.ENV.getBool('WEBGL_PACK');
    tf.ENV.set('WEBGL_PACK', true);
    const webglSizeUploadUniformSaved =
        tf.ENV.getNumber('WEBGL_SIZE_UPLOAD_UNIFORM');
    tf.ENV.set('WEBGL_SIZE_UPLOAD_UNIFORM', 0);
    const a = tf.tensor2d([1, 2], [2, 1]);
    const b = tf.tensor2d([1], [1, 1]);
    const c = tf.matMul(a, b);
    backend.readSync(c.dataId);
    tf.ENV.set('WEBGL_PACK', false);
    const d = tf.add(c, 1);
    tf.ENV.set('WEBGL_PACK', webglPackFlagSaved);
    tf.ENV.set('WEBGL_SIZE_UPLOAD_UNIFORM', webglSizeUploadUniformSaved);
    expectArraysClose(await d.data(), [2, 3]);
  });

  it('delayed storage, overwriting', () => {
    const backend = new MathBackendWebGL(null);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const texManager = backend.getTextureManager();
    const t = tf.Tensor.make([3], {}, 'float32');
    backend.write(t.dataId, new Float32Array([1, 2, 3]));
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    // overwrite.
    backend.write(t.dataId, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    expectArraysClose(
        backend.readSync(t.dataId) as Float32Array,
        new Float32Array([4, 5, 6]));
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(
        backend.readSync(t.dataId) as Float32Array,
        new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
});

describeWithFlags('Custom window size', WEBGL_ENVS, () => {
  it('Set screen area to be 1x1', () => {
    // This will set the screen size to 1x1 to make sure the page limit is
    // very small.
    spyOnProperty(window, 'screen', 'get')
        .and.returnValue({height: 1, width: 1});

    tf.registerBackend('custom-webgl', () => new MathBackendWebGL());
    tf.setBackend('custom-webgl');

    // Allocate ~40KB.
    const a = tf.ones([100, 100]);
    // No gpu memory used yet because of delayed storage.
    expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU).toBe(0);

    // Expect console.warn() to be called.
    let numWarnCalls = 0;
    spyOn(console, 'warn').and.callFake(() => {
      numWarnCalls++;
    });

    a.square();
    expect(numWarnCalls).toBe(1);
    expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU)
        .toBe(100 * 100 * 4 * 2);

    // Allocate another 40KB.
    a.square();

    // Expect console.warn() to NOT be called more than once.
    expect(numWarnCalls).toBe(1);
    expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU)
        .toBe(100 * 100 * 4 * 3);
    tf.removeBackend('custom-webgl');
  });
});

const SIZE_UPLOAD_UNIFORM = 4;
// Run only for environments that have 32bit floating point support.
const FLOAT32_WEBGL_ENVS = {
  flags: {'WEBGL_RENDER_FLOAT32_ENABLED': true},
  predicate: WEBGL_ENVS.predicate
};

describeWithFlags('upload tensors as uniforms', FLOAT32_WEBGL_ENVS, () => {
  let savedUploadUniformValue: number;

  beforeAll(() => {
    savedUploadUniformValue = tf.ENV.get('WEBGL_SIZE_UPLOAD_UNIFORM') as number;
    tf.ENV.set('WEBGL_SIZE_UPLOAD_UNIFORM', SIZE_UPLOAD_UNIFORM);
  });

  afterAll(() => {
    tf.ENV.set('WEBGL_SIZE_UPLOAD_UNIFORM', savedUploadUniformValue);
  });

  it('small tensor gets uploaded as scalar', () => {
    let m = tf.memory() as WebGLMemoryInfo;
    expect(m.numBytesInGPU).toBe(0);

    const a = tf.zeros([SIZE_UPLOAD_UNIFORM - 1]);
    a.square();

    // Only the result lives on the gpu, the input is gone.
    m = tf.memory() as WebGLMemoryInfo;
    expect(m.numBytesInGPU).toBe(a.size * 4);
  });

  it('large tensor gets uploaded to gpu', () => {
    let m = tf.memory() as WebGLMemoryInfo;
    expect(m.numBytesInGPU).toBe(0);

    const a = tf.zeros([SIZE_UPLOAD_UNIFORM + 1]);
    a.square();

    // Both the result and the input live on the gpu.
    m = tf.memory() as WebGLMemoryInfo;
    expect(m.numBytesInGPU).toBe(a.size * 4 * 2);
  });

  it('download and re-upload an output of a shader', async () => {
    const vals = new Float32Array(SIZE_UPLOAD_UNIFORM + 1);
    vals.fill(2);
    const a = tf.square(vals);
    a.dataSync();            // Download to CPU.
    const res = a.square();  // Re-upload to GPU.

    const expected = new Float32Array(SIZE_UPLOAD_UNIFORM + 1);
    expected.fill(16);
    expectArraysClose(await res.data(), expected);
  });
});

describeWithFlags('indexing for large tensors', FLOAT32_WEBGL_ENVS, () => {
  it('properly indexes large tensors', async () => {
    const range = 3000 * 3000;
    const aData = new Float32Array(range);
    for (let i = 0; i < range; i++) {
      aData[i] = i / range;
    }

    const a = tf.tensor1d(aData);
    const aRelu = a.relu();

    expectArraysClose(await a.data(), aData);
    expectArraysClose(await aRelu.data(), aData);
  });
});

describeWithFlags('debug on webgl', WEBGL_ENVS, () => {
  beforeAll(() => {
    // Silences debug warnings.
    spyOn(console, 'warn');
    tf.ENV.set('DEBUG', true);
  });

  afterAll(() => {
    tf.ENV.set('DEBUG', false);
  });

  it('debug mode errors when overflow in tensor construction', () => {
    const savedRenderFloat32Flag =
        tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    const a = () => tf.tensor1d([2, Math.pow(2, 17)], 'float32');
    expect(a).toThrowError();
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });

  it('debug mode errors when underflow in tensor construction', () => {
    const savedRenderFloat32Flag =
        tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    const a = () => tf.tensor1d([2, 1e-8], 'float32');
    expect(a).toThrowError();
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });
});

describeWithFlags('memory webgl', WEBGL_ENVS, () => {
  it('unreliable is falsy/not present when all tensors are numeric', () => {
    tf.tensor(1);
    const mem = tf.memory();
    expect(mem.numTensors).toBe(1);
    expect(mem.numDataBuffers).toBe(1);
    expect(mem.numBytes).toBe(4);
    expect(mem.unreliable).toBeFalsy();
  });
});

// We do not yet fully support half float backends. These tests are a starting
// point.
describeWithFlags('backend without render float32 support', WEBGL_ENVS, () => {
  const savedRenderFloat32Flag = tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED');

  beforeAll(() => {
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
  });

  beforeEach(() => {
    tf.registerBackend('half-float-webgl', () => new MathBackendWebGL(null));
  });

  afterEach(() => {
    tf.removeBackend('half-float-webgl');
  });

  afterAll(() => {
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });

  it('basic usage', async () => {
    tf.setBackend('half-float-webgl');

    const a = tf.tensor2d([1, 2], [1, 2]);
    const b = tf.tensor2d([1, 2], [1, 2]);
    const c = tf.add(a, b);
    expectArraysClose(await c.data(), [2, 4]);
  });

  it('disposing tensors should not cause errors', () => {
    tf.setBackend('half-float-webgl');
    expect(() => tf.tidy(() => {
      const a = tf.tensor2d([1, 2], [1, 2]);
      const b = tf.tensor2d([1, 2], [1, 2]);
      const c = tf.add(a, b);
      c.dataSync();
      return c.add(tf.tensor2d([2, 4], [1, 2]));
    })).not.toThrowError();
  });
});

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
    const time = await tf.time(async () => a.square().data()) as
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

describe('WebGL backend has sync init', () => {
  it('can do matmul without waiting for ready', async () => {
    tf.registerBackend('my-webgl', () => {
      return new MathBackendWebGL();
    });
    const a = tf.tensor1d([5]);
    const b = tf.tensor1d([3]);
    const res = tf.dot(a, b);
    expectArraysClose(await res.data(), 15);
    tf.dispose([a, b, res]);
    tf.removeBackend('my-webgl');
  });
});
