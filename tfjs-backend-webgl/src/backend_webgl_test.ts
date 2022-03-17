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

import * as tf from '@tensorflow/tfjs-core';
import {engine, GPUData, test_util, util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

const {expectArraysClose, expectArraysEqual} = test_util;
const {decodeString} = util;

import {getBinaryCache, MathBackendWebGL, WebGLMemoryInfo, WebGLTimingInfo} from './backend_webgl';
import {computeBytes} from './texture_manager';
import {PhysicalTextureType} from './tex_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import {GPGPUContext} from './gpgpu_context';

function decodeStrings(bytes: Uint8Array[]): string[] {
  return bytes.map(b => decodeString(b));
}

const RENDER_FLOAT32_ENVS = {
  flags: {'WEBGL_RENDER_FLOAT32_ENABLED': true},
  predicate: WEBGL_ENVS.predicate
};

describeWithFlags('forced f16 render', RENDER_FLOAT32_ENVS, () => {
  beforeEach(() => {
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', false);
  });
  afterAll(() => tf.env().reset());

  it('should overflow if larger than 66k', async () => {
    const a = tf.tensor1d([Math.pow(2, 17)], 'float32');
    const b = tf.relu(a);
    const data = await b.data();
    if (data && data.length === 1 && !isNaN(data[0])) {  // value could be NaN
      expect(data).toBeLessThan(Math.pow(2, 17));
    }
  });

  it('should error in debug mode', () => {
    // Silence debug warnings.
    spyOn(console, 'warn');

    const debug = tf.env().get('DEBUG');
    tf.env().set('DEBUG', true);
    const a = () => tf.tensor1d([2, Math.pow(2, 17)], 'float32');
    expect(a).toThrowError();
    tf.env().set('DEBUG', debug);
  });
});

describeWithFlags('lazy packing and unpacking', WEBGL_ENVS, () => {
  let webglLazilyUnpackFlagSaved: boolean;
  let webglCpuForwardFlagSaved: boolean;

  beforeAll(() => {
    webglLazilyUnpackFlagSaved = tf.env().getBool('WEBGL_LAZILY_UNPACK');
    webglCpuForwardFlagSaved = tf.env().getBool('WEBGL_CPU_FORWARD');
    tf.env().set('WEBGL_LAZILY_UNPACK', true);
    tf.env().set('WEBGL_CPU_FORWARD', false);
  });

  afterAll(() => {
    tf.env().set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.env().set('WEBGL_CPU_FORWARD', webglCpuForwardFlagSaved);
  });

  it('should not leak memory when lazily unpacking', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    // c is packed to 1x1 RGBA texture.
    const c = tf.matMul(a, b);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;

    const webglPackBinaryOperationsFlagSaved =
        tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS');
    tf.env().set('WEBGL_PACK_BINARY_OPERATIONS', false);
    // Add will unpack c before the operation to 2
    tf.add(c, 1);
    tf.env().set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(tf.memory().numBytes - startNumBytes).toEqual(16);
    expect(tf.memory().numTensors - startNumTensors).toEqual(1);
    // result is unpacked 2x2 R texture.
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU - startNumBytesInGPU)
        .toEqual(4 * tf.util.bytesPerElement(a.dtype));
  });

  it('should not leak memory when lazily packing', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.add(a, 1);

    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const startNumBytesInGPU = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;

    tf.matMul(b, c);

    expect(tf.memory().numBytes - startNumBytes).toEqual(36);
    expect(tf.memory().numTensors - startNumTensors).toEqual(1);
    // result [3, 3] is packed to four RGBA pixel texture b is packed to two
    // RGBA texels texture: total 6 * 4 = 24 components.
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU - startNumBytesInGPU)
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

  it('register string tensor with values', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const t = engine().makeTensor(['a', 'b', 'c'], [3], 'string');
    expectArraysEqual(
        decodeStrings(backend.readSync(t.dataId) as Uint8Array[]),
        ['a', 'b', 'c']);
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
    const t = engine().makeTensor(new Float32Array([1, 2, 3]), [3], 'float32');
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
    tf.copyRegisteredKernels('webgl', 'test-storage');
    tf.setBackend('test-storage');

    const webglPackFlagSaved = tf.env().getBool('WEBGL_PACK');
    tf.env().set('WEBGL_PACK', true);
    const webglSizeUploadUniformSaved =
        tf.env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM');
    tf.env().set('WEBGL_SIZE_UPLOAD_UNIFORM', 0);
    const a = tf.tensor2d([1, 2], [2, 1]);
    const b = tf.tensor2d([1], [1, 1]);
    const c = tf.matMul(a, b);
    backend.readSync(c.dataId);
    tf.env().set('WEBGL_PACK', false);
    const d = tf.add(c, 1);
    tf.env().set('WEBGL_PACK', webglPackFlagSaved);
    tf.env().set('WEBGL_SIZE_UPLOAD_UNIFORM', webglSizeUploadUniformSaved);
    expectArraysClose(await d.data(), [2, 3]);
  });

  it('delayed storage, overwriting', () => {
    const backend = new MathBackendWebGL(null);
    tf.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');

    const texManager = backend.getTextureManager();
    const t = engine().makeTensor(new Float32Array([1, 2, 3]), [3], 'float32');
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(
        backend.readSync(t.dataId) as Float32Array,
        new Float32Array([1, 2, 3]));
    backend.getTexture(t.dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(
        backend.readSync(t.dataId) as Float32Array,
        new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
});

describeWithFlags('Webgl backend disposal', WEBGL_ENVS, () => {
  it('register and dispose a backend outside unit test', () => {
    // Simulate outside unit test environment.
    tf.ENV.set('IS_TEST', false);

    const backend = new MathBackendWebGL();
    tf.registerBackend('test-disposal', () => backend);
    tf.copyRegisteredKernels('webgl', 'test-disposal');
    tf.setBackend('test-disposal');
    // Compile and run a program.
    tf.zeros([1000]).sqrt().dataSync();

    // Dispose the backend.
    tf.backend().dispose();

    // Make sure the cache is empty.
    const cache = getBinaryCache(tf.ENV.getNumber('WEBGL_VERSION'));
    expect(Object.keys(cache).length).toBe(0);
    tf.removeBackend('test-disposal');
  });

  it('register and dispose a backend inside unit test', () => {
    // Simulate inside unit test environment.
    tf.ENV.set('IS_TEST', true);

    const backend = new MathBackendWebGL();
    tf.registerBackend('test-disposal', () => backend);
    tf.copyRegisteredKernels('webgl', 'test-disposal');
    tf.setBackend('test-disposal');
    // Compile and run a program.
    tf.zeros([1000]).sqrt().dataSync();

    // Dispose the backend.
    tf.backend().dispose();

    // Make sure the cache is NOT empty.
    const cache = getBinaryCache(tf.ENV.getNumber('WEBGL_VERSION'));
    expect(Object.keys(cache).length).toBeGreaterThan(0);
    tf.removeBackend('test-disposal');
  });

  it('register, dispose and re-register a backend outside unit test', () => {
    // Simulate outside unit test environment.
    tf.ENV.set('IS_TEST', false);

    tf.registerBackend('test-disposal', () => new MathBackendWebGL());
    tf.copyRegisteredKernels('webgl', 'test-disposal');
    tf.setBackend('test-disposal');
    // Compile and run a program.
    tf.zeros([1000]).sqrt().dataSync();

    // Dispose the backend.
    tf.backend().dispose();
    tf.removeBackend('test-disposal');

    // Re-register a backend.
    tf.registerBackend('test-disposal', () => new MathBackendWebGL());
    tf.copyRegisteredKernels('webgl', 'test-disposal');
    tf.setBackend('test-disposal');
    // Compile and run a program.
    tf.zeros([1000]).sqrt().dataSync();

    // Dispose the 2nd backend.
    tf.backend().dispose();

    // Make sure the cache is empty.
    const cache = getBinaryCache(tf.ENV.getNumber('WEBGL_VERSION'));
    expect(Object.keys(cache).length).toBe(0);
    tf.removeBackend('test-disposal');
  });
});

describeWithFlags('Custom window size', WEBGL_ENVS, () => {
  const customBackendName = 'custom-webgl';

  beforeAll(() => {
    const kernelFunc = tf.getKernel('Square', 'webgl').kernelFunc;
    tf.registerKernel(
        {kernelName: 'Square', backendName: customBackendName, kernelFunc});
  });

  afterAll(() => {
    tf.unregisterKernel('Square', customBackendName);
  });

  it('Set screen area to be 1x1', () => {
    // This will set the screen size to 1x1 to make sure the page limit is
    // very small.
    spyOnProperty(window, 'screen', 'get')
        .and.returnValue({height: 1, width: 1});

    tf.registerBackend(customBackendName, () => new MathBackendWebGL());
    tf.setBackend(customBackendName);

    // Allocate ~40KB.
    const a = tf.ones([100, 100]);
    // No gpu memory used yet because of delayed storage.
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU).toBe(0);

    // Expect console.warn() to be called.
    let numWarnCalls = 0;
    spyOn(console, 'warn').and.callFake(() => {
      numWarnCalls++;
    });

    a.square();
    expect(numWarnCalls).toBe(1);
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU)
        .toBe(100 * 100 * 4 * 2);

    // Allocate another 40KB.
    a.square();

    // Expect console.warn() to NOT be called more than once.
    expect(numWarnCalls).toBe(1);
    expect((tf.memory() as WebGLMemoryInfo).numBytesInGPU)
        .toBe(100 * 100 * 4 * 3);
    tf.removeBackend(customBackendName);
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
    savedUploadUniformValue =
        tf.env().get('WEBGL_SIZE_UPLOAD_UNIFORM') as number;
    tf.env().set('WEBGL_SIZE_UPLOAD_UNIFORM', SIZE_UPLOAD_UNIFORM);
  });

  afterAll(() => {
    tf.env().set('WEBGL_SIZE_UPLOAD_UNIFORM', savedUploadUniformValue);
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
    tf.enableDebugMode();
  });

  it('debug mode errors when overflow in tensor construction', () => {
    const savedRenderFloat32Flag =
        tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED');
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    const a = () => tf.tensor1d([2, Math.pow(2, 17)], 'float32');
    expect(a).toThrowError();
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });

  it('debug mode errors when underflow in tensor construction', () => {
    const savedRenderFloat32Flag =
        tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED');
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    const a = () => tf.tensor1d([2, 1e-8], 'float32');
    expect(a).toThrowError();
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });
});

const WEBGL1_ENVS = {
  flags: {'WEBGL_VERSION': 1},
  predicate: WEBGL_ENVS.predicate
};

const WEBGL2_ENVS = {
  flags: {'WEBGL_VERSION': 2},
  predicate: WEBGL_ENVS.predicate
};

describeWithFlags('computeBytes counts bytes correctly', WEBGL1_ENVS, () => {
  it('for all physical texture types', () => {
    const gpgpu = new GPGPUContext();

    const shapeRC: [number, number] = [2, 3];

    let bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.UNPACKED_FLOAT16, gpgpu.gl,
        gpgpu.textureConfig, false /* isPacked */);
    expect(bytesForTex).toBe(96);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.UNPACKED_FLOAT32, gpgpu.gl,
        gpgpu.textureConfig, false /* isPacked */);
    expect(bytesForTex).toBe(96);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(32);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_2X2_FLOAT32, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(32);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_2X2_FLOAT16, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(32);

    gpgpu.dispose();
  });
});

describeWithFlags('computeBytes counts bytes correctly', WEBGL2_ENVS, () => {
  it('test every physical tex type input to computeBytes', () => {
    const gpgpu = new GPGPUContext();

    const shapeRC: [number, number] = [2, 3];

    let bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.UNPACKED_FLOAT16, gpgpu.gl,
        gpgpu.textureConfig, false /* isPacked */);
    expect(bytesForTex).toBe(12);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.UNPACKED_FLOAT32, gpgpu.gl,
        gpgpu.textureConfig, false /* isPacked */);
    expect(bytesForTex).toBe(24);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(8);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_2X2_FLOAT32, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(32);

    bytesForTex = computeBytes(
        shapeRC, PhysicalTextureType.PACKED_2X2_FLOAT16, gpgpu.gl,
        gpgpu.textureConfig, true /* isPacked */);
    expect(bytesForTex).toBe(16);

    gpgpu.dispose();
  });
});

describeWithFlags('aggressive texture deletion', WEBGL_ENVS, () => {
  it('basic', () => {
    const savedDeleteThreshold =
        tf.env().get('WEBGL_DELETE_TEXTURE_THRESHOLD') as number;
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);

    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    tf.matMul(a, b);

    const startNumBytesAllocated =
        (tf.memory() as WebGLMemoryInfo).numBytesInGPUAllocated;

    a.dispose();
    b.dispose();

    expect(
        startNumBytesAllocated -
        (tf.memory() as WebGLMemoryInfo).numBytesInGPUAllocated)
        .toBeGreaterThan(0);

    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', savedDeleteThreshold);
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

describeWithFlags('manual gl flush', WEBGL_ENVS, () => {
  it('works when manual gl flush is enabled', async () => {
    const savedGlThreshold = tf.env().get('WEBGL_FLUSH_THRESHOLD') as number;
    tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([1, 1, -3, 2, 2, 1], [2, 3]);

    const result = tf.div(tf.div(tf.mul(a, b), a), b);
    expectArraysClose(await result.data(), [1, 1, 1, 1, 1, 1]);
    tf.env().set('WEBGL_FLUSH_THRESHOLD', savedGlThreshold);
  });
});
// We do not yet fully support half float backends. These tests are a starting
// point.
describeWithFlags('backend without render float32 support', WEBGL_ENVS, () => {
  const savedRenderFloat32Flag =
      tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED');
  const customWebGLBackendName = 'half-float-webgl';

  beforeAll(() => {
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', false);
  });

  beforeEach(() => {
    tf.copyRegisteredKernels('webgl', customWebGLBackendName);
    tf.registerBackend(
        customWebGLBackendName, () => new MathBackendWebGL(null));
  });

  afterEach(() => {
    tf.removeBackend(customWebGLBackendName);
  });

  afterAll(() => {
    tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', savedRenderFloat32Flag);
  });

  it('basic usage', async () => {
    tf.setBackend(customWebGLBackendName);

    const a = tf.tensor2d([1, 2], [1, 2]);
    const b = tf.tensor2d([1, 2], [1, 2]);
    const c = tf.add(a, b);
    expectArraysClose(await c.data(), [2, 4]);
  });

  it('disposing tensors should not cause errors', () => {
    tf.setBackend(customWebGLBackendName);
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
    const time = await tf.time(() => a.square()) as WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + dataSync', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square().dataSync()) as WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    const time =
        await tf.time(async () => a.square().data()) as WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('preupload (not included) + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    // Pre-upload a on gpu.
    a.square();
    const time = await tf.time(() => a.sqrt()) as WebGLTimingInfo;
    // The tensor was already on gpu.
    expect(time.uploadWaitMs === 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('returns error for kernelMs if query timer extension is unavailable',
     async () => {
       const savedQueryReliableValue =
           tf.env().get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE');
       tf.env().set('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', false);

       const a = tf.zeros([10, 10]);
       const time = await tf.backend().time(() => a.sqrt()) as WebGLTimingInfo;
       expect(time.kernelMs).toEqual({
         error: 'WebGL query timers are not supported in this environment.'
       });

       tf.env().set(
           'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE',
           savedQueryReliableValue);
     });
});

describeWithFlags('keeping data on gpu ', WEBGL2_ENVS, () => {
  let flag: boolean;

  beforeAll(() => {
    flag = tf.env().getBool('WEBGL_CPU_FORWARD');
    tf.env().set('WEBGL_CPU_FORWARD', false);
  });

  afterAll(() => {
    tf.env().set('WEBGL_CPU_FORWARD', flag);
  });

  it('has a valid texture for dtype=float32.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const texShape = [2, 2];
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);
    const res = b.dataToGPU();
    expectArraysEqual(res.texShape, texShape);

    const webGLBackend = tf.backend() as MathBackendWebGL;
    const buffer = webGLBackend.gpgpu.createBufferFromTexture(
        res.texture, res.texShape[0], res.texShape[1]);
    const vals = webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 12);
    expectArraysEqual(vals, data);
  });

  it('uses user defined texShape.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);
    const texShape = [1, 3] as [number, number];
    const res = b.dataToGPU({customTexShape: texShape});
    expectArraysEqual(res.texShape, texShape);

    const webGLBackend = tf.backend() as MathBackendWebGL;
    const buffer = webGLBackend.gpgpu.createBufferFromTexture(
        res.texture, res.texShape[0], res.texShape[1]);
    const vals = webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 12);
    expectArraysEqual(vals, data);
  });

  it('has a valid texture for dtype=int32.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const texShape = [2, 2];
    const a = tf.tensor(data, [1, 3, 4], 'int32');
    const b = tf.add(a, 0);
    const res = b.dataToGPU();
    expectArraysEqual(res.texShape, texShape);

    const webGLBackend = tf.backend() as MathBackendWebGL;

    const buffer = webGLBackend.gpgpu.createBufferFromTexture(
        res.texture, res.texShape[0], res.texShape[1]);
    const vals = webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 12);

    expectArraysEqual(vals, data);
  });

  it('has no memory leak.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);

    const webGLBackend = tf.backend() as MathBackendWebGL;
    const startNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGLBackend.numDataIds();

    const res = b.dataToGPU();
    res.tensorRef.dispose();

    const endNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGLBackend.numDataIds();

    expect(endNumBytes).toEqual(startNumBytes);
    expect(endTensor).toEqual(startTensor);
    expect(endDataBuckets).toEqual(startDataBuckets);
  });

  it('can be used in tidy.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const webGLBackend = tf.backend() as MathBackendWebGL;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGLBackend.numDataIds();

    const result = tf.tidy(() => {
      const a = tf.tensor(data, [1, 3, 4]);
      const b = tf.add(a, 0);
      return b.dataToGPU() as {} as tf.Tensor;
    });

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGLBackend.numDataIds();

    expect(endTensor).toEqual(startTensor + 1);
    expect(endDataBuckets).toEqual(startDataBuckets + 1);

    const res = result as {} as GPUData;
    const buffer = webGLBackend.gpgpu.createBufferFromTexture(
        res.texture, res.texShape[0], res.texShape[1]);
    const vals = webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 12);
    expectArraysEqual(vals, data);
  });

  it('tidy has no memory leak.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const webGLBackend = tf.backend();
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGLBackend.numDataIds();

    tf.tidy(() => {
      const a = tf.tensor(data, [1, 3, 4]);
      const b = tf.add(a, 0);
      b.dataToGPU();
      return b;
    });

    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGLBackend.numDataIds();

    expect(endTensor).toEqual(startTensor + 1);
    expect(endDataBuckets).toEqual(startDataBuckets + 1);
  });

  it('throws error when user defined texShape is too small.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);

    expect(() => {
      b.dataToGPU({customTexShape: [1, 1]});
    }).toThrowError();
  });
});

describeWithFlags('keeping data on gpu ', WEBGL1_ENVS, () => {
  let flag: boolean;
  const webGLBackend = (tf.backend() as MathBackendWebGL);

  beforeAll(() => {
    flag = tf.env().getBool('WEBGL_CPU_FORWARD');
    tf.env().set('WEBGL_CPU_FORWARD', false);
  });

  afterAll(() => {
    tf.env().set('WEBGL_CPU_FORWARD', flag);
  });

  it('has a valid texture for dtype=float32.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const texShape = [2, 2];
    const size = 12;
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);
    const res = b.dataToGPU();
    expectArraysEqual(res.texShape, texShape);

    if (tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      const tmpData = webGLBackend.texData.get(res.tensorRef.dataId);
      const vals = webGLBackend.gpgpu
                       .downloadMatrixFromPackedTexture(
                           tmpData.texture.texture, ...tmpData.texture.texShape)
                       .subarray(0, size);
      expectArraysEqual(vals, data);
    }
  });

  it('uses user defined texShape.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);
    const texShape = [1, 3] as [number, number];
    const size = 12;
    const res = b.dataToGPU({customTexShape: texShape});
    expectArraysEqual(res.texShape, texShape);

    if (tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      const tmpData = webGLBackend.texData.get(res.tensorRef.dataId);
      const vals = webGLBackend.gpgpu
                       .downloadMatrixFromPackedTexture(
                           tmpData.texture.texture, ...tmpData.texture.texShape)
                       .subarray(0, size);
      expectArraysEqual(vals, data);
    }
  });

  it('has a valid texture for dtype=int32.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const texShape = [2, 2];
    const size = 12;
    const a = tf.tensor(data, [1, 3, 4], 'int32');
    const b = tf.add(a, 0);
    const res = b.dataToGPU();
    expectArraysEqual(res.texShape, texShape);

    if (tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
      const tmpData = webGLBackend.texData.get(res.tensorRef.dataId);
      const vals = webGLBackend.gpgpu
                       .downloadMatrixFromPackedTexture(
                           tmpData.texture.texture, ...tmpData.texture.texShape)
                       .subarray(0, size);
      expectArraysEqual(vals, data);
    }
  });

  it('has no memory leak.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);

    const webGLBackend = tf.backend() as MathBackendWebGL;
    const startNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGLBackend.numDataIds();

    const res = b.dataToGPU();
    res.tensorRef.dispose();

    const endNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGLBackend.numDataIds();

    expect(endNumBytes).toEqual(startNumBytes);
    expect(endTensor).toEqual(startTensor);
    expect(endDataBuckets).toEqual(startDataBuckets);
  });

  it('throws error when user defined texShape is too small.', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const a = tf.tensor(data, [1, 3, 4]);
    const b = tf.add(a, 0);

    expect(() => {
      b.dataToGPU({customTexShape: [1, 1]});
    }).toThrowError();
  });
});

describeWithFlags('caching on cpu', WEBGL_ENVS, () => {
  const customBackendName = 'cache-on-cpu';

  beforeAll(() => {
    tf.env().set('WEBGL_CPU_FORWARD', false);
    const kernelFunc = tf.getKernel('Square', 'webgl').kernelFunc;
    tf.registerKernel(
        {kernelName: 'Square', backendName: customBackendName, kernelFunc});
  });

  afterAll(() => {
    tf.unregisterKernel('Square', customBackendName);
  });

  it('caches on cpu after async read', async () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend(customBackendName, () => backend);
    tf.setBackend(customBackendName);

    const t = tf.square(2);
    const info = backend.getDataInfo(t.dataId);

    // Make sure the tensor is on the GPU.
    expect(info.values == null).toBe(true);

    await t.data();

    // Make sure the tensor is cached on CPU.
    expect(info.values).not.toBe(null);

    tf.removeBackend(customBackendName);
  });

  it('caches on cpu after sync read', () => {
    const backend = new MathBackendWebGL();
    tf.registerBackend(customBackendName, () => backend);
    tf.setBackend(customBackendName);

    const t = tf.square(2);
    const info = backend.getDataInfo(t.dataId);

    // Make sure the tensor is on the GPU.
    expect(info.values == null).toBe(true);

    t.dataSync();

    // Make sure the tensor is cached on CPU.
    expect(info.values).not.toBe(null);

    tf.removeBackend(customBackendName);
  });
});

describeWithFlags('WebGL backend has sync init', WEBGL_ENVS, () => {
  it('can do matmul without waiting for ready', async () => {
    const customWebGLBackendName = 'my-webgl';

    tf.copyRegisteredKernels('webgl', customWebGLBackendName);

    tf.registerBackend(customWebGLBackendName, () => {
      return new MathBackendWebGL();
    });
    tf.setBackend(customWebGLBackendName);
    const a = tf.tensor1d([5]);
    const b = tf.tensor1d([3]);
    const res = tf.dot(a, b);
    expectArraysClose(await res.data(), 15);
    tf.dispose([a, b, res]);
    tf.removeBackend(customWebGLBackendName);
  });
});

describeWithFlags('custom canvas ', WEBGL_ENVS, () => {
  const customBackendName = 'custom-webgl';
  let flag: boolean;

  beforeAll(() => {
    flag = tf.env().getBool('WEBGL_CPU_FORWARD');
    tf.env().set('WEBGL_CPU_FORWARD', false);
    const kernelFunc = tf.getKernel('Square', 'webgl').kernelFunc;
    tf.registerKernel(
        {kernelName: 'Square', backendName: customBackendName, kernelFunc});
  });

  afterAll(() => {
    tf.env().set('WEBGL_CPU_FORWARD', flag);
    tf.unregisterKernel('Square', customBackendName);
  });

  it('works.', () => {
    const customCanvas = document.createElement('canvas');
    customCanvas.width = 300;
    customCanvas.height = 200;

    const backend = new MathBackendWebGL(customCanvas);
    tf.registerBackend(customBackendName, () => backend);
    tf.setBackend(customBackendName);

    const t = tf.square(2);
    const data = t.dataSync();

    expectArraysEqual(data, [4]);

    tf.removeBackend(customBackendName);
  });
});
describeWithFlags('Parallel compilation', WEBGL_ENVS, () => {
  // TODO(lina128): Also test async after parallel compilation flag is
  // implemented in context object. We have to keep the test sync for now,
  // because it's a global flag, the async test will affect other tests.
  it('does not have memory leak.', () => {
    const savedWebGLCPUForward = tf.env().get('WEBGL_CPU_FORWARD');
    tf.env().set('WEBGL_CPU_FORWARD', false);

    const customWebGLBackendName = 'my-webgl';
    tf.copyRegisteredKernels('webgl', customWebGLBackendName);
    tf.registerBackend(customWebGLBackendName, () => {
      return new MathBackendWebGL();
    });
    tf.setBackend(customWebGLBackendName);

    const a0 = tf.tensor1d([1, 1, 1]);
    const b0 = tf.tensor1d([1, 1, 1]);
    const c0 = tf.add(a0, b0);
    const data = c0.dataSync();

    expectArraysClose(data, [2, 2, 2]);
    tf.dispose([a0, b0, c0]);
    tf.removeBackend(customWebGLBackendName);

    // TODO(lina128): Also test use an existing backend after parallel
    // compilation flag is implemented in context object. The current approach
    // assumes there's no binary cache, and it doesn't check existing cache.
    const customWebGLBackendName1 = 'my-webgl1';
    tf.copyRegisteredKernels('webgl', customWebGLBackendName1);
    tf.registerBackend(customWebGLBackendName1, () => {
      return new MathBackendWebGL();
    });
    tf.setBackend(customWebGLBackendName1);
    const webGLBackend = tf.backend() as MathBackendWebGL;

    const startNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const startTensor = tf.memory().numTensors;
    const startDataBuckets = webGLBackend.numDataIds();

    const a1 = tf.tensor1d([1, 1, 1]);
    const b1 = tf.tensor1d([1, 1, 1]);

    // Pre-compile round.
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    const c1 = tf.add(a1, b1);
    webGLBackend.checkCompileCompletion();
    webGLBackend.getUniformLocations();

    // Warm-up upload and download round.
    tf.env().set('ENGINE_COMPILE_ONLY', false);
    const c2 = tf.add(a1, b1);
    c2.dataSync();

    // Actual inference.
    const c3 = tf.add(a1, b1);
    expectArraysEqual(c3.dataSync(), [2, 2, 2]);

    tf.dispose([a1, b1, c1, c2, c3]);
    const endNumBytes = (tf.memory() as WebGLMemoryInfo).numBytesInGPU;
    const endTensor = tf.memory().numTensors;
    const endDataBuckets = webGLBackend.numDataIds();

    // We only check numBytesInGPU. For parallel compilation,
    // numBytesInGPUAllocated will be more because of the two pass uploadToGPU,
    // but they will all be freed, resulting in endNumbytes equal to
    // startNumBytes.
    expect(startNumBytes).toEqual(endNumBytes);
    expect(startTensor).toEqual(endTensor);
    expect(endDataBuckets).toEqual(startDataBuckets);

    tf.removeBackend(customWebGLBackendName1);

    tf.env().set('WEBGL_CPU_FORWARD', savedWebGLCPUForward);
  });
});
