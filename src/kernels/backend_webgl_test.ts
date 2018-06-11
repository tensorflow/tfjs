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

// tslint:disable-next-line:max-line-length
import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual, WEBGL_ENVS} from '../test_util';
// tslint:disable-next-line:max-line-length
import {MathBackendWebGL, SIZE_UPLOAD_UNIFORM, WebGLMemoryInfo} from './backend_webgl';

describeWithFlags('backendWebGL', WEBGL_ENVS, () => {
  it('delayed storage, reading', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('delayed storage, overwriting', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    // overwrite.
    backend.write(dataId, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    backend.getTexture(dataId);
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage reading', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage overwriting', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.write(dataId, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    expectArraysClose(backend.readSync(dataId), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(dataId);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('disposal of backend disposes all textures', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    const dataId = {};
    backend.register(dataId, [3], 'float32');
    backend.write(dataId, new Float32Array([1, 2, 3]));
    const dataId2 = {};
    backend.register(dataId2, [3], 'float32');
    backend.write(dataId2, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(2);
    backend.dispose();
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
});

describeWithFlags('Custom window size', WEBGL_ENVS, () => {
  it('Set screen area to be 1x1', async () => {
    // This will set the screen size to 1x1 to make sure the page limit is
    // very small.
    spyOnProperty(window, 'screen', 'get')
        .and.returnValue({height: 1, width: 1});
    const oldBackend = tf.getBackend();

    tf.ENV.registerBackend('custom-webgl', () => new MathBackendWebGL());
    tf.setBackend('custom-webgl');

    // Allocate a 100x100 tensor.
    const a = tf.ones([100, 100]);
    // No gpu memory used yet because of delayed storage.
    expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU).toBe(0);

    await a.square().data();
    // Everything got paged out of gpu after the run finished.
    expect((tf.memory() as tf.webgl.WebGLMemoryInfo).numBytesInGPU).toBe(0);

    expectArraysEqual(a, new Float32Array(100 * 100).fill(1));
    tf.setBackend(oldBackend);
    tf.ENV.removeBackend('custom-webgl');
  });
});

describeWithFlags('upload tensors as uniforms', WEBGL_ENVS, () => {
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
});
