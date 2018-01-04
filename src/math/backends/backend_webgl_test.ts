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

import * as test_util from '../../test_util';
import {Tests} from '../../test_util';
import {MathBackendWebGL} from './backend_webgl';

const tests: Tests = () => {
  it('delayed storage, reading', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    backend.register(0, [3], 'float32');
    backend.write(0, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(0);
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(0), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(0);
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(0);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('delayed storage, overwriting', () => {
    const delayedStorage = true;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    backend.register(0, [3], 'float32');
    backend.write(0, new Float32Array([1, 2, 3]));
    backend.getTexture(0);
    expect(texManager.getNumUsedTextures()).toBe(1);
    // overwrite.
    backend.write(0, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    test_util.expectArraysClose(
        backend.readSync(0), new Float32Array([4, 5, 6]));
    backend.getTexture(0);
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(0), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage reading', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    backend.register(0, [3], 'float32');
    backend.write(0, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(0), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(0);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('immediate storage overwriting', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    backend.register(0, [3], 'float32');
    backend.write(0, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.write(0, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(0), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(0);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it('disposal of backend disposes all textures', () => {
    const delayedStorage = false;
    const backend = new MathBackendWebGL(null, delayedStorage);
    const texManager = backend.getTextureManager();
    backend.register(0, [3], 'float32');
    backend.write(0, new Float32Array([1, 2, 3]));
    backend.register(1, [3], 'float32');
    backend.write(1, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(2);
    backend.dispose();
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
};

test_util.describeCustom('backend_webgl', tests, [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
