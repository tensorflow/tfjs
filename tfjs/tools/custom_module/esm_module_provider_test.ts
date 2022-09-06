/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {esmImportProvider} from './esm_module_provider';

describe('ESM Module Provider', () => {
  it('importCoreStr forwardModeOnly=true', () => {
    const forwardModeOnly = true;
    const res = esmImportProvider.importCoreStr(forwardModeOnly);
    expect(res).toContain(
        `import {registerKernel} from '@tensorflow/tfjs-core/dist/base'`);
    expect(res).not.toContain(
        `import {registerGradient} from '@tensorflow/tfjs-core/dist/base'`);
    expect(res).toContain(
        `import '@tensorflow/tfjs-core/dist/base_side_effects';`);
    expect(res).toContain(`export * from '@tensorflow/tfjs-core/dist/base';`);
  });

  it('importCoreStr forwardModeOnly=false', () => {
    const forwardModeOnly = false;
    const res = esmImportProvider.importCoreStr(forwardModeOnly);
    expect(res).toContain(
        `import {registerKernel} from '@tensorflow/tfjs-core/dist/base'`);
    expect(res).toContain(
        `import {registerGradient} from '@tensorflow/tfjs-core/dist/base'`);
    expect(res).toContain(
        `import '@tensorflow/tfjs-core/dist/base_side_effects';`);
    expect(res).toContain(`export * from '@tensorflow/tfjs-core/dist/base';`);
  });

  it('importConverterStr', () => {
    const res = esmImportProvider.importConverterStr();
    expect(res).toBe(`export * from '@tensorflow/tfjs-converter';`);
  });

  it('importBackendStr cpu', () => {
    const res = esmImportProvider.importBackendStr('cpu');
    expect(res).toBe(`export * from '@tensorflow/tfjs-backend-cpu/dist/base';`);
  });

  it('importBackendStr webgl', () => {
    const res = esmImportProvider.importBackendStr('webgl');
    expect(res).toBe(
        `export * from '@tensorflow/tfjs-backend-webgl/dist/base';`);
  });

  it('importKernelStr Max cpu', () => {
    const res = esmImportProvider.importKernelStr('Max', 'cpu');
    expect(res.importStatement).toContain('import {maxConfig as Max_cpu}');
    expect(res.importStatement)
        .toContain(`from '@tensorflow/tfjs-backend-cpu/dist/kernels/Max'`);

    expect(res.kernelConfigId).toBe('Max_cpu');
  });

  it('importGradientConfigStr Max', () => {
    const res = esmImportProvider.importGradientConfigStr('Max');
    expect(res.importStatement).toContain('import {maxGradConfig}');
    expect(res.importStatement)
        .toContain(`from '@tensorflow/tfjs-core/dist/gradients/Max_grad'`);

    expect(res.gradConfigId).toBe('maxGradConfig');
  });

  it('importGradientConfigStr Max', () => {
    const res = esmImportProvider.importGradientConfigStr('Max');
    expect(res.importStatement).toContain('import {maxGradConfig}');
    expect(res.importStatement)
        .toContain(`from '@tensorflow/tfjs-core/dist/gradients/Max_grad'`);

    expect(res.gradConfigId).toBe('maxGradConfig');
  });

  it('importOpForConverterStr add', () => {
    const res = esmImportProvider.importOpForConverterStr('add');
    expect(res).toBe(`export {add} from '@tensorflow/tfjs-core/dist/ops/add';`);
  });

  it('importOpForConverterStr gatherND', () => {
    const res = esmImportProvider.importOpForConverterStr('gatherND');
    expect(res).toBe(
        `export {gatherND} from '@tensorflow/tfjs-core/dist/ops/gather_nd';`);
  });

  it('importOpForConverterStr batchToSpaceND', () => {
    const res = esmImportProvider.importOpForConverterStr('batchToSpaceND');
    expect(res).toBe(
        // tslint:disable-next-line:max-line-length
        `export {batchToSpaceND} from '@tensorflow/tfjs-core/dist/ops/batch_to_space_nd';`);
  });

  it('importOpForConverterStr concat1d', () => {
    const res = esmImportProvider.importOpForConverterStr('concat1d');
    expect(res).toBe(
        `export {concat1d} from '@tensorflow/tfjs-core/dist/ops/concat_1d';`);
  });

  it('importOpForConverterStr avgPool3d', () => {
    const res = esmImportProvider.importOpForConverterStr('avgPool3d');
    expect(res).toBe(
        // tslint:disable-next-line:max-line-length
        `export {avgPool3d} from '@tensorflow/tfjs-core/dist/ops/avg_pool_3d';`);
  });

  it('importOpForConverterStr stridedSlice', () => {
    const res = esmImportProvider.importOpForConverterStr('stridedSlice');
    expect(res).toBe(
        // tslint:disable-next-line: max-line-length
        `export {stridedSlice} from '@tensorflow/tfjs-core/dist/ops/strided_slice';`);
  });

  it('importNamespacedOpsForConverterStr image.resizeBilinear', () => {
    const res = esmImportProvider.importNamespacedOpsForConverterStr(
        'image', ['resizeBilinear']);
    expect(res).toBe(
        // tslint:disable-next-line: max-line-length
        `import {resizeBilinear as resizeBilinear_image} from '@tensorflow/tfjs-core/dist/ops/image/resize_bilinear';
export const image = {
\tresizeBilinear: resizeBilinear_image,
};`);
  });

  it('importNamespacedOpsForConverterStr two ops in namespace', () => {
    const res = esmImportProvider.importNamespacedOpsForConverterStr(
        'image', ['resizeBilinear', 'resizeNearestNeighbor']);
    expect(res).toBe(
        // tslint:disable-next-line: max-line-length
        `import {resizeBilinear as resizeBilinear_image} from '@tensorflow/tfjs-core/dist/ops/image/resize_bilinear';
import {resizeNearestNeighbor as resizeNearestNeighbor_image} from '@tensorflow/tfjs-core/dist/ops/image/resize_nearest_neighbor';
export const image = {
\tresizeBilinear: resizeBilinear_image,
\tresizeNearestNeighbor: resizeNearestNeighbor_image,
};`);
  });
});
