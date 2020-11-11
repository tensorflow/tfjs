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

import {getCustomConverterOpsModule, getCustomModuleString} from './custom_module';
import {CustomTFJSBundleConfig, ImportProvider} from './types';

const mockImportProvider: ImportProvider = {
  importCoreStr: () => 'import CORE',
  importConverterStr: () => 'import CONVERTER',
  importBackendStr: (name: string) => `import BACKEND ${name}`,
  importKernelStr: (kernelName: string, backend: string) => ({
    importStatement: `import KERNEL ${kernelName} from BACKEND ${backend}`,
    kernelConfigId: `${kernelName}_${backend}`
  }),
  importGradientConfigStr: (kernel: string) => ({
    importStatement: `import GRADIENT ${kernel}`,
    gradConfigId: `${kernel}_GRAD_CONFIG`,
  }),
  importOpForConverterStr: (opSymbol: string) => {
    return `export * from ${opSymbol}`;
  },
  importNamespacedOpsForConverterStr: (
      namespace: string, opSymbols: string[]) => {
    return `export ${opSymbols.join(',')} as ${namespace} from ${namespace}/`;
  }
};

describe('getCustomModuleString forwardModeOnly=true', () => {
  const forwardModeOnly = true;
  it('one kernel, one backend', () => {
    const config = {
      kernels: ['MathKrnl'],
      backends: ['FastBcknd'],
      models: [] as string[],
      forwardModeOnly
    };
    const {tfjs, core} = getCustomModuleString(
        // cast because FastBcknd is not a valid backend per the type
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(core).toContain('import CORE');
    expect(tfjs).toContain('import CORE');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(tfjs).not.toContain('GRADIENT');
  });

  it('one kernel, one backend, one model', () => {
    const config = {
      kernels: ['MathKrnl'],
      backends: ['FastBcknd'],
      models: ['model1.json'],
      forwardModeOnly
    };
    const {tfjs, core} = getCustomModuleString(
        // cast because FastBcknd is not a valid backend per the type
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(core).toContain('import CORE');
    expect(tfjs).toContain('import CORE');
    expect(tfjs).toContain('import CONVERTER');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(tfjs).not.toContain('GRADIENT');
  });

  it('one kernel, two backend', () => {
    const config = {
      kernels: ['MathKrnl'],
      backends: ['FastBcknd', 'SlowBcknd'],
      models: [] as string[],
      forwardModeOnly
    };

    const {tfjs} = getCustomModuleString(
        // cast because the backends are not truly valid backend per the type
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import CORE');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(tfjs).toContain('import BACKEND SlowBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND SlowBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_SlowBcknd)');

    expect(tfjs).not.toContain('GRADIENT');
  });

  it('two kernels, one backend', () => {
    const config = {
      kernels: ['MathKrnl', 'MathKrn2'],
      backends: ['FastBcknd'],
      models: [] as string[],
      forwardModeOnly
    };
    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import CORE');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrn2 from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');
    expect(tfjs).toContain('registerKernel(MathKrn2_FastBcknd)');

    expect(tfjs).not.toContain('GRADIENT');
  });

  it('two kernels, two backends', () => {
    const config = {
      kernels: ['MathKrnl', 'MathKrn2'],
      backends: ['FastBcknd', 'SlowBcknd'],
      models: [] as string[],
      forwardModeOnly
    };
    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import CORE');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrn2 from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');
    expect(tfjs).toContain('registerKernel(MathKrn2_FastBcknd)');

    expect(tfjs).toContain('import BACKEND SlowBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND SlowBcknd');
    expect(tfjs).toContain('import KERNEL MathKrn2 from BACKEND SlowBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_SlowBcknd)');
    expect(tfjs).toContain('registerKernel(MathKrn2_SlowBcknd)');

    expect(tfjs).not.toContain('GRADIENT');
  });
});

describe('getCustomModuleString forwardModeOnly=false', () => {
  const forwardModeOnly = false;

  it('one kernel, one backend', () => {
    const config = {
      kernels: ['MathKrnl'],
      backends: ['FastBcknd'],
      models: [] as string[],
      forwardModeOnly
    };

    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import CORE');

    expect(tfjs).toContain('import BACKEND FastBcknd');
    expect(tfjs).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(tfjs).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(tfjs).toContain('import GRADIENT MathKrnl');
    expect(tfjs).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');
  });

  it('one kernel, two backend', () => {
    const config = {
      kernels: ['MathKrnl'],
      backends: ['FastBcknd', 'SlowBcknd'],
      models: [] as string[],
      forwardModeOnly
    };

    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import GRADIENT MathKrnl');
    expect(tfjs).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    const gradIndex = tfjs.indexOf('GRADIENT');
    expect(tfjs.indexOf('GRADIENT', gradIndex + 1))
        .toBe(-1, `Gradient import appears twice in:\n ${tfjs}`);
  });

  it('two kernels, one backend', () => {
    const config = {
      kernels: ['MathKrnl', 'MathKrn2'],
      backends: ['FastBcknd'],
      models: [] as string[],
      forwardModeOnly
    };

    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import GRADIENT MathKrnl');
    expect(tfjs).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    expect(tfjs).toContain('import GRADIENT MathKrn2');
    expect(tfjs).toContain('registerGradient(MathKrn2_GRAD_CONFIG)');
  });

  it('two kernels, two backends', () => {
    const config = {
      kernels: ['MathKrnl', 'MathKrn2'],
      backends: ['FastBcknd', 'SlowBcknd'],
      models: [] as string[],
      forwardModeOnly
    };

    const {tfjs} = getCustomModuleString(
        config as CustomTFJSBundleConfig, mockImportProvider);

    expect(tfjs).toContain('import GRADIENT MathKrnl');
    expect(tfjs).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    expect(tfjs).toContain('import GRADIENT MathKrn2');
    expect(tfjs).toContain('registerGradient(MathKrn2_GRAD_CONFIG)');
  });
});

describe('getCustomConverterOpsModule', () => {
  it('non namespaced ops', () => {
    const result =
        getCustomConverterOpsModule(['add', 'sub'], mockImportProvider);

    expect(result).toContain('export * from add');
    expect(result).toContain('export * from sub');
  });

  it('namespaced ops', () => {
    const result = getCustomConverterOpsModule(
        ['image.resizeBilinear', 'image.resizeNearestNeighbor'],
        mockImportProvider);

    expect(result).toContain(
        'export resizeBilinear,resizeNearestNeighbor as image from image/');
  });
});
