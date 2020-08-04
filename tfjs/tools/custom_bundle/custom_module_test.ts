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

import {getCustomModuleString} from './custom_module';
import {ModuleProvider} from './types';

const mockModuleProvider: ModuleProvider = {
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
};

describe('ESM Module Provider forwardModeOnly=true', () => {
  const forwardModeOnly = true;
  it('one kernel, one backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl'], ['FastBcknd'], forwardModeOnly, mockModuleProvider);

    expect(res).toContain('import CORE');
    expect(res).toContain('import CONVERTER');

    expect(res).toContain('import BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(res).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(res).not.toContain('GRADIENT');
  });

  it('one kernel, two backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl'], ['FastBcknd', 'SlowBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import CORE');
    expect(res).toContain('import CONVERTER');

    expect(res).toContain('import BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(res).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(res).toContain('import BACKEND SlowBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND SlowBcknd');
    expect(res).toContain('registerKernel(MathKrnl_SlowBcknd)');

    expect(res).not.toContain('GRADIENT');
  });

  it('two kernels, one backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl', 'MathKrn2'], ['FastBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import CORE');
    expect(res).toContain('import CONVERTER');

    expect(res).toContain('import BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrn2 from BACKEND FastBcknd');
    expect(res).toContain('registerKernel(MathKrnl_FastBcknd)');
    expect(res).toContain('registerKernel(MathKrn2_FastBcknd)');

    expect(res).not.toContain('GRADIENT');
  });

  it('two kernels, two backends', () => {
    const res = getCustomModuleString(
        ['MathKrnl', 'MathKrn2'], ['FastBcknd', 'SlowBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import CORE');
    expect(res).toContain('import CONVERTER');

    expect(res).toContain('import BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrn2 from BACKEND FastBcknd');
    expect(res).toContain('registerKernel(MathKrnl_FastBcknd)');
    expect(res).toContain('registerKernel(MathKrn2_FastBcknd)');

    expect(res).toContain('import BACKEND SlowBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND SlowBcknd');
    expect(res).toContain('import KERNEL MathKrn2 from BACKEND SlowBcknd');
    expect(res).toContain('registerKernel(MathKrnl_SlowBcknd)');
    expect(res).toContain('registerKernel(MathKrn2_SlowBcknd)');

    expect(res).not.toContain('GRADIENT');
  });
});

describe('ESM Module Provider forwardModeOnly=false', () => {
  const forwardModeOnly = false;

  it('one kernel, one backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl'], ['FastBcknd'], forwardModeOnly, mockModuleProvider);

    expect(res).toContain('import CORE');
    expect(res).toContain('import CONVERTER');

    expect(res).toContain('import BACKEND FastBcknd');
    expect(res).toContain('import KERNEL MathKrnl from BACKEND FastBcknd');
    expect(res).toContain('registerKernel(MathKrnl_FastBcknd)');

    expect(res).toContain('import GRADIENT MathKrnl');
    expect(res).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');
  });

  it('one kernel, two backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl'], ['FastBcknd', 'SlowBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import GRADIENT MathKrnl');
    expect(res).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    const gradIndex = res.indexOf('GRADIENT');
    expect(res.indexOf('GRADIENT', gradIndex + 1))
        .toBe(-1, `Gradient import appears twice in:\n ${res}`);
  });

  it('two kernels, one backend', () => {
    const res = getCustomModuleString(
        ['MathKrnl', 'MathKrn2'], ['FastBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import GRADIENT MathKrnl');
    expect(res).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    expect(res).toContain('import GRADIENT MathKrn2');
    expect(res).toContain('registerGradient(MathKrn2_GRAD_CONFIG)');
  });

  it('two kernels, two backends', () => {
    const res = getCustomModuleString(
        ['MathKrnl', 'MathKrn2'], ['FastBcknd', 'SlowBcknd'], forwardModeOnly,
        mockModuleProvider);

    expect(res).toContain('import GRADIENT MathKrnl');
    expect(res).toContain('registerGradient(MathKrnl_GRAD_CONFIG)');

    expect(res).toContain('import GRADIENT MathKrn2');
    expect(res).toContain('registerGradient(MathKrn2_GRAD_CONFIG)');
  });
});
