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

import {ModuleProvider} from './types';

export function getCustomModuleString(
    kernels: string[],
    backends: string[],
    forwardModeOnly: boolean,
    moduleProvider: ModuleProvider,
    ): string {
  const result: string[] = [];

  addLine(result, moduleProvider.importCoreStr());
  addLine(result, moduleProvider.importConverterStr());

  for (const backend of backends) {
    addLine(result, `\n//backend = ${backend}`);
    addLine(result, moduleProvider.importBackendStr(backend));
    for (const kernelName of kernels) {
      const kernelImport = moduleProvider.importKernelStr(kernelName, backend);
      addLine(result, kernelImport.importStatement);
      addLine(result, registerKernelStr(kernelImport.kernelConfigId));
    }
  }

  if (!forwardModeOnly) {
    addLine(result, `\n//Gradients`);
    for (const kernelName of kernels) {
      const gradImport = moduleProvider.importGradientConfigStr(kernelName);
      addLine(result, gradImport.importStatement);
      addLine(result, registerGradientConfigStr(gradImport.gradConfigId));
    }
  }

  return result.join('\n');
}

function addLine(target: string[], line: string) {
  target.push(line);
}

function registerKernelStr(kernelConfigId: string) {
  return `registerKernel(${kernelConfigId});`;
}

function registerGradientConfigStr(gradConfigId: string) {
  return `registerGradient(${gradConfigId});`;
}
