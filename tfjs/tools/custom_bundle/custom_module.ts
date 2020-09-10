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

import {CustomModuleFiles, ModuleProvider} from './types';

export function getCustomModuleString(
    kernels: string[],
    backends: string[],
    forwardModeOnly: boolean,
    moduleProvider: ModuleProvider,
    ): CustomModuleFiles {
  const tfjs: string[] = [];

  // A custom tfjs module
  addLine(tfjs, moduleProvider.importCoreStr());
  addLine(tfjs, moduleProvider.importConverterStr());

  for (const backend of backends) {
    addLine(tfjs, `\n//backend = ${backend}`);
    addLine(tfjs, moduleProvider.importBackendStr(backend));
    for (const kernelName of kernels) {
      const kernelImport = moduleProvider.importKernelStr(kernelName, backend);
      addLine(tfjs, kernelImport.importStatement);
      addLine(tfjs, registerKernelStr(kernelImport.kernelConfigId));
    }
  }

  if (!forwardModeOnly) {
    addLine(tfjs, `\n//Gradients`);
    for (const kernelName of kernels) {
      const gradImport = moduleProvider.importGradientConfigStr(kernelName);
      addLine(tfjs, gradImport.importStatement);
      addLine(tfjs, registerGradientConfigStr(gradImport.gradConfigId));
    }
  }

  // A custom tfjs core module for imports within tfjs packages
  const core: string[] = [];
  addLine(core, moduleProvider.importCoreStr());
  return {
    core: core.join('\n'),
    tfjs: tfjs.join('\n'),
  };
}

export function getCustomConverterOpsModule(
    ops: string[], moduleProvider: ModuleProvider): string {
  const result: string[] = [];

  // Separate namespaced apis from non namespaced ones as they require a
  // different export pattern that treats each namespace as a whole.

  const flatOps = [];
  const namespacedOps: {[key: string]: string[]} = {};

  for (const opSymbol of ops) {
    if (opSymbol.match(/\./)) {
      const parts = opSymbol.split(/\./);
      const namespace = parts[0];
      const opName = parts[1];

      if (namespacedOps[namespace] == null) {
        namespacedOps[namespace] = [];
      }
      namespacedOps[namespace].push(opName);
    } else {
      flatOps.push(opSymbol);
    }
  }

  // Group the namespaced symbols by namespace
  for (const namespace of Object.keys(namespacedOps)) {
    const opSymbols = namespacedOps[namespace];
    result.push(moduleProvider.importNamespacedOpsForConverterStr(
        namespace, opSymbols));
  }

  for (const opSymbol of flatOps) {
    result.push(moduleProvider.importOpForConverterStr(opSymbol));
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
