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

import * as fs from 'fs';
import * as mkdirp from 'mkdirp';
import * as path from 'path';

import {getCustomConverterOpsModule, getCustomModuleString} from './custom_module';
import {getOpsForConfig} from './model_parser';
import {CustomTFJSBundleConfig, ImportProvider, ModuleProvider, SupportedBackend} from './types';
import {kernelNameToVariableName, opNameToFileName} from './util';

export function getModuleProvider(opts: {}): ModuleProvider {
  return new ESMModuleProvider();
}

class ESMModuleProvider implements ModuleProvider {
  /**
   * Writes out custom tfjs module(s) to disk.
   */
  produceCustomTFJSModule(config: CustomTFJSBundleConfig) {
    const {kernels, backends, forwardModeOnly, outputPath} = config;

    const moduleStrs = getCustomModuleString(
        kernels, backends, forwardModeOnly, esmImportProvider);

    mkdirp.sync(outputPath);
    console.log(`Writing custom tfjs module to ${outputPath}`);

    const customTfjsFileName = 'custom_tfjs.js';
    const customTfjsCoreFileName = 'custom_tfjs_core.js';

    // Write a custom module for @tensorflow/tfjs and @tensorflow/tfjs-core
    fs.writeFileSync(
        path.join(outputPath, customTfjsCoreFileName), moduleStrs.core);
    fs.writeFileSync(
        path.join(outputPath, customTfjsFileName), moduleStrs.tfjs);

    // Write a custom module tfjs-core ops used by converter executors

    let kernelToOps;
    let mappingPath;
    try {
      mappingPath =
          require.resolve('@tensorflow/tfjs-converter/metadata/kernel2op.json');
      kernelToOps = JSON.parse(fs.readFileSync(mappingPath, 'utf-8'));
    } catch (e) {
      console.log(`Error loading kernel to ops mapping file ${mappingPath}`);
      console.log(e);
    }

    const converterOps = getOpsForConfig(config, kernelToOps);
    if (converterOps.length > 0) {
      const converterOpsModule =
          getCustomConverterOpsModule(converterOps, esmImportProvider);

      const customConverterOpsFileName = 'custom_ops_for_converter.ts';

      fs.writeFileSync(
          path.join(outputPath, customConverterOpsFileName),
          converterOpsModule);
    }
  }
}

/**
 * An import provider to generate custom esm modules.
 */
// Exported for tests.
export const esmImportProvider: ImportProvider = {
  importCoreStr() {
    return `
import {registerKernel, registerGradient} from '@tensorflow/tfjs-core/dist/base';
import '@tensorflow/tfjs-core/dist/base_side_effects';
export * from '@tensorflow/tfjs-core/dist/base';
  `;
  },

  importConverterStr() {
    return `export * from '@tensorflow/tfjs-converter';`;
  },

  importBackendStr(backend: SupportedBackend) {
    const backendPkg = getBackendPath(backend);
    return `export * from '${backendPkg}/dist/base';`;
  },

  importKernelStr(kernelName: string, backend: SupportedBackend) {
    // TODO(yassogba) validate whether the target file referenced by
    // importStatement exists and warn the user if it doesn't. That could happen
    // here or in an earlier validation phase that uses this function

    const backendPkg = getBackendPath(backend);
    const kernelConfigId = `${kernelName}_${backend}`;
    const importStatement =
        `import {${kernelNameToVariableName(kernelName)}Config as ${
            kernelConfigId}} from '${backendPkg}/dist/kernels/${kernelName}';`;

    return {importStatement, kernelConfigId};
  },

  importGradientConfigStr(kernelName: string) {
    // TODO(yassogba) validate whether the target file referenced by
    // importStatement exists and warn the user if it doesn't. That could happen
    // here or in an earlier validation phase that uses this function

    const gradConfigId = `${kernelNameToVariableName(kernelName)}GradConfig`;
    const importStatement =
        `import {${gradConfigId}} from '@tensorflow/tfjs-core/dist/gradients/${
            kernelName}_grad';`;

    return {importStatement, gradConfigId};
  },

  importOpForConverterStr(opSymbol) {
    const opFileName = opNameToFileName(opSymbol);
    return `export {${opSymbol}} from '@tensorflow/tfjs-core/dist/ops/${
        opFileName}';`;
  },

  importNamespacedOpsForConverterStr(namespace, opSymbols) {
    const result: string[] = [];

    for (const opSymbol of opSymbols) {
      const opFileName = opNameToFileName(opSymbol);
      const opAlias = `${opSymbol}_${namespace}`;
      result.push(`import {${opSymbol} as ${
          opAlias}} from '@tensorflow/tfjs-core/dist/ops/${namespace}/${
          opFileName}';`);
    }

    result.push(`export const ${namespace} = {`);
    for (const opSymbol of opSymbols) {
      const opAlias = `${opSymbol}_${namespace}`;
      result.push(`\t${opSymbol}: ${opAlias},`);
    }
    result.push(`};`);

    return result.join('\n');
  }
};

function getBackendPath(backend: SupportedBackend) {
  switch (backend) {
    case 'cpu':
      return '@tensorflow/tfjs-backend-cpu';
    case 'webgl':
      return '@tensorflow/tfjs-backend-webgl';
    case 'wasm':
      return '@tensorflow/tfjs-backend-wasm';
    default:
      throw new Error(`Unsupported backend ${backend}`);
  }
}
