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

export enum SupportedBackends {
  cpu = 'cpu',
  webgl = 'webgl',
  wasm = 'wasm'
}
export type SupportedBackend = keyof typeof SupportedBackends;

export interface CustomTFJSBundleConfig {
  entries?: string[];             // paths to javascript files to walk
  models?: string[];              // paths to model.json files to walk
  backends?: SupportedBackend[];  // backends to include/use kernels from
  forwardModeOnly?: boolean;      // whether to drop gradients
  outputPath: string;             // path to output folder
  kernels?: string[];             // Kernels to include
  // tslint:disable-next-line: no-any
  moduleOptions: any;             // Extra params to pass to a module provider
  normalizedOutputPath?: string;  // Computed internally
}

// Interface for an object that can provide functionality to generate
// imports for module in that build environment (e.g. OSS vs g3).
export interface ImportProvider {
  importCoreStr: (forwardModeOnly: boolean) => string;
  importOpForConverterStr: (opSymbol: string) => string;
  importNamespacedOpsForConverterStr:
      (namespace: string, opSymbols: string[]) => string;
  importConverterStr: () => string;
  importBackendStr: (backendPkg: string) => string;
  importKernelStr: (kernelName: string, backend: string) => {
    importPath: string, importStatement: string, kernelConfigId: string
  };
  importGradientConfigStr: (kernelName: string) => {
    importPath: string, importStatement: string, gradConfigId: string
  };
  validateImportPath: (importPath: string) => boolean;
}

// An object that can output a custom model given a config
export interface ModuleProvider {
  produceCustomTFJSModule: (config: CustomTFJSBundleConfig) => void;
}

export interface CustomModuleFiles {
  core: string;
  tfjs: string;
}
