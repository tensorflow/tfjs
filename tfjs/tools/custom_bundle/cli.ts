#!/usr/bin/env node

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

/**
 * Entry point for cli tool to build custom tfjs bundles
 */
import * as fs from 'fs';
import * as path from 'path';
import * as chalk from 'chalk';
import * as yargs from 'yargs';

import {OP_SCOPE_SUFFIX} from '@tensorflow/tfjs-core';

import {CustomTFJSBundleConfig, SupportedBackends, ModuleProvider} from './types';
import {getModuleProvider} from './esm_module_provider';

// Will be configured when loading the config file.
let moduleProvider: ModuleProvider;

const BASE_PATH = process.env.BASE_PATH || process.cwd();

const DEFAULT_CUSTOM_BUNDLE_ARGS: Partial<CustomTFJSBundleConfig> = {
  entries: [],
  models: [],
  kernels: [],
  forwardModeOnly: true,
  backends: ['cpu', 'webgl'],
  moduleOptions: {},
};

const argParser = yargs.options({
  config: {
    description: 'Path to custom bundle config file.',
    type: 'string',
    demandOption: true
  }
});

const args = argParser.argv;

function bail(errorMsg: string) {
  console.log(chalk.red(errorMsg));
  process.exit(1);
}

function validateArgs(): CustomTFJSBundleConfig {
  let configFilePath = args.config;
  if (configFilePath == null) {
    bail(`Error: no config file passed`);
  }

  configFilePath = path.resolve(BASE_PATH, configFilePath);

  if (!fs.existsSync(configFilePath)) {
    bail(`Error: config file does not exist at ${configFilePath}`);
  }
  let config;
  try {
    config = JSON.parse(fs.readFileSync(configFilePath, 'utf-8'));
  } catch (error) {
    bail(`Error could not read/parse JSON config file. \n ${error.message}`);
  }

  if (config.outputPath == null) {
    bail('Error: config must specify "outputPath" property');
  }

  console.log(`Using custom bundle configuration from ${configFilePath}.`);

  const finalConfig = Object.assign({}, DEFAULT_CUSTOM_BUNDLE_ARGS, config);

  if (finalConfig.entries.length !== 0) {
    bail('Error: config.entries not yet supported');
  }

  // if (finalConfig.models.length !== 0) {
  // TODO validate that all these paths exist.
  // bail('Error: config.models not yet supported');
  // }

  for (const requestedBackend of finalConfig.backends) {
    if (requestedBackend !== SupportedBackends.cpu &&
        requestedBackend !== SupportedBackends.webgl &&
        requestedBackend !== SupportedBackends.wasm) {
      bail(`Error: Unsupported backend specified '${requestedBackend}'`);
    }
  }

  // Normalize the paths to absolute paths.
  function normalizePath(p: string) {
    return path.resolve(BASE_PATH, p);
  }

  finalConfig.models = finalConfig.models.map(normalizePath);
  finalConfig.entries = finalConfig.entries.map(normalizePath);
  finalConfig.normalizedOutputPath = normalizePath(finalConfig.outputPath);

  moduleProvider = getModuleProvider(finalConfig.moduleOptions);

  console.log('Final Configuration', finalConfig);

  return finalConfig;
}

function getKernelNamesForConfig(config: CustomTFJSBundleConfig) {
  // Later on this will do a union of kernels from entries, models and
  // kernels, (and kernels used by the converter itself) Currently we only
  // support directly listing kernels. remember that this also needs to handle
  // kernels used by gradients if forwardModeOnly is false.

  // Ops in core that are implemented as custom ops may appear in tf.profile
  // they will have __op as a suffix. These do not have corresponding backend
  // kernels so we need to filter them out.
  function isNotCustomOp(kernelName: string) {
    // opSuffix value is defined in tfjs-core/src/operation.ts
    // duplicating it here to avoid an export.
    return !kernelName.endsWith(OP_SCOPE_SUFFIX);
  }

  return config.kernels.filter(isNotCustomOp);
}

const customConfig = validateArgs();
const kernelsToInclude = getKernelNamesForConfig(customConfig);
customConfig.kernels = kernelsToInclude;
if (moduleProvider != null) {
  moduleProvider.produceCustomTFJSModule(customConfig);
} else {
  throw new Error('No module provider has been initialized.');
}
