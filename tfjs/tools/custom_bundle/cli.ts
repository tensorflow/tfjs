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

import * as argparse from 'argparse';
import * as chalk from 'chalk';
import * as fs from 'fs';

import {getCustomModuleString} from './custom_module';
import {CustomTFJSBundleConfig, SupportedBackends} from './types';
import {esmModuleProvider} from './esm_module_provider';

const DEFAULT_CUSTOM_BUNDLE_ARGS: Partial<CustomTFJSBundleConfig> = {
  entries: [],
  models: [],
  kernels: [],
  forwardModeOnly: true,
  backends: ['cpu', 'webgl'],
};

const parser = new argparse.ArgumentParser();
parser.addArgument(
    '--config', {help: 'path to custom bundle config file.', required: true});

function bail(errorMsg: string) {
  console.log(chalk.red(errorMsg));
  process.exit(1);
}

function validateArgs(): CustomTFJSBundleConfig {
  const args = parser.parseArgs();
  const configFilePath = args.config;
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

  console.log(`Using custom bundle configuration from ${
      configFilePath}. Final config:`);
  const replacer: null = null;
  const space = 2;
  console.log(`${JSON.stringify(config, replacer, space)}\n`);

  const finalConfig = Object.assign({}, DEFAULT_CUSTOM_BUNDLE_ARGS, config);

  if (finalConfig.entries.length !== 0) {
    bail('Error: config.entries not yet supported');
  }

  if (finalConfig.models.length !== 0) {
    bail('Error: config.models not yet supported');
  }

  for (const requestedBackend of finalConfig.backends) {
    if (requestedBackend !== SupportedBackends.cpu &&
        requestedBackend !== SupportedBackends.webgl &&
        requestedBackend !== SupportedBackends.wasm) {
      bail(`Error: Unsupported backend specified '${requestedBackend}'`);
    }
  }

  return finalConfig;
}

function getKernelNamesForConfig(config: CustomTFJSBundleConfig) {
  // Later on this will do a union of kernels from entries, models and kernels,
  // (and kernels used by the converter itself) Currently we only support
  // directly listing kernels. remember that this also needs to handle
  // kernels used by gradients if forwardModeOnly is false.
  return config.kernels;
}
function produceCustomTFJSModule(
    kernels: string[], backends: string[], forwardModeOnly: boolean,
    outputPath: string) {
  const moduleStr = getCustomModuleString(
      kernels, backends, forwardModeOnly, esmModuleProvider);

  console.log(`Writing custom tfjs module to ${outputPath}`);
  fs.writeFileSync(outputPath, moduleStr);
}

const customConfig = validateArgs();
const kernelsToInclude = getKernelNamesForConfig(customConfig);
produceCustomTFJSModule(
    kernelsToInclude, customConfig.backends, customConfig.forwardModeOnly,
    customConfig.outputPath);
