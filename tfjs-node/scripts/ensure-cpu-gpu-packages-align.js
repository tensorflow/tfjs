/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
 * Ensures that the GPU and CPU packages align.
 */
const fs = require('fs');
const cpuPackage = require('../../tfjs-node/package.json');
const gpuPackage = require('../../tfjs-node-gpu/package.json');

process.on('unhandledRejection', e => {
  throw e;
});

/**
 * Ensure package.json aligns.
 */
const FIELDS_TO_IGNORE = [
  'name', 'scripts/install', 'scripts/test', 'scripts/prepare', 'scripts/prep',
  'scripts/upload-windows-addon', 'scripts/build-npm',
  'scripts/prep-gpu-windows'
];

const cpuPackageKeys = Object.keys(cpuPackage);
const gpuPackageKeys = Object.keys(gpuPackage);
cpuPackageKeys.forEach((key, i) => {
  if (gpuPackageKeys[i] != cpuPackageKeys[i]) {
    throw new Error(
        `CPU and GPU package have different keys: ` +
        `${gpuPackageKeys[i]} and ${cpuPackageKeys[i]}.`);
  }
});
if (cpuPackageKeys.length != gpuPackageKeys.length) {
  throw new Error(`CPU and GPU package.jsons have different top-level fields.`);
}

// Ensure the cpu and gpu packages have the same keys.
cpuPackageKeys.forEach(key => {
  const cpuPackageValue = cpuPackage[key];
  const gpuPackageValue = gpuPackage[key];

  if (typeof cpuPackageValue !== 'object') {
    if (cpuPackageValue != gpuPackageValue &&
        FIELDS_TO_IGNORE.indexOf(key) === -1) {
      throw new Error(
          `CPU package key '${key}' with value ` +
          `${JSON.stringify(cpuPackageValue)} does not match GPU value ` +
          `${JSON.stringify(gpuPackageValue)}.`);
    }
  } else {
    const cpuFieldKeys = Object.keys(cpuPackageValue);
    cpuFieldKeys.forEach(fieldKey => {
      const cpuFieldValue = '' + cpuPackageValue[fieldKey];
      const gpuFieldValue = '' + gpuPackageValue[fieldKey];
      const deepKey = `${key}/${fieldKey}`;
      if (cpuFieldValue !== gpuFieldValue &&
          FIELDS_TO_IGNORE.indexOf(deepKey) === -1) {
        throw new Error(
            `CPU package key '${deepKey}' with value ` +
            `${JSON.stringify(cpuFieldValue)} does not match GPU value ` +
            `${JSON.stringify(gpuFieldValue)}.`);
      }
    });
  }
});
