/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// Importing local_storage and indexed_db is necessary for the routers to be
// registered.
import './indexed_db';
import './local_storage';

import {browserFiles} from './browser_files';
import {browserHTTPRequest, http, isHTTPScheme} from './http';
import {concatenateArrayBuffers, decodeWeights, encodeWeights, getModelArtifactsInfoForJSON} from './io_utils';
import {fromMemory, withSaveHandler} from './passthrough';
import {getLoadHandlers, getSaveHandlers, registerLoadRouter, registerSaveRouter} from './router_registry';
import {IOHandler, LoadHandler, LoadOptions, ModelArtifacts, ModelArtifactsInfo, ModelJSON, ModelStoreManager, OnProgressCallback, RequestDetails, SaveConfig, SaveHandler, SaveResult, WeightGroup, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeights, weightsLoaderFactory} from './weights_loader';

export {copyModel, listModels, moveModel, removeModel} from './model_management';
export {
  browserFiles,
  browserHTTPRequest,
  concatenateArrayBuffers,
  decodeWeights,
  encodeWeights,
  fromMemory,
  getLoadHandlers,
  getModelArtifactsInfoForJSON,
  getSaveHandlers,
  http,
  IOHandler,
  isHTTPScheme,
  LoadHandler,
  LoadOptions,
  loadWeights,
  ModelArtifacts,
  ModelArtifactsInfo,
  ModelJSON,
  ModelStoreManager,
  OnProgressCallback,
  registerLoadRouter,
  registerSaveRouter,
  RequestDetails,
  SaveConfig,
  SaveHandler,
  SaveResult,
  WeightGroup,
  weightsLoaderFactory,
  WeightsManifestConfig,
  WeightsManifestEntry,
  withSaveHandler
};
