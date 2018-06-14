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

// tslint:disable:max-line-length
// Importing local_storage and indexed_db is necessary for the routers to be
// registered.
import './indexed_db';
import './local_storage';

import {browserFiles} from './browser_files';
import {browserHTTPRequest} from './browser_http';
import {concatenateArrayBuffers, decodeWeights, encodeWeights, getModelArtifactsInfoForJSON} from './io_utils';
import {ModelManagement} from './model_management';
import {IORouterRegistry} from './router_registry';

import {IOHandler, LoadHandler, ModelArtifacts, ModelStoreManager, SaveConfig, SaveHandler, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeights} from './weights_loader';
// tslint:enable:max-line-length

const registerSaveRouter = IORouterRegistry.registerSaveRouter;
const registerLoadRouter = IORouterRegistry.registerLoadRouter;
const getSaveHandlers = IORouterRegistry.getSaveHandlers;
const getLoadHandlers = IORouterRegistry.getLoadHandlers;

const copyModel = ModelManagement.copyModel;
const listModels = ModelManagement.listModels;
const moveModel = ModelManagement.moveModel;
const removeModel = ModelManagement.removeModel;

export {
  browserFiles,
  browserHTTPRequest,
  concatenateArrayBuffers,
  copyModel,
  decodeWeights,
  encodeWeights,
  getLoadHandlers,
  getModelArtifactsInfoForJSON,
  getSaveHandlers,
  IOHandler,
  listModels,
  LoadHandler,
  loadWeights,
  ModelArtifacts,
  ModelStoreManager,
  moveModel,
  registerLoadRouter,
  registerSaveRouter,
  removeModel,
  SaveConfig,
  SaveHandler,
  SaveResult,
  WeightsManifestConfig,
  WeightsManifestEntry
};
