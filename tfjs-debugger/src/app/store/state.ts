/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {ActionReducerMap} from '@ngrx/store';

import {Configuration} from '../data_model/configuration';

import {configsReducer} from './reducers';

export interface Configs {
  config1: Configuration;
  config2: Configuration;
}

/** The main app state. */
export interface AppState {
  configs: Configs;
}

/** The initial app state. */
export const initialState: AppState = {
  configs: {config1: {}, config2: {}},
};

/** Reducers for each app state field. */
export const appReducers: ActionReducerMap<AppState> = {
  configs: configsReducer,
};
