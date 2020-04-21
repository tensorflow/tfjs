/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const SWATCHES = {
  'webgpu_min': '#F1523E',
  'webgpu_mean': '#F1523E',
  'webgl_min': '#3f51b5',
  'webgl_mean': '#3f51b5'
};

const STROKES = {
  'webgpu_min': '2',
  'webgpu_mean': '0',
  'webgl_min': '2',
  'webgl_mean': '0'
};

const TARGETS = ['canary'];
const MOMENT_DISPLAY_FORMAT = 'MM/DD/YYYY';
const MAX_NUM_LOGS = 50;
const START_LOGGING_DATE = '2019-08-16';
const CHART_HEIGHT = 200;
