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

import {visor} from '../../dist/index';

let visorInstance;

function start() {
  visorInstance = visor();
  visorInstance.surface({name: 'Surface 1'});
  visorInstance.surface({name: 'Surface 2'});

  visorInstance.surface({name: 'Surface 3', tab: 'Tab 2'});
  visorInstance.surface({name: 'Surface 4', tab: 'Tab 3'});
}

start();
