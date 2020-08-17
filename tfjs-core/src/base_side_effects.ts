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

// Required side effectful code for tfjs-core (in any build)

// Engine is the global singleton that needs to be initialized before the rest
// of the app.
import './engine';
// Register backend-agnostic flags.
import './flags';
// Register platforms
import './platforms/platform_browser';
import './platforms/platform_node';

import * as ops from './ops/ops';
import {setOpHandler} from './tensor';

setOpHandler(ops);
