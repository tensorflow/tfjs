#!/usr/bin/env node
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
import * as tf from '../../src/index';
import '../../src/public/chained_ops/register_all_chained_ops';
import '../../src/register_all_gradients';
import '@tensorflow/tfjs-backend-cpu';
import {parseAndEvaluateSnippets} from './util';

parseAndEvaluateSnippets(tf);
