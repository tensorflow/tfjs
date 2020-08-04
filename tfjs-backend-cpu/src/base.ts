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

/*
 * base.ts contains all the exports from tfjs-backend-cpu
 * without auto-kernel registration
 */
import {registerBackend} from '@tensorflow/tfjs-core';
import {MathBackendCPU} from './backend_cpu';
import * as shared from './shared';

export {MathBackendCPU} from './backend_cpu';
export {version as version_cpu} from './version';
export {shared};

// Side effects for default initialization of MathBackendCPU
registerBackend('cpu', () => new MathBackendCPU(), 1 /* priority */);
