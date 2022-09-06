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

// base.ts is the webgl backend without auto kernel registration.

import {device_util, registerBackend} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from './backend_webgl';
export {version as version_webgl} from './version';

if (device_util.isBrowser()) {
  registerBackend('webgl', () => new MathBackendWebGL(), 2 /* priority */);
}

// Export webgl utilities
export * from './webgl';

// Export forceHalfFlost under webgl namespace for the union bundle.
import {forceHalfFloat} from './webgl';
export const webgl = {forceHalfFloat};
