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
import {ENV} from '../environment';

import {Platform} from './platform';

// We are wrapping this within an object so it can be stubbed by Jasmine.
export const getNodeFetch = {
  // tslint:disable-next-line:no-require-imports
  importFetch: () => require('node-fetch')
};

export let systemFetch: (url: string, init?: RequestInit) => Promise<Response>;
export class PlatformNode implements Platform {
  fetch(path: string, requestInits?: RequestInit): Promise<Response> {
    if (ENV.global.fetch != null) {
      return ENV.global.fetch(path, requestInits);
    }

    if (systemFetch == null) {
      systemFetch = getNodeFetch.importFetch();
    }
    return systemFetch(path, requestInits);
  }
}

if (ENV.get('IS_NODE')) {
  ENV.setPlatform('node', new PlatformNode());
}
