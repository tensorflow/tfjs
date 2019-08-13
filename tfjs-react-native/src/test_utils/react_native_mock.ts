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

// We mock this library as it cannot be loaded in a browser yet we do want
// to do JS only unit tests.

interface ImageResolvedAssetSource {
  uri: string;
}

// tslint:disable-next-line
export const Image = {
  resolveAssetSource: (resourceId: string|number): ImageResolvedAssetSource => {
    return {
      uri: `http://localhost/assets/${resourceId}`,
    };
  }
};
