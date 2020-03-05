/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// As we move tfjs-backend-cpu and tfjs-backend-webgl out of tfjs-core. This
// file will still allow other packated to import symbols from
// "from '@tensorflow/tfjs-core/dist/jasmine_util';" irrespective of where
// jasmine_util actually lives. This is achieved by copying the transpiled
// version of this file to 'dist' during yarn build

// Use a path mapping so that when this file is copied the import has the
// true path. The path will be made relative in the build process.
export * from 'tfjs-core/src/jasmine_util';
