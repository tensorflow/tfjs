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

// TODO(smilkov): Import from "@tensoflow/tfjs-automl" when the package
// is released.
import * as automl from '../../src/index';

const MODEL_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';

async function run() {
  const model = await automl.loadObjectDetectionModel(MODEL_URL);
  const image = document.getElementById('daisy');
  const predictions = await model.detect(image);
  console.log(predictions);
}

run();
