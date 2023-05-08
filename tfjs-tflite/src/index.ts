/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

export * from './tflite_model';
export * from './types/tflite_web_model_runner';
export * from './tflite_task_library_client/image_classifier';
export * from './tflite_task_library_client/image_segmenter';
export * from './tflite_task_library_client/object_detector';
export * from './tflite_task_library_client/nl_classifier';
export * from './tflite_task_library_client/bert_nl_classifier';
export * from './tflite_task_library_client/bert_qa';
export {setWasmPath} from './tflite_task_library_client/common';
export {getWasmFeatures} from './tflite_task_library_client/common';
