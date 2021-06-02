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

import {BertNLClassifierClass, BertNLClassifierOptions} from './bert_nl_classifier';
import {BertQuestionAnswererClass} from './bert_qa';
import {WasmFeatures} from './common';
import {ClassificationResult, ImageClassifierClass, ImageClassifierOptions} from './image_classifier';
import {ImageSegmenterClass, ImageSegmenterOptions, SegmentationResult} from './image_segmenter';
import {NLClassifierClass} from './nl_classifier';
import {DetectionResult, ObjectDetectorClass, ObjectDetectorOptions} from './object_detector';
import {TFLiteWebModelRunnerClass} from './tflite_web_model_runner';

export declare interface TFLiteWebAPIClient {
  tflite_web_api: {
    /**
     * Sets the path to load all the WASM module files from.
     *
     * TFLite web API will automatically load WASM module with the best
     * performance based on whether the current browser supports WebAssembly
     * SIMD and multi-threading.
     *
     * @param path The path to load WASM module files from.
     *     For relative path, use :
     *         'relative/path/' (relative to current url path) or
     *         '/relative/path' (relative to the domain root).
     *     For absolute path, use https://some-server.com/absolute/path/.
     */
    setWasmPath(path: string): void;

    /**
     * Gets the WASM features supported by user's browser.
     */
    getWasmFeatures(): Promise<WasmFeatures>;
  };

  // Generic TFLite model runner.
  TFLiteWebModelRunner: TFLiteWebModelRunnerClass;

  // NLClassifier.
  NLClassifier: NLClassifierClass;

  // BertQuestionAnswerer.
  BertQuestionAnswerer: BertQuestionAnswererClass;

  // BertNLClassifier.
  BertNLClassifier: BertNLClassifierClass;
  BertNLClassifierOptions: new() => BertNLClassifierOptions;

  // ImageClassifier
  ImageClassifier: ImageClassifierClass;
  ImageClassifierOptions: new() => ImageClassifierOptions;
  ClassificationResult: new() => ClassificationResult;

  // ObjectDetector
  ObjectDetector: ObjectDetectorClass;
  ObjectDetectorOptions: new() => ObjectDetectorOptions;
  DetectionResult: new() => DetectionResult;

  // ImageSegmenter
  ImageSegmenter: ImageSegmenterClass;
  ImageSegmenterOptions: new() => ImageSegmenterOptions;
  SegmentationResult: new() => SegmentationResult;
}
