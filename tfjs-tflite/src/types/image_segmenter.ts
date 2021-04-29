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

import {BaseTaskLibrary} from './common';

/** Different output types. */
export declare enum OutputType {
  CATEGORY_MASK = 1.0,
  CONFIDENCE_MASK = 2.0,
  UNSPECIFIED = 0.0,
}

/** ImageSegmenterOptions proto instance. */
export declare interface ImageSegmenterOptions {
  getOutputType(): OutputType;
  setOutputType(value: OutputType): ImageSegmenterOptions;
  hasOutputType(): boolean;

  getNumThreads(): number;
  setNumThreads(value: number): ImageSegmenterOptions;
  hasNumThreads(): boolean;

  // TODO: add more fields as needed.
}

/** SegmentationResult proto instance.  */
export declare interface SegmentationResult {
  getSegmentationList(): Segmentation[];
}

/** Segmentation proto instance.  */
export declare interface Segmentation {
  getHeight(): number;
  getWidth(): number;
  getCategoryMask_asU8(): Uint8Array;
  getColoredLabelsList(): ColoredLabel[];

  // TODO: add more fields as needed.
}

/** ColoredLabel proto instance.  */
export declare interface ColoredLabel {
  getR(): number;
  getG(): number;
  getB(): number;
  getClassName(): string;
  getDisplayName(): string;

  // TODO: add more fields as needed.
}

/** ImageSegmenter class type. */
export declare interface ImageSegmenterClass {
  /**
   * The factory function to create an ImageSegmenter instance.
   *
   * @param model The path to load the TFLite model from, or the model content
   *     in memory.
   * @param options Available options.
   */
  create(model: string|ArrayBuffer, options?: ImageSegmenterOptions):
      Promise<ImageSegmenter>;
}

export declare class ImageSegmenter extends BaseTaskLibrary {
  /** Performs segmentation on the given image-like element. */
  segment(input: ImageData|HTMLImageElement|HTMLCanvasElement|
          HTMLVideoElement): SegmentationResult|undefined;
}
