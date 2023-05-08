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

import * as tfliteWebAPIClient from '../tflite_web_api_client';
import {ImageSegmenter as TaskLibraryImageSegmenter} from '../types/image_segmenter';
import {BaseTaskLibraryClient, CommonTaskLibraryOptions, getDefaultNumThreads} from './common';

/** Different output types. */
export enum OutputType {
  CATEGORY_MASK = 1.0,
  CONFIDENCE_MASK = 2.0,
  UNSPECIFIED = 0.0,
}

/** ImageSegmenter options. */
export interface ImageSegmenterOptions extends CommonTaskLibraryOptions {
  outputType?: OutputType;
}

/** A single segmentation in the result. */
export interface Segmentation {
  /**
   * The width of the mask. This is an intrinsic parameter of the model being
   * used, and does not depend on the input image dimensions.
   */
  width: number;
  /**
   * The height of the mask. This is an intrinsic parameter of the model being
   * used, and does not depend on the input image dimensions.
   */
  height: number;
  /**
   * This is a flattened 2D-array of size `width` x `height`, in row major
   * order. The value of each pixel in this mask represents the class to which
   * the pixel in the mask belongs. See `coloredLabels` for instructions on how
   * to get pixel labels and display color.
   */
  categoryMask: Uint8Array;
  /**
   * The list of colored labels for all the supported categories. Depending on
   * which is present, this list is in 1:1 correspondence with:
   * - * `categoryMask` pixel values, i.e. a pixel with value `i` is associated
   * with `coloredLabels[i]`,
   * - `confidenceMasks` indices, i.e. `confidence_masks[i]` is associated with
   * `coloredLabels[i]` (TODO: to be supported here).
   */
  coloredLabels: ColoredLabel[];
}

/** A label associated with an RGB color, for display purpose. */
export interface ColoredLabel {
  /** The red color component for the label, in the [0, 255] range. */
  r: number;
  /** The green color component for the label, in the [0, 255] range. */
  g: number;
  /** The blue color component for the label, in the [0, 255] range. */
  b: number;
  /**
   * The class name, as provided in the label map packed in the TFLite Model
   * Metadata.
   */
  className: string;
  /**
   * The display name, as provided in the label map (if available) packed in
   * the TFLite Model Metadata.
   */
  displayName: string;
}

/**
 * Client for ImageSegmenter TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class ImageSegmenter extends BaseTaskLibraryClient {
  constructor(protected override instance: TaskLibraryImageSegmenter) {
    super(instance);
  }

  static async create(
      model: string|ArrayBuffer,
      options?: ImageSegmenterOptions): Promise<ImageSegmenter> {
    const optionsProto = new tfliteWebAPIClient.tfweb.ImageSegmenterOptions();
    if (options) {
      // Set defaults.
      if (options.outputType) {
        optionsProto.setOutputType(options.outputType);
      }
      if (options.numThreads !== undefined) {
        optionsProto.setNumThreads(options.numThreads);
      }
    }
    if (!options || options.numThreads === undefined) {
      optionsProto.setNumThreads(await getDefaultNumThreads());
    }
    const instance = await tfliteWebAPIClient.tfweb.ImageSegmenter.create(
        model, optionsProto);
    return new ImageSegmenter(instance);
  }

  segment(input: ImageData|HTMLImageElement|HTMLCanvasElement|
          HTMLVideoElement): Segmentation[] {
    const result = this.instance.segment(input);
    if (!result) {
      return [];
    }

    const segmentations: Segmentation[] = [];
    if (result.getSegmentationList().length > 0) {
      result.getSegmentationList().forEach(seg => {
        const coloredLabels: ColoredLabel[] =
            seg.getColoredLabelsList().map(label => {
              return {
                r: label.getR(),
                g: label.getG(),
                b: label.getB(),
                className: label.getClassName(),
                displayName: label.getDisplayName(),
              };
            });
        segmentations.push({
          width: seg.getWidth(),
          height: seg.getHeight(),
          categoryMask: seg.getCategoryMask_asU8(),
          coloredLabels,
        });
      });
    }
    return segmentations;
  }
}
