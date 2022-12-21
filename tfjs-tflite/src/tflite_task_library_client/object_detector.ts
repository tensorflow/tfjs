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
import {ObjectDetector as TaskLibraryObjectDetector} from '../types/object_detector';
import {BaseTaskLibraryClient, Class, CommonTaskLibraryOptions, convertProtoClassesToClasses, getDefaultNumThreads} from './common';

/** ObjectDetector options. */
export interface ObjectDetectorOptions extends CommonTaskLibraryOptions {
  /**
   * Maximum number of top scored results to return. If < 0, all results will
   * be returned. If 0, an invalid argument error is returned.
   */
  maxResults?: number;

  /**
   * Score threshold in [0,1), overrides the ones provided in the model metadata
   * (if any). Results below this value are rejected.
   */
  scoreThreshold?: number;
}

/** A single detected object in the result. */
export interface Detection {
  boundingBox: BoundingBox;
  classes: Class[];
}

/** A bounding box for the detected object. */
export interface BoundingBox {
  originX: number;
  originY: number;
  width: number;
  height: number;
}

/**
 * Client for ObjectDetector TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class ObjectDetector extends BaseTaskLibraryClient {
  constructor(protected override instance: TaskLibraryObjectDetector) {
    super(instance);
  }

  static async create(
      model: string|ArrayBuffer,
      options?: ObjectDetectorOptions): Promise<ObjectDetector> {
    const optionsProto = new tfliteWebAPIClient.tfweb.ObjectDetectorOptions();
    if (options) {
      if (options.maxResults !== undefined) {
        optionsProto.setMaxResults(options.maxResults);
      }
      if (options.scoreThreshold !== undefined) {
        optionsProto.setScoreThreshold(options.scoreThreshold);
      }
      if (options.numThreads !== undefined) {
        optionsProto.setNumThreads(options.numThreads);
      }
    }
    if (!options || options.numThreads === undefined) {
      optionsProto.setNumThreads(await getDefaultNumThreads());
    }
    const instance = await tfliteWebAPIClient.tfweb.ObjectDetector.create(
        model, optionsProto);
    return new ObjectDetector(instance);
  }

  detect(input: ImageData|HTMLImageElement|HTMLCanvasElement|
         HTMLVideoElement): Detection[] {
    const result = this.instance.detect(input);
    if (!result) {
      return [];
    }

    const detections: Detection[] = [];
    if (result.getDetectionsList().length > 0) {
      result.getDetectionsList().forEach(detection => {
        const boundingBoxProto = detection.getBoundingBox();
        const boundingBox: BoundingBox = {
          originX: boundingBoxProto.getOriginX(),
          originY: boundingBoxProto.getOriginY(),
          width: boundingBoxProto.getWidth(),
          height: boundingBoxProto.getHeight(),
        };
        const classes =
            convertProtoClassesToClasses(detection.getClassesList());
        detections.push({boundingBox, classes});
      });
    }
    return detections;
  }
}
