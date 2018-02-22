/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// tslint:disable-next-line:max-line-length
import * as dl from 'deeplearn';
import {Tensor1D, Tensor2D, Tensor3D} from 'deeplearn';
import {SqueezeNet} from 'deeplearn-squeezenet';
import * as model_util from '../util';

export class KNNImageClassifier {
  private squeezeNet: SqueezeNet;

  // A concatenated matrix of all class logits matrices, lazily created and
  // used during prediction.
  private trainLogitsMatrix: Tensor2D;

  private classLogitsMatrices: Tensor2D[] = [];
  private classExampleCount: number[] = [];

  private varsLoaded = false;
  private squashLogitsDenominator = dl.scalar(300);

  /**
   * Contructor for the class.
   *
   * @param numClasses The number of classes to be able to detect.
   * @param k The number of nearest neighbors to look at when predicting.
   * @param math A math implementation for performing the calculations.
   */
  constructor(private numClasses: number, private k: number) {
    for (let i = 0; i < this.numClasses; i++) {
      this.classLogitsMatrices.push(null);
      this.classExampleCount.push(0);
    }

    this.squeezeNet = new SqueezeNet();
  }

  /**
   * Loads necessary variables for SqueezeNet.
   */
  async load(): Promise<void> {
    await this.squeezeNet.load();
    this.varsLoaded = true;
  }

  /**
   * Clears the saved images from the specified class.
   */
  clearClass(classIndex: number) {
    if (classIndex >= this.numClasses) {
      console.log('Cannot clear invalid class ${classIndex}');
      return;
    }

    this.classLogitsMatrices[classIndex] = null;
    this.classExampleCount[classIndex] = 0;
    this.clearTrainLogitsMatrix();
  }

  /**
   * Adds the provided image to the specified class.
   */
  addImage(image: Tensor3D, classIndex: number): void {
    if (!this.varsLoaded) {
      console.warn('Cannot add images until vars have been loaded.');
      return;
    }
    if (classIndex >= this.numClasses) {
      console.warn('Cannot add to invalid class ${classIndex}');
    }
    this.clearTrainLogitsMatrix();

    dl.tidy(() => {
      // Add the squeezenet logits for the image to the appropriate class
      // logits matrix.
      const logits = this.squeezeNet.predict(image);
      const imageLogits = this.normalizeVector(logits);

      const logitsSize = imageLogits.shape[0];
      if (this.classLogitsMatrices[classIndex] == null) {
        this.classLogitsMatrices[classIndex] = imageLogits.as2D(1, logitsSize);
      } else {
        const newTrainLogitsMatrix =
            this.classLogitsMatrices[classIndex]
                .as2D(this.classExampleCount[classIndex], logitsSize)
                .concat(imageLogits.as2D(1, logitsSize), 0);

        this.classLogitsMatrices[classIndex].dispose();
        this.classLogitsMatrices[classIndex] = newTrainLogitsMatrix;
      }

      dl.keep(this.classLogitsMatrices[classIndex]);

      this.classExampleCount[classIndex]++;
    });
  }

  /**
   * You are probably are looking for "predictClass".
   *
   * This method returns the K-nearest neighbors as distances in the database.
   *
   * This unfortunately deviates from standard behavior for nearest neighbors
   * classifiers, making this method relatively useless:
   * http://scikit-learn.org/stable/modules/neighbors.html
   *
   * TODO(nsthorat): Return the class indices once we have GPU kernels for topK
   * and take. This method is useless on its own, but matches our Model API.
   *
   * @param image The input image.
   * @returns cosine distances for each entry in the database.
   */
  predict(image: Tensor3D): Tensor1D {
    if (!this.varsLoaded) {
      throw new Error('Cannot predict until vars have been loaded.');
    }

    return dl.tidy(() => {
      const logits = this.squeezeNet.predict(image);
      const imageLogits = this.normalizeVector(logits);
      const logitsSize = imageLogits.shape[0];

      // Lazily create the logits matrix for all training images if necessary.
      if (this.trainLogitsMatrix == null) {
        let newTrainLogitsMatrix = null;

        for (let i = 0; i < this.numClasses; i++) {
          newTrainLogitsMatrix = this.concatWithNulls(
              newTrainLogitsMatrix, this.classLogitsMatrices[i]);
        }
        this.trainLogitsMatrix = newTrainLogitsMatrix;
      }

      if (this.trainLogitsMatrix == null) {
        console.warn('Cannot predict without providing training images.');
        return null;
      }

      dl.keep(this.trainLogitsMatrix);

      const numExamples = this.getNumExamples();
      return this.trainLogitsMatrix.as2D(numExamples, logitsSize)
          .matMul(imageLogits.as2D(logitsSize, 1))
          .as1D();
    });
  }

  /**
   * Predicts the class of the provided image using KNN from the previously-
   * added images and their classes.
   *
   * @param image The image to predict the class for.
   * @returns A dict of the top class for the image and an array of confidence
   * values for all possible classes.
   */
  async predictClass(image: Tensor3D):
      Promise<{classIndex: number, confidences: number[]}> {
    let imageClass = -1;
    const confidences = new Array<number>(this.numClasses);
    if (!this.varsLoaded) {
      throw new Error('Cannot predict until vars have been loaded.');
    }

    const knn = this.predict(image).asType('float32');
    const numExamples = this.getNumExamples();
    const kVal = Math.min(this.k, numExamples);
    const topK = model_util.topK(await knn.data() as Float32Array, kVal);
    knn.dispose();
    const topKIndices = topK.indices;

    if (topKIndices == null) {
      return {classIndex: imageClass, confidences};
    }

    const indicesForClasses = [];
    const topKCountsForClasses = [];
    for (let i = 0; i < this.numClasses; i++) {
      topKCountsForClasses.push(0);
      let num = this.classExampleCount[i];
      if (i > 0) {
        num += indicesForClasses[i - 1];
      }
      indicesForClasses.push(num);
    }

    for (let i = 0; i < topKIndices.length; i++) {
      for (let classForEntry = 0; classForEntry < indicesForClasses.length;
           classForEntry++) {
        if (topKIndices[i] < indicesForClasses[classForEntry]) {
          topKCountsForClasses[classForEntry]++;
          break;
        }
      }
    }

    let topConfidence = 0;
    for (let i = 0; i < this.numClasses; i++) {
      const probability = topKCountsForClasses[i] / kVal;
      if (probability > topConfidence) {
        topConfidence = probability;
        imageClass = i;
      }
      confidences[i] = probability;
    }

    return {classIndex: imageClass, confidences};
  }

  getClassExampleCount(): number[] {
    return this.classExampleCount;
  }

  /**
   * Clear the lazily-loaded train logits matrix due to a change in
   * training data.
   */
  private clearTrainLogitsMatrix() {
    if (this.trainLogitsMatrix != null) {
      this.trainLogitsMatrix.dispose();
      this.trainLogitsMatrix = null;
    }
  }

  private concatWithNulls(ndarray1: Tensor2D, ndarray2: Tensor2D): Tensor2D {
    if (ndarray1 == null && ndarray2 == null) {
      return null;
    }
    if (ndarray1 == null) {
      return ndarray2.clone();
    } else if (ndarray2 === null) {
      return ndarray1.clone();
    }
    return ndarray1.concat(ndarray2, 0);
  }

  /**
   * Normalize the provided vector to unit length.
   */
  private normalizeVector(vec: Tensor1D) {
    // This hack is here for numerical stability on devices without floating
    // point textures. We divide by a constant so that the sum doesn't overflow
    // our fixed point precision. Remove this once we use floating point
    // intermediates with proper dynamic range quantization.
    const squashedVec = dl.div(vec, this.squashLogitsDenominator);
    const sqrtSum = squashedVec.square().sum().sqrt();

    return dl.div(squashedVec, sqrtSum);
  }

  private getNumExamples() {
    let total = 0;
    for (let i = 0; i < this.classExampleCount.length; i++) {
      total += this.classExampleCount[i];
    }

    return total;
  }

  dispose() {
    this.squeezeNet.dispose();
    this.clearTrainLogitsMatrix();
    this.classLogitsMatrices.forEach(
        classLogitsMatrix => classLogitsMatrix.dispose());
    this.squashLogitsDenominator.dispose();
  }
}
