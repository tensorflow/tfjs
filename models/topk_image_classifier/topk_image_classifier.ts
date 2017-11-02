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
import {Array1D, Array2D, Array3D, NDArrayMath, NDArrayMathCPU} from '../../src';
import {Model} from '../model';
import {SqueezeNet} from '../squeezenet/squeezenet';

export class TopKImageClassifier extends Model {
  private classLogitsMatrices: Array2D[] = [];
  private classExampleCount: number[] = [];
  private squeezeNet: SqueezeNet;

  // A concatenated matrix of all class logits matrices, lazily created and
  // used during prediction.
  private trainLogitsMatrix: Array2D;

  private varsLoaded = false;
  private mathCPU: NDArrayMathCPU;

  /**
   * Contructor for the class.
   *
   * @param numClasses The number of classes to be able to detect.
   * @param k The number of nearest neighbors to look at when predicting.
   * @param math A math implementation for performing the calculations.
   */
  constructor(
      private numClasses: number, private k: number,
      private math: NDArrayMath) {
    super();
    this.mathCPU = new NDArrayMathCPU();
    for (let i = 0; i < this.numClasses; i++) {
      this.classLogitsMatrices.push(null);
      this.classExampleCount.push(0);
    }
    this.squeezeNet = new SqueezeNet(this.math);
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
  async addImage(image: Array3D, classIndex: number): Promise<void> {
    if (!this.varsLoaded) {
      console.warn('Cannot add images until vars have been loaded.');
      return;
    }
    if (classIndex >= this.numClasses) {
      console.warn('Cannot add to invalid class ${classIndex}');
    }
    this.clearTrainLogitsMatrix();

    await this.math.scope(async (keep, track) => {
      // Add the squeezenet logits for the image to the appropriate class
      // logits matrix.
      const predResults = await this.squeezeNet.predict(image);
      const imageLogits = this.normalizeVector(predResults.logits);
      const logitsSize = imageLogits.shape[0];
      if (!this.classLogitsMatrices[classIndex]) {
        this.classLogitsMatrices[classIndex] = imageLogits.as2D(1, logitsSize);
      } else {
        const newTrainLogitsMatrix = this.math.concat2D(
            this.classLogitsMatrices[classIndex].as2D(
                this.classExampleCount[classIndex], logitsSize),
            imageLogits.as2D(1, logitsSize), 0);
        this.classLogitsMatrices[classIndex].dispose();
        this.classLogitsMatrices[classIndex] = newTrainLogitsMatrix;
      }
      keep(this.classLogitsMatrices[classIndex]);
      this.classExampleCount[classIndex]++;
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
  async predict(image: Array3D):
      Promise<{classIndex: number, confidences: number[]}> {
    let imageClass = -1;
    const confidences = new Array<number>(this.numClasses);
    if (!this.varsLoaded) {
      console.warn('Cannot predict until vars have been loaded.');
      return {classIndex: imageClass, confidences};
    }
    const topKIndices = await this.math.scope(async (keep) => {
      const predResults = await this.squeezeNet.predict(image);
      const imageLogits = this.normalizeVector(predResults.logits);
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

      keep(this.trainLogitsMatrix);
      const numExamples = this.getNumExamples();
      const knn = this.math
                      .matMul(
                          this.trainLogitsMatrix.as2D(numExamples, logitsSize),
                          imageLogits.as2D(logitsSize, 1))
                      .as1D();
      await knn.data();
      const kVal = Math.min(this.k, numExamples);
      const topK = this.mathCPU.topK(knn, kVal);
      return topK.indices;
    });
    if (!topKIndices) {
      return {classIndex: imageClass, confidences};
    }
    const indices = topKIndices.dataSync();
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
    for (let i = 0; i < indices.length; i++) {
      for (let classForEntry = 0; classForEntry < indicesForClasses.length;
           classForEntry++) {
        if (indices[i] < indicesForClasses[classForEntry]) {
          topKCountsForClasses[classForEntry]++;
          break;
        }
      }
    }

    let topConfidence = 0;
    const kVal = Math.min(this.k, this.getNumExamples());
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

  dispose() {
    this.squeezeNet.dispose();
    this.clearTrainLogitsMatrix();
    this.classLogitsMatrices.forEach(
        classLogitsMatrix => classLogitsMatrix.dispose());
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

  private concatWithNulls(ndarray1: Array2D, ndarray2: Array2D): Array2D {
    if (ndarray1 == null && ndarray2 == null) {
      return null;
    }
    if (ndarray1 == null) {
      return this.math.clone(ndarray2);
    } else if (ndarray2 === null) {
      return this.math.clone(ndarray1);
    }
    return this.math.concat2D(ndarray1, ndarray2, 0);
  }

  private getNumExamples() {
    let total = 0;
    for (let i = 0; i < this.classExampleCount.length; i++) {
      total += this.classExampleCount[i];
    }

    return total;
  }

  /**
   * Normalize the provided vector to unit length.
   */
  private normalizeVector(vec: Array1D) {
    const squared = this.math.multiplyStrict(vec, vec);
    const sum = this.math.sum(squared);
    const sqrtSum = this.math.sqrt(sum);
    return this.math.divide(vec, sqrtSum);
  }
}
