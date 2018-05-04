/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {TypedArray} from '@tensorflow/tfjs-core/dist/types';
import {sizeFromShape} from '@tensorflow/tfjs-core/dist/util';
import {equal} from 'assert';
import {createWriteStream, existsSync, readFileSync} from 'fs';
import {get} from 'https';
import {createGunzip} from 'zlib';

const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';

const NUM_TRAIN_EXAMPLES = 60000;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_DIMENSION_SIZE = 28;
const IMAGE_FLAT_SIZE = IMAGE_DIMENSION_SIZE * IMAGE_DIMENSION_SIZE;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

function downloadFile(filename: string): Promise<string> {
  return new Promise((resolve) => {
    const url = `${BASE_URL}${filename}.gz`;
    if (existsSync(filename)) {
      return resolve();
    }
    const file = createWriteStream(filename);
    console.log('  * Downloading from ', url);
    get(url, (response) => {
      const unzip = createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', resolve);
    });
  });
}

function loadHeaderValues(buffer: Buffer, headerLength: number): number[] {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka BE)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(filename: string): Promise<TypedArray[]> {
  await downloadFile(filename);

  const buffer = readFileSync(filename);

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_DIMENSION_SIZE * IMAGE_DIMENSION_SIZE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  equal(headerValues[0], 2051);  // magic number for images
  equal(headerValues[1], NUM_TRAIN_EXAMPLES);
  equal(headerValues[2], IMAGE_DIMENSION_SIZE);
  equal(headerValues[3], IMAGE_DIMENSION_SIZE);

  const downsize = 1.0 / 255.0;

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++) * downsize;
    }
    images.push(array);
  }

  equal(images.length, headerValues[1]);
  return images;
}

async function loadLabels(filename: string): Promise<TypedArray[]> {
  await downloadFile(filename);

  const buffer = readFileSync(filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  equal(headerValues[0], 2049);  // magic number for labels
  equal(headerValues[1], NUM_TRAIN_EXAMPLES);

  const labels = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Uint8Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  equal(labels.length, headerValues[1]);
  return labels;
}

export class MnistDataset {
  protected dataset: TypedArray[][]|null;
  protected batchIndex = 0;

  async loadData(): Promise<void> {
    this.dataset = await Promise.all(
        [loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE)]);
  }

  reset() {
    this.batchIndex = 0;
  }

  hasMoreData(): boolean {
    return this.batchIndex < NUM_TRAIN_EXAMPLES;
  }

  nextTrainBatch(batchSize: number): {image: tf.Tensor2D, label: tf.Tensor2D} {
    const batchIndexMax = this.batchIndex + batchSize > NUM_TRAIN_EXAMPLES ?
        NUM_TRAIN_EXAMPLES - this.batchIndex :
        batchSize + this.batchIndex;
    const size = batchIndexMax - this.batchIndex;

    // Only create one big array to hold batch of images.
    const imagesShape = [size, IMAGE_FLAT_SIZE];
    const images = new Float32Array(sizeFromShape(imagesShape));

    const labelsShape = [size, 1];
    const labels = new Int32Array(sizeFromShape(labelsShape));

    let imageOffset = 0;
    let labelOffset = 0;
    while (this.batchIndex < batchIndexMax) {
      images.set(this.dataset[0][this.batchIndex], imageOffset);
      labels.set(this.dataset[1][this.batchIndex], labelOffset);

      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
      this.batchIndex++;
    }

    return {
      image: tf.tensor2d(images, [size, IMAGE_FLAT_SIZE]),
      label: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}
