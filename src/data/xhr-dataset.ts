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

import {NDArray} from '../math/ndarray';
import * as util from '../util';

import {InMemoryDataset} from './dataset';

const PARSING_IMAGE_CANVAS_HEIGHT_PX = 1000;

export interface NDArrayInfo {
  path: string;
  name: string;
  dataType: 'uint8'|'float32'|'png';
  shape: number[];
}

export interface XhrDatasetConfig {
  data: NDArrayInfo[];

  labelClassNames?: string[];
  // Paths to pre-built models.
  modelConfigs: {[modelName: string]: XhrModelConfig};
}

export interface XhrModelConfig { path: string; }

export function getXhrDatasetConfig(jsonConfigPath: string):
    Promise<{[datasetName: string]: XhrDatasetConfig}> {
  return new Promise<{[datasetName: string]: XhrDatasetConfig}>(
      (resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', jsonConfigPath);

        xhr.onload = () => {
          resolve(
              JSON.parse(xhr.responseText) as
              {[datasetName: string]: XhrDatasetConfig});
        };
        xhr.onerror = (error) => {
          reject(error);
        };
        xhr.send();
      });
}

export class XhrDataset extends InMemoryDataset {
  protected xhrDatasetConfig: XhrDatasetConfig;

  constructor(xhrDatasetConfig: XhrDatasetConfig) {
    super(xhrDatasetConfig.data.map(x => x.shape));
    this.xhrDatasetConfig = xhrDatasetConfig;
  }

  protected getNDArray<T extends NDArray>(info: NDArrayInfo): Promise<T[]> {
    const dataPromise = info.dataType === 'png' ?
        parseTypedArrayFromPng(info, info.shape as [number, number, number]) :
        parseTypedArrayFromBinary(info);

    const inputSize = util.sizeFromShape(info.shape);
    return dataPromise.then(data => {
      const ndarrays: T[] = [];
      for (let i = 0; i < data.length / inputSize; i++) {
        const values = data.subarray(i * inputSize, (i + 1) * inputSize);
        const ndarray =
            NDArray.make(
                info.shape, {values: new Float32Array(values)}, 'float32') as T;
        ndarrays.push(ndarray);
      }
      return ndarrays;
    });
  }

  fetchData(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const promises = this.xhrDatasetConfig.data.map(x => this.getNDArray(x));
      Promise.all(promises).then((data: NDArray[][]) => {
        this.dataset = data;
        resolve();
      });
    });
  }
}

function parseTypedArrayFromBinary(info: NDArrayInfo):
    Promise<Float32Array|Uint8Array> {
  return new Promise<Float32Array|Uint8Array>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', info.path);
    xhr.responseType = 'arraybuffer';
    xhr.onload = event => {
      const data = (info.dataType === 'float32') ?
          new Float32Array(xhr.response) :
          new Uint8Array(xhr.response);
      resolve(data);
    };
    xhr.onerror = err => reject(err);
    xhr.send();
  });
}

function parseGrayscaleImageData(
    data: Uint8Array|Uint8ClampedArray, result: Uint8Array,
    resultOffset: number): void {
  let idx = resultOffset;
  for (let i = 0; i < data.length; i += 4) {
    result[idx++] = data[i];
  }
}

function parseRGBImageData(
    data: Uint8Array|Uint8ClampedArray, result: Uint8Array,
    resultOffset: number): void {
  let idx = resultOffset;
  for (let i = 0; i < data.length; i += 4) {
    result[idx] = data[i];
    result[idx + 1] = data[i + 1];
    result[idx + 2] = data[i + 2];
    idx += 3;
  }
}

function parseImage(
    img: HTMLImageElement, shape: [number, number, number]): Uint8Array {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const N = img.height;
  const inputSize = util.sizeFromShape(shape);
  const result = new Uint8Array(N * inputSize);
  if (img.width !== shape[0] * shape[1]) {
    throw new Error(
        `Image width (${img.width}) must be multiple of ` +
        `rows*columns (${shape[0]}*${shape[1]}) of the ndarray`);
  }
  // TODO(smilkov): Canvas has max width of 32,767px. This approach
  // (canvas.width = shape[0] * shape[1]) works with examples up to 181x181px.
  // Consider having the canvas in un-flat format, i.e.
  // canvas.width = shape[1]; canvas.height = DRAW_BATCH * shape[0];
  canvas.width = img.width;

  // Ideally we want canvas.height=img.height (which is N), but canvas size is
  // limited by the browser, so we do multiple passes with a smaller canvas.
  canvas.height = PARSING_IMAGE_CANVAS_HEIGHT_PX;
  const sx = 0;
  const sWidth = canvas.width;
  let sHeight = canvas.height;
  const dx = 0;
  const dy = 0;
  const dWidth = sWidth;
  let dHeight = sHeight;
  const depth = shape[2];
  let offset = 0;
  const numPasses = Math.ceil(N / canvas.height);
  for (let pass = 0; pass < numPasses; ++pass) {
    const sy = pass * canvas.height;
    if ((pass === numPasses - 1) && (N % canvas.height > 0)) {
      // Last pass is a special case.
      canvas.height = N % canvas.height;
      sHeight = canvas.height;
      dHeight = sHeight;
    }
    ctx.drawImage(img, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    (depth === 1) ? parseGrayscaleImageData(data, result, offset) :
                    parseRGBImageData(data, result, offset);
    offset += canvas.height * inputSize;
  }
  return result;
}

function parseTypedArrayFromPng(
    info: NDArrayInfo, shape: [number, number, number]): Promise<Uint8Array> {
  return new Promise<Uint8Array>((resolve, reject) => {
    let img = new Image();
    img.setAttribute('crossOrigin', '');
    img.onload = () => {
      const result = parseImage(img, shape);
      img.src = '';
      img = null;
      resolve(result);
    };
    img.src = info.path;
  });
}
