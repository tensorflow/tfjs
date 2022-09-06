/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {memory, registerBackend, setBackend, test_util} from '@tensorflow/tfjs';
// tslint:disable-next-line: no-imports-from-dist
import {TestKernelBackend} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as fs from 'fs';
import {promisify} from 'util';

import {getImageType, ImageType} from './image';
import * as tf from './index';

const readFile = promisify(fs.readFile);

describe('decode images', () => {
  it('decode png', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/image_png_test.png');
    const imageTensor = tf.node.decodePng(uint8array);

    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [238, 101, 0, 50, 50, 50, 100, 50, 0, 200, 100, 50]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode png 1 channels', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/image_png_test.png');
    const imageTensor = tf.node.decodeImage(uint8array, 1);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(await imageTensor.data(), [130, 50, 59, 124]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode png 3 channels', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/image_png_test.png');
    const imageTensor = tf.node.decodeImage(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [238, 101, 0, 50, 50, 50, 100, 50, 0, 200, 100, 50]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode png 4 channels', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_png_4_channel_test.png');
    const imageTensor = tf.node.decodeImage(uint8array, 4);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 4]);
    test_util.expectArraysEqual(await imageTensor.data(), [
      238, 101, 0, 255, 50, 50, 50, 255, 100, 50, 0, 255, 200, 100, 50, 255
    ]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode bmp', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/image_bmp_test.bmp');
    const imageTensor = tf.node.decodeBmp(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [238, 101, 0, 50, 50, 50, 100, 50, 0, 200, 100, 50]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode bmp through decodeImage', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/image_bmp_test.bmp');
    const imageTensor = tf.node.decodeImage(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [238, 101, 0, 50, 50, 50, 100, 50, 0, 200, 100, 50]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode jpg', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_jpeg_test.jpeg');
    const imageTensor = tf.node.decodeJpeg(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode jpeg node bindings do not leak', async () => {
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_jpeg_test.jpeg');

    // Warm up the node bindings
    for (let i = 0; i < 10_000; i++) {
      const imageTensor = tf.node.decodeJpeg(uint8array);
      imageTensor.dispose();
    }

    // Check if decodeJpeg leaks memory by running it many times.
    const beforeMem = process.memoryUsage().rss;
    for (let i = 0; i < 100_000; i++) {
      const imageTensor = tf.node.decodeJpeg(uint8array);
      imageTensor.dispose();
    }
    const afterMem = process.memoryUsage().rss;

    // Due to GC fluctuations, There has to be a large 1Mb margain for this
    // test, but if decodeJpeg leaks more than 10 bytes per run, it will be
    // detected.
    expect(afterMem).toBeLessThan(beforeMem + 1e6 /* 1Mb */);
  });

  it('decode jpg 1 channel', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_jpeg_test.jpeg');
    const imageTensor = tf.node.decodeImage(uint8array, 1);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 1]);
    test_util.expectArraysEqual(await imageTensor.data(), [130, 47, 56, 121]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode jpg 3 channels', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_jpeg_test.jpeg');
    const imageTensor = tf.node.decodeImage(uint8array, 3);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode jpg with 0 channels, use the number of channels in the ' +
         'JPEG-encoded image',
     async () => {
       const beforeNumTensors = memory().numTensors;
       const uint8array = await getUint8ArrayFromImage(
           'test_objects/images/image_jpeg_test.jpeg');
       const imageTensor = tf.node.decodeImage(uint8array);
       expect(imageTensor.dtype).toBe('int32');
       expect(imageTensor.shape).toEqual([2, 2, 3]);
       test_util.expectArraysEqual(
           await imageTensor.data(),
           [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]);
       expect(memory().numTensors).toBe(beforeNumTensors + 1);
     });

  it('decode jpg with downscale', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array = await getUint8ArrayFromImage(
        'test_objects/images/image_jpeg_test.jpeg');
    const imageTensor = tf.node.decodeJpeg(uint8array, 0, 2);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([1, 1, 3]);
    test_util.expectArraysEqual(await imageTensor.data(), [147, 75, 25]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode gif', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/gif_test.gif');
    const imageTensor = tf.node.decodeImage(uint8array);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 2, 3]);
    test_util.expectArraysEqual(await imageTensor.data(), [
      238, 101, 0,  50, 50, 50,  100, 50, 0,   200, 100, 50,
      200, 100, 50, 34, 68, 102, 170, 0,  102, 255, 255, 255
    ]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('decode gif with no expandAnimation', async () => {
    const beforeNumTensors = memory().numTensors;
    const uint8array =
        await getUint8ArrayFromImage('test_objects/images/gif_test.gif');
    const imageTensor = tf.node.decodeImage(uint8array, 3, 'int32', false);
    expect(imageTensor.dtype).toBe('int32');
    expect(imageTensor.shape).toEqual([2, 2, 3]);
    test_util.expectArraysEqual(
        await imageTensor.data(),
        [238, 101, 0, 50, 50, 50, 100, 50, 0, 200, 100, 50]);
    expect(memory().numTensors).toBe(beforeNumTensors + 1);
  });

  it('throw error if request non int32 dtype', async done => {
    try {
      const uint8array = await getUint8ArrayFromImage(
          'test_objects/images/image_png_test.png');
      tf.node.decodeImage(uint8array, 0, 'uint8');
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'decodeImage could only return Tensor of type `int32` for now.');
      done();
    }
  });

  it('throw error if decode invalid image type', async done => {
    try {
      const uint8array = await getUint8ArrayFromImage('package.json');
      tf.node.decodeImage(uint8array);
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'Expected image (BMP, JPEG, PNG, or GIF), ' +
              'but got unsupported image type');
      done();
    }
  });

  it('throw error if backend is not tensorflow', async done => {
    try {
      const testBackend = new TestKernelBackend();
      registerBackend('fake', () => testBackend);
      setBackend('fake');

      const uint8array = await getUint8ArrayFromImage(
          'test_objects/images/image_png_test.png');
      tf.node.decodeImage(uint8array);
      done.fail();
    } catch (err) {
      expect(err.message)
          .toBe(
              'Expect the current backend to be "tensorflow", but got "fake"');
      setBackend('tensorflow');
      done();
    }
  });
});

describe('encode images', () => {
  it('encodeJpeg', async () => {
    const imageTensor = tf.tensor3d(
        new Uint8Array([239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]),
        [2, 2, 3]);
    const beforeNumTensors = memory().numTensors;
    const jpegEncodedData = await tf.node.encodeJpeg(imageTensor);
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
    imageTensor.dispose();
  });

  it('encodeJpeg grayscale', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array([239, 0, 47, 0]), [2, 2, 1]);
    const beforeNumTensors = memory().numTensors;
    const jpegEncodedData = await tf.node.encodeJpeg(imageTensor, 'grayscale');
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
    imageTensor.dispose();
  });

  it('encodeJpeg with parameters', async () => {
    const imageTensor = tf.tensor3d(
        new Uint8Array([239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]),
        [2, 2, 3]);
    const format = 'rgb';
    const quality = 50;
    const progressive = true;
    const optimizeSize = true;
    const chromaDownsampling = false;
    const densityUnit = 'cm';
    const xDensity = 500;
    const yDensity = 500;

    const beforeNumTensors = memory().numTensors;
    const jpegEncodedData = await tf.node.encodeJpeg(
        imageTensor, format, quality, progressive, optimizeSize,
        chromaDownsampling, densityUnit, xDensity, yDensity);
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
    imageTensor.dispose();
  });

  it('encodePng', async () => {
    const imageTensor = tf.tensor3d(
        new Uint8Array([239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]),
        [2, 2, 3]);
    const beforeNumTensors = memory().numTensors;
    const pngEncodedData = await tf.node.encodePng(imageTensor);
    const pngDecodedTensor = await tf.node.decodePng(pngEncodedData);
    const pngDecodedData = await pngDecodedTensor.data();
    pngDecodedTensor.dispose();
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
    test_util.expectArraysEqual(await imageTensor.data(), pngDecodedData);
    imageTensor.dispose();
  });

  it('encodePng grayscale', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array([239, 0, 47, 0]), [2, 2, 1]);
    const beforeNumTensors = memory().numTensors;
    const pngEncodedData = await tf.node.encodePng(imageTensor);
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
    imageTensor.dispose();
  });

  it('encodePng with parameters', async () => {
    const imageTensor = tf.tensor3d(
        new Uint8Array([239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]),
        [2, 2, 3]);
    const compression = 4;

    const beforeNumTensors = memory().numTensors;
    const pngEncodedData = await tf.node.encodePng(imageTensor, compression);
    expect(memory().numTensors).toBe(beforeNumTensors);
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
    imageTensor.dispose();
  });
});

async function getUint8ArrayFromImage(path: string) {
  const image = await readFile(path);
  const buf = Buffer.from(image);
  return new Uint8Array(buf);
}
