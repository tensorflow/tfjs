/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {tensor3d, test_util} from '@tensorflow/tfjs-core';
import {describeBrowserEnvs, setupFakeVideoStream} from '../util/test_utils';
import {WebcamIterator} from './webcam_iterator';

describeBrowserEnvs('WebcamIterator', () => {
  beforeEach(() => {
    setupFakeVideoStream();
  });

  it('create webcamIterator with html element', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 100;
    videoElement.height = 200;

    const webcamIterator = await WebcamIterator.create(videoElement);
    const result = await webcamIterator.next();
    expect(result.done).toBeFalsy();
    expect(result.value.shape).toEqual([200, 100, 3]);
  });

  it('create webcamIterator with html element and capture', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 100;
    videoElement.height = 200;

    const webcamIterator = await WebcamIterator.create(videoElement);
    const result = await webcamIterator.capture();
    expect(result.shape).toEqual([200, 100, 3]);
  });

  it('create webcamIterator with no html element', async () => {
    const webcamIterator = await WebcamIterator.create(
        null, {resizeWidth: 100, resizeHeight: 200});
    const result = await webcamIterator.next();
    expect(result.done).toBeFalsy();
    expect(result.value.shape).toEqual([200, 100, 3]);
  });

  it('create webcamIterator with no html element and capture', async () => {
    const webcamIterator = await WebcamIterator.create(
        null, {resizeWidth: 100, resizeHeight: 200});
    const result = await webcamIterator.capture();
    expect(result.shape).toEqual([200, 100, 3]);
  });

  it('create webcamIterator with no html element and no size', async done => {
    try {
      await WebcamIterator.create();
      done.fail();
    } catch (e) {
      expect(e.message).toEqual(
          'Please provide webcam video element, or resizeWidth and ' +
          'resizeHeight to create a hidden video element.');
      done();
    }
  });

  it('resize and center crop has correct shape with html element', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 100;
    videoElement.height = 200;
    const webcamIterator = await WebcamIterator.create(
        videoElement, {resizeWidth: 30, resizeHeight: 40, centerCrop: true});
    const result = await webcamIterator.next();
    expect(result.done).toBeFalsy();
    expect(result.value.shape).toEqual([40, 30, 3]);
  });

  it('resize and center crop has correct pixel with html element', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 4;
    videoElement.height = 4;
    const webcamIterator = await WebcamIterator.create(
        videoElement, {resizeWidth: 2, resizeHeight: 2, centerCrop: true});
    const originalImg = tensor3d([
      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
      [
        [1, 1, 1],
        [11, 12, 13],
        [14, 15, 16],
        [1, 1, 1],
      ],
      [
        [1, 1, 1],
        [1, 2, 3],
        [4, 5, 6],
        [1, 1, 1],
      ],
      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ]);
    const croppedImg = webcamIterator.cropAndResizeFrame(originalImg);
    test_util.expectArraysClose(
        croppedImg, tensor3d([
          [[6.625, 7.1875, 7.75], [8.3125, 8.875, 9.4375]],

          [[1, 1.5625, 2.125], [2.6875, 3.25, 3.8125]]
        ]));
  });

  it('resize in bilinear method has correct shape with html element',
     async () => {
       const videoElement = document.createElement('video');
       videoElement.width = 100;
       videoElement.height = 200;

       const webcamIterator = await WebcamIterator.create(
           videoElement,
           {resizeWidth: 30, resizeHeight: 40, centerCrop: false});
       const result = await webcamIterator.next();
       expect(result.done).toBeFalsy();
       expect(result.value.shape).toEqual([40, 30, 3]);
     });

  it('resize in bilinear method has correct pixel with html element',
     async () => {
       const videoElement = document.createElement('video');
       videoElement.width = 4;
       videoElement.height = 4;
       const webcamIterator = await WebcamIterator.create(
           videoElement, {resizeWidth: 2, resizeHeight: 2, centerCrop: false});
       const originalImg = tensor3d([
         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [
           [1, 1, 1],
           [11, 12, 13],
           [14, 15, 16],
           [1, 1, 1],
         ],
         [
           [1, 1, 1],
           [1, 2, 3],
           [4, 5, 6],
           [1, 1, 1],
         ],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
       ]);
       const croppedImg = webcamIterator.cropAndResizeFrame(originalImg);
       test_util.expectArraysClose(croppedImg, tensor3d([
                                     [[1, 1, 1], [1, 1, 1]],

                                     [[1, 1, 1], [1, 1, 1]]
                                   ]));
     });

  it('webcamIterator could stop', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 100;
    videoElement.height = 100;

    const webcamIterator = await WebcamIterator.create(videoElement);
    const result1 = await webcamIterator.next();
    expect(result1.done).toBeFalsy();
    expect(result1.value.shape).toEqual([100, 100, 3]);

    await webcamIterator.stop();
    const result2 = await webcamIterator.next();
    expect(result2.done).toBeTruthy();
    expect(result2.value).toBeNull();
  });

  it('webcamIterator could restart', async () => {
    const videoElement = document.createElement('video');
    videoElement.width = 100;
    videoElement.height = 100;

    const webcamIterator = await WebcamIterator.create(videoElement);
    const result1 = await webcamIterator.next();
    expect(result1.done).toBeFalsy();
    expect(result1.value.shape).toEqual([100, 100, 3]);

    webcamIterator.stop();
    const result2 = await webcamIterator.next();
    expect(result2.done).toBeTruthy();
    expect(result2.value).toBeNull();

    // Reset fake media stream after stopped the stream.
    setupFakeVideoStream();

    await webcamIterator.start();
    // Skip validation when it's in Firefox and Mac OS, because BrowserStack for
    // Firefox on travis does not support restarting experimental function
    // HTMLCanvasElement.captureStream().
    if (navigator.userAgent.search('Firefox') < 0 &&
        navigator.userAgent.search('OS X') < 0) {
      const result3 = await webcamIterator.next();
      expect(result3.done).toBeFalsy();
      expect(result3.value.shape).toEqual([100, 100, 3]);
    }
  });
});
