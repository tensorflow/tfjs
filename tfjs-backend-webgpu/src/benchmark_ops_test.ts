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
 * =============================================================================
 */

import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs-core';

import {ops as posenetOps} from './benchmark_util/posenetOps';
import {benchmarkAndLog, describeWebGPU} from './test_util';

const getInputInfo = (obj: any) => {
  let info = `${obj.name} shapes: ${JSON.stringify(obj.inputs)}`;

  if (obj.args) {
    info += `, args: ${JSON.stringify(obj.args)}`;
  }

  return info;
};

describeWebGPU('Posenet benchmarks', () => {
  beforeEach(() => {
    tf.setBackend('webgl');
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 999999999;
  });

  posenetOps.forEach(obj => {
    const opName = obj.name;

    fit(opName, async () => {
      const inputs = (obj.inputs as any).map((input: number[]) => {
        return tf.randomNormal(input);
      });

      let additionalArgs: any = [];
      if ((obj as any).args) {
        additionalArgs = (obj as any).args;
      }

      await benchmarkAndLog(getInputInfo(obj), () => {
        const f = (tf as any)[opName];
        return f.call(tf, ...inputs.concat(additionalArgs));
      });
    });
  });

  fit('posenet_resnet', async () => {
    const posenetModel = await posenet.load({
      architecture: 'ResNet50',
      outputStride: 32,
      inputResolution: 257,
      quantBytes: 2
    });
    const image = tf.zeros([257, 257, 3]) as tf.Tensor3D;

    await benchmarkAndLog('posenet_resnet', async () => {
      const pose = await posenetModel.estimateSinglePose(image);
      return pose;
    }, null, false, 10);
  }, 100000000000000000);

  fit('posenet_mobilenet', async () => {
    const posenetModel = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 257,
      quantBytes: 4
    });
    const image = tf.zeros([257, 257, 3]) as tf.Tensor3D;

    await benchmarkAndLog('posenet_mobilenet', async () => {
      const pose = await posenetModel.estimateSinglePose(image);
      return pose;
    }, null, false, 10);
  }, 100000000000000000);

  afterAll(() => {
    function download(filename: string, text: string) {
      const element = document.createElement('a');
      element.setAttribute(
          'href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
      element.setAttribute('download', filename);

      element.style.display = 'none';
      document.body.appendChild(element);

      element.click();

      document.body.removeChild(element);
    }

    // Start file download.
    const dateObj = new Date();
    let date = dateObj.getDate().toString();
    if (date.length === 1) {
      date = `0${date}`;
    }

    download(
        `${('0' + ((dateObj.getMonth() + 1).toString())).slice(-2)}_${date}_${
            dateObj.getFullYear()}.json`,
        JSON.stringify(window.records));
  });
});
