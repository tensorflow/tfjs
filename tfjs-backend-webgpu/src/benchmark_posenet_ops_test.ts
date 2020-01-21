import * as tf from '@tensorflow/tfjs-core';

import {ops} from './benchmark_util/posenetReadout';
import {benchmarkAndLog, describeWebGPU} from './test_util';

const opNameMap: any = {
  'PadV2': 'pad',
  'Conv2D': 'conv2d'
};

describeWebGPU('benchmark-posenet', () => {
  beforeEach(() => {
    tf.setBackend('webgl');
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 999999999;
  });

  ops.filter(obj => {
       return obj.name !== 'FromPixels';
     })
      .forEach(obj => {
        let opName =
            `${obj.name.substring(0, 1).toLowerCase()}${obj.name.substring(1)}`;
        if (opNameMap[obj.name]) {
          opName = opNameMap[obj.name];
        }

        fit(opName, async () => {
          const inputs = (obj.inputs as any).map((input: number[]) => {
            return tf.randomNormal(input);
          });

          let additionalArgs: any = [];
          if (opName === 'conv2d') {
            additionalArgs = [1, 'same'];
          }

          await benchmarkAndLog(
              `${obj.name}_${(obj.inputs as any).join('|')}`,
              () => (tf as any)[opName].call(
                  tf, ...inputs.concat(additionalArgs)));
        });
      });
});
