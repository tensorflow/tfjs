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
          } else if (opName === 'pad') {
            const paddings = [];
            for (let i = 0; i < (obj.inputs as any)[0].length; i++) {
              paddings.push([0, 1]);
            }
            additionalArgs = [paddings];
          } else if (opName === 'resizeBilinear') {
            additionalArgs = [[obj.result[1], obj.result[2]]];
          } else if (opName === 'cast') {
            additionalArgs = ['float32'];
          }

          await benchmarkAndLog(
              `${obj.name}_${(obj.inputs as any).join('|')}`, () => {
                let f = (tf as any)[opName];
                if (obj.scope) {
                  f = (tf as any)[obj.scope][opName];
                }
                return f.call(tf, ...inputs.concat(additionalArgs));
              });
        });
      });
});
