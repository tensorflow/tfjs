import * as tf from '@tensorflow/tfjs-core';

import {ops} from './benchmark_util/posenetReadout';
import {benchmarkAndLog, describeWebGPU} from './test_util';

describeWebGPU('benchmark-posenet', () => {
  beforeEach(() => {
    tf.setBackend('webgl');
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 999999999;
  });

  ops.forEach(obj => {
    const opName = obj.name;

    fit(opName, async () => {
      const inputs = (obj.inputs as any).map((input: number[]) => {
        return tf.randomNormal(input);
      });

      let additionalArgs: any = [];
      if ((obj as any).args) {
        additionalArgs = (obj as any).args;
      }

      await benchmarkAndLog(
          `${obj.name}_${(obj.inputs as any).join('|')}`, () => {
            const f = (tf as any)[opName];
            return f.call(tf, ...inputs.concat(additionalArgs));
          });
    });
  });
});
