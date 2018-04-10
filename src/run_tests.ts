import * as tfc from '@tensorflow/tfjs-core';
import {bindTensorFlowBackend} from './index';

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
bindTensorFlowBackend();

tfc.test_util.setBeforeAll(() => {});
tfc.test_util.setAfterAll(() => {});
tfc.test_util.setTestEnvFeatures([{BACKEND: 'tensorflow'}]);

const IGNORE_LIST: string[] = [
  // Methods using browser-specific api.
  'loadWeights',
  'time',

  // Backend methods.
  'memory',
  'variable',
  'debug',
  'tidy',

  // Optimizers.
  'RMSPropOptimizer',
  'MomentumOptimizer',
  'AdagradOptimizer',
  'AdamaxOptimizer',
  'AdamOptimizer',
  'SGDOptimizer',
  'AdadeltaOptimizer',
  'optimizer',

  // Unimplemented ops.
  'clip',
  'leakyRelu',
  'elu',
  'expm1',
  'log1p',
  'resizeBilinear',
  'argmin',
  'argmax',
  'avgPool',
  'multinomial',
  'localResponseNormalization',
  'logicalXor',
  'depthwiseConv2D',
  'conv1d',
  'conv2dTranspose',
  'conv2d',
  'atan2',
  'squaredDifference',
  'prelu',
  'batchNormalization2D',
  'batchNormalization3D',
  'batchNormalization4D',
  'tile',

  // Ops with bugs. Some are higher-level ops.
  'mean',
  'relu',
  'norm',
  'moments',
  'sum',  // In browser we allow sum(bool), but TF requires numeric dtype.
  'max',  // Doesn't propagate NaN.
  'min',  // Doesn't propagate NaN.
  'logicalAnd',
  'logicalNot',
  'greaterEqual',
  'greater',
  'lessEqual',
  'less',
  'notEqualStrict',
  'notEqual',
  'equalStrict',
  'equal',
  'oneHot',
  'gather',
  'fromPixels',
  'pow',

  // Depends on ops being fixed first.
  'gradients',
  'customGradient',
];

const runner = new jasmineCtor();
runner.loadConfig({
  spec_files: [
    'src/**/*_test.ts', 'node_modules/@tensorflow/tfjs-core/dist/**/*_test.js'
  ]
});

const env = jasmine.getEnv();

// Filter method that returns boolean, if a given test should return.
env.specFilter = spec => {
  // Return false (skip the test) if the test is in the ignore list.
  for (let i = 0; i < IGNORE_LIST.length; ++i) {
    if (spec.getFullName().startsWith(IGNORE_LIST[i])) {
      return false;
    }
  }
  // Otherwise run the test.
  return true;
};

runner.execute();
