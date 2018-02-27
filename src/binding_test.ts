import * as dl from 'deeplearn';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';

// TODO(kreeger): Drop the 'webgl' hack when deeplearn 0.5.1 is released to
// allow proper registration of new backends.

// TODO(kreeger): This anonymous function should throw an exception if the
// binding is not installed.
dl.ENV.registerBackend('webgl', () => new NodeJSKernelBackend());
dl.Environment.setBackend('webgl');
dl.ENV.engine.startScope();

const t1: dl.Tensor2D = dl.tensor2d([[1, 2], [3, 4]]);
const t2: dl.Tensor2D = dl.tensor2d([[5, 6], [7, 8]]);
const result = t1.matMul(t2);
const padded = dl.pad2d(result, [[1, 1], [1, 1]]);

console.log('t1', t1.dataSync());
console.log('t2', t2.dataSync());
console.log('matmul: ', result.dataSync());
console.log('padded: ', padded.dataSync());

dl.ENV.engine.endScope(null);
