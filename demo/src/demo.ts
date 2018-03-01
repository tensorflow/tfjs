import * as dl from 'deeplearn';
import * as tf from 'tfjs-node';

tf.bindTensorFlowBackend();

dl.tidy(() => {
  const t1: dl.Tensor2D = dl.tensor2d([[1, 2], [3, 4]]);
  const t2: dl.Tensor2D = dl.tensor2d([[5, 6], [7, 8]]);
  const result = t1.matMul(t2);
  const padded = dl.pad2d(result, [[1, 1], [1, 1]]);

  console.log('t1', t1.dataSync());
  console.log('t2', t2.dataSync());
  console.log('matmul: ', result.dataSync());
  console.log('padded: ', padded.dataSync());
});
