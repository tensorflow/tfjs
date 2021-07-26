import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';


const t = tf.tensor1d([1,2,3,4,5]);
t.print();
