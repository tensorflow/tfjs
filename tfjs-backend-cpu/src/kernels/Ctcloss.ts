import { grad, Rank, Tensor, Tensor1D, Tensor2D } from "@tensorflow/tfjs";
import { KernelConfig, KernelFunc, CtcLoss } from "@tensorflow/tfjs-core";
const nInf = -Infinity;
const BLANK = 6509;

function alpha(log_y: Tensor<Rank.R2>, labels: Tensor<Rank.R1>): number[][] {
  const [T, V]: number[] = log_y.shape;
  const log_y_data = log_y.arraySync();
  const label_data = labels.arraySync();
  const label_length = label_data.length;

  const log_alpha = tf
    .ones<Rank.R2>([T, label_length])
    .mul<Tensor<Rank.R2>>(nInf)
    .arraySync();

  log_alpha[0][0] = log_y_data[0][label_data[0]];
  log_alpha[0][1] = log_y_data[0][label_data[1]];

  for (let t = 1; t < T; t++) {
    for (let i = 0; i < label_length; i++) {
      const character = label_data[i];

      let sum = log_alpha[t - 1][i];

      if (i - 1 >= 0) {
        sum = logSumExp(sum, log_alpha[t - 1][i - 1]);
      }
      if (
        i - 2 >= 0 &&
        character !== BLANK &&
        character !== label_data[i - 2]
      ) {
        sum = logSumExp(sum, log_alpha[t - 1][i - 2]);
      }

      log_alpha[t][i] = sum + log_y_data[t][character];
    }
  }

  return log_alpha;
}

function beta(log_y: Tensor<Rank.R2>, labels: Tensor<Rank.R1>): number[][] {
  const [T, V]: number[] = log_y.shape;
  const log_y_data = log_y.arraySync();
  const label_data = labels.arraySync();
  const label_length = label_data.length;
  const log_beta = tf
    .ones([T, label_length])
    .mul<Tensor<Rank.R2>>(nInf)
    .arraySync();

  log_beta[T - 1][label_length - 1] =
    log_y_data[T - 1][label_data[label_length - 1]];
  log_beta[T - 1][label_length - 2] =
    log_y_data[T - 1][label_data[label_length - 2]];

  for (let t = T - 2; t > -1; t--) {
    for (let i = 0; i < label_length; i++) {
      const character = label_data[i];
      let sum = log_beta[t + 1][i];
      if (i + 1 < label_length) {
        sum = logSumExp(sum, log_beta[t + 1][i + 1]);
      }
      if (
        i + 2 < label_length &&
        character !== BLANK &&
        character !== label_data[i + 1]
      ) {
        sum = logSumExp(sum, log_beta[t + 1][i + 2]);
      }
      log_beta[t][i] = sum + log_y_data[t][character];
    }
  }

  return log_beta;
}

function alpha_vanilla(
  y: Tensor<Rank.R2>,
  labels: Tensor<Rank.R1>
): number[][] {
  const [T, V]: number[] = y.shape;
  const y_data = y.arraySync();
  const label_data = labels.arraySync();
  const label_length = label_data.length;

  const alpha = tf
    .zeros<Rank.R2>([T, label_length])
    .arraySync();

  alpha[0][0] = y_data[0][label_data[0]];
  alpha[0][1] = y_data[0][label_data[1]];

  for (let t = 1; t < T; t++) {
    for (let i = 0; i < label_length; i++) {
      const character = label_data[i];

      let sum = alpha[t - 1][i];

      if (i - 1 >= 0) {
        sum += alpha[t - 1][i - 1];
      }
      if (
        i - 2 >= 0 &&
        character !== BLANK &&
        character !== label_data[i - 2]
      ) {
        sum += alpha[t - 1][i - 2];
      }

      alpha[t][i] = sum * y_data[t][character];
    }
  }

  return alpha;
}

function beta_vanilla(y: Tensor<Rank.R2>, labels: Tensor<Rank.R1>): number[][] {
  const [T, V]: number[] = y.shape;
  const y_data = y.arraySync();
  const label_data = labels.arraySync();
  const label_length = label_data.length;
  const beta = tf.zeros([T, label_length]).arraySync();

  beta[T - 1][label_length - 1] = y_data[T - 1][V - 1];
  beta[T - 1][label_length - 2] = y_data[T - 1][V - 2];

  for (let t = T - 2; t > -1; t--) {
    for (let i = 0; i < label_length; i++) {
      const character = label_data[i];
      let sum = beta[t + 1][i];
      if (i + 1 < label_length) {
        sum += beta[t + 1][i + 1];
      }
      if (
        i + 2 < label_length &&
        character !== BLANK &&
        character !== label_data[i + 2]
      ) {
        sum += beta[t + 1][i + 2];
      }
      beta[t][i] = sum * y_data[t][character];
    }
  }

  return beta;
}

function _logSumExp(a: number, b: number) {
  if (a < b) {
    [a, b] = [b, a];
  }
  if (b == nInf) {
    return a;
  }

  return a + Math.log(1 + Math.exp(b - a));
}

function logSumExp(...args) {
  return args.reduce((acc, item) => {
    return _logSumExp(acc, item);
  }, nInf);
}

export function batch_gradient(y: Tensor<Rank.R3>, labels: Tensor<Rank.R2>) {
  const [B, T, V]: number[] = y.shape;
  const [_, label_length] = labels.shape;
  const y_array: Tensor<Rank.R2>[] = y.split(B, 0);
  const label_array: Tensor<Rank.R1>[] = labels.split(B, 0);
  const grads = [];
  for (let b = 0; b < B; b++) {
    grads.push(
      gradient(
        y_array[b].reshape([T, V]),
        label_array[b].reshape([label_length])
      )
    );
  }
  return tf.stack(grads);
}

function gradient(
  y: Tensor<Rank.R2>,
  labels: Tensor<Rank.R1>
): Tensor<Rank.R2> {
  const [T, V]: number[] = y.shape;
  const y_data = y.arraySync();
  const labels_blank = insert_blank(labels);
  const label_data = labels_blank.arraySync();
  const [label_length] = labels_blank.shape;
  const alpha = alpha_vanilla(y, labels_blank);
  const beta = beta_vanilla(y, labels_blank);

  const probability =
    alpha[T - 1][label_length - 1] + alpha[T - 1][label_length - 2];
  const grad = tf
    .zeros<Rank.R2>([T, V])
    .arraySync();

  for (let t = 0; t < T; t++) {
    for (let w = 0; w < V; w++) {
      const lab = label_data
        .map((item, i) => (item === w ? i : -1))
        .filter((i) => i > -1);
      lab.forEach((i) => {
        grad[t][w] += alpha[t][i] * beta[t][i];
      });
      grad[t][w] /= Math.pow(y_data[t][w], 2);
    }
  }

  for (let t = 0; t < T; t++) {
    for (let w = 0; w < V; w++) {
      grad[t][w] /= -probability;
    }
  }

  return tf.tensor2d(grad);
}

export function batch_ctc_loss(
  y: Tensor<Rank.R3>,
  labels: Tensor<Rank.R2>
): Tensor<Rank.R1> {
  const [Batch, T, V]: number[] = y.shape;
  const [_, label_length] = labels.shape;
  // const log_y = y.log().neg();
  const losses = Array.from({ length: Batch });
  const y_array = y.split(Batch, 0);
  const label_array = labels.split(Batch, 0);
  for (let b = 0; b < Batch; b++) {
    const label_data = label_array[b].reshape([label_length]).arraySync();
    // const log_y_item = log_y_array[b].reshape<Tensor2D>([T, V]);
    // const blank_label_item = insert_blank(
    // label_array[b].reshape([label_length])
    // );
    const res = handleResult(
      y_array[b]
        .reshape<Tensor2D>([T, V])
        .argMax(1)
        .arraySync(),
      label_length
    ).reduce((acc, item, index) => {
      acc += Math.sqrt(Math.abs(label_data[index] - item));
      return acc;
    }, 0);

    // const L = 2 * label_length + 1;

    // const alpha_data = alpha(log_y_item, blank_label_item);
    // const beta_data = beta(log_y_item, blank_label_item);
    // const y_data = log_y_item.arraySync();

    losses[b] = res;
    // losses[b] = logSumExp(alpha_data[T - 1][L - 1], alpha_data[T - 1][L - 2]);
    // logSumExp(beta_data[0][0], beta_data[0][1])
  }
  // console.log(y.argMax(2).arraySync());
  return tf.tensor1d(losses);
}

function insert_blank(labels: Tensor<Rank.R1>): Tensor1D {
  const array = labels.arraySync();
  return tf.tensor1d([BLANK].concat(array.map((item) => [item, BLANK]).flat()));
}

function handleResult(predict: number[], label_length) {
  let chars = [];
  predict.forEach((item, i) => {
    if (item != BLANK && !(i > 0 && item === predict[i - 1])) {
      chars.push(item);
    }
  });
  if (chars.length < label_length) {
    chars = chars.concat(
      Array.from({ length: label_length - chars.length }).map((item) => BLANK)
    );
  } else if (chars.length > label_length) {
    chars = chars.slice(0, label_length);
  }
  return chars;
}

export const ctcLossConfig: KernelConfig = {
  kernelName: CtcLoss,
  backendName: "cpu",
  kernelFunc: (cropAndResize as {}) as KernelFunc,
};
