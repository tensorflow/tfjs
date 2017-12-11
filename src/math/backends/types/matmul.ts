import {Array2D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

export interface MatMulNode extends KernelNode {
  inputAndArgs: MatMulInputConfig;
  output: Array2D;
}

export interface MatMulInputConfig extends KernelInputConfig {
  inputs: MatMulInputArrays;
  args: {aOrientation: MatrixOrientation; bOrientation: MatrixOrientation};
}

export interface MatMulInputArrays extends KernelInputArrays {
  a: Array2D;
  b: Array2D;
}

export enum MatrixOrientation {
  REGULAR,
  TRANSPOSED
}
