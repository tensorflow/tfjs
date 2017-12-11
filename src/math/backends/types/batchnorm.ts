
import {Array1D, Array2D, Array3D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// 3D
export interface BatchNorm3DNode extends KernelNode {
  inputAndArgs: BatchNorm3DInputConfig;
  output: Array3D;
}

export interface BatchNorm3DInputConfig extends KernelInputConfig {
  inputs: BatchNorm3DInputArrays;
  args: {varianceEpsilon: number};
}

export interface BatchNorm3DInputArrays extends KernelInputArrays {
  x: Array3D;
  mean: Array3D|Array1D;
  variance: Array3D|Array1D;
  scale?: Array3D|Array1D;
  offset?: Array3D|Array1D;
}

// 2D
export interface BatchNorm2DNode extends KernelNode {
  inputAndArgs: BatchNorm2DInputConfig;
  output: Array2D;
}

export interface BatchNorm2DInputConfig extends KernelInputConfig {
  inputs: BatchNorm2DInputArrays;
  args: {varianceEpsilon: number};
}

export interface BatchNorm2DInputArrays extends KernelInputArrays {
  x: Array2D;
  mean: Array2D|Array1D;
  variance: Array2D|Array1D;
  scale?: Array2D|Array1D;
  offset?: Array2D|Array1D;
}
