import {Array1D, Array2D, Array3D, Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// 1D
export interface Concat1DNode extends KernelNode {
  inputAndArgs: Concat1DInputConfig;
  output: Array1D;
}

export interface Concat1DInputConfig extends KernelInputConfig {
  inputs: Concat1DInputArrays;
}

export interface Concat1DInputArrays extends KernelInputArrays {
  a: Array1D;
  b: Array1D;
}

// 2D
export interface Concat2DNode extends KernelNode {
  inputAndArgs: Concat2DInputConfig;
  output: Array2D;
}

export interface Concat2DInputConfig extends KernelInputConfig {
  inputs: Concat2DInputArrays;
  args: {axis: number};
}

export interface Concat2DInputArrays extends KernelInputArrays {
  a: Array2D;
  b: Array2D;
}

// 3D
export interface Concat3DNode extends KernelNode {
  inputAndArgs: Concat3DInputConfig;
  output: Array3D;
}

export interface Concat3DInputConfig extends KernelInputConfig {
  inputs: Concat3DInputArrays;
  args: {axis: number};
}

export interface Concat3DInputArrays extends KernelInputArrays {
  a: Array3D;
  b: Array3D;
}

// 4D
export interface Concat4DNode extends KernelNode {
  inputAndArgs: Concat4DInputConfig;
  output: Array4D;
}

export interface Concat4DInputConfig extends KernelInputConfig {
  inputs: Concat4DInputArrays;
  args: {axis: number};
}

export interface Concat4DInputArrays extends KernelInputArrays {
  a: Array4D;
  b: Array4D;
}
