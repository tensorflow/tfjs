import {Array1D, Array2D, Array3D, Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// 1D
export interface Slice1DNode extends KernelNode {
  inputAndArgs: Slice1DInputConfig;
  output: Array1D;
}

export interface Slice1DInputConfig extends KernelInputConfig {
  inputs: Slice1DInputArrays;
  args: {begin: number; size: number;};
}

export interface Slice1DInputArrays extends KernelInputArrays { x: Array1D; }

// 2D
export interface Slice2DNode extends KernelNode {
  inputAndArgs: Slice2DInputConfig;
  output: Array2D;
}

export interface Slice2DInputConfig extends KernelInputConfig {
  inputs: Slice2DInputArrays;
  args: {begin: [number, number]; size: [number, number];};
}

export interface Slice2DInputArrays extends KernelInputArrays { x: Array2D; }

// 3D
export interface Slice3DNode extends KernelNode {
  inputAndArgs: Slice3DInputConfig;
  output: Array3D;
}

export interface Slice3DInputConfig extends KernelInputConfig {
  inputs: Slice3DInputArrays;
  args: {begin: [number, number, number]; size: [number, number, number];};
}

export interface Slice3DInputArrays extends KernelInputArrays { x: Array3D; }

// 4D
export interface Slice4DNode extends KernelNode {
  inputAndArgs: Slice4DInputConfig;
  output: Array4D;
}

export interface Slice4DInputConfig extends KernelInputConfig {
  inputs: Slice4DInputArrays;
  args: {
    begin: [number, number, number, number];
    size: [number, number, number, number];
  };
}

export interface Slice4DInputArrays extends KernelInputArrays { x: Array4D; }
