import {Conv2DInfo} from '../../conv_util';
import {Array1D, Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// Conv2D
export interface Conv2DNode extends KernelNode {
  inputAndArgs: Conv2DInputConfig;
  output: Array4D;
}

export interface Conv2DInputConfig extends KernelInputConfig {
  inputs: Conv2DInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DInputArrays extends KernelInputArrays {
  x: Array4D;
  filter: Array4D;
  bias?: Array1D;
}

// Conv2DDerInput
export interface Conv2DDerInputNode extends KernelNode {
  inputAndArgs: Conv2DDerInputInputConfig;
  output: Array4D;
}

export interface Conv2DDerInputInputConfig extends KernelInputConfig {
  inputs: Conv2DDerInputInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DDerInputInputArrays extends KernelInputArrays {
  dy: Array4D;
  filter: Array4D;
}

// Conv2DDerFilter
export interface Conv2DDerFilterNode extends KernelNode {
  inputAndArgs: Conv2DDerFilterInputConfig;
  output: Array4D;
}

export interface Conv2DDerFilterInputConfig extends KernelInputConfig {
  inputs: Conv2DDerFilterInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DDerFilterInputArrays extends KernelInputArrays {
  x: Array4D;
  dy: Array4D;
}

// Conv2DDerBias
export interface Conv2DDerBiasNode extends KernelNode {
  inputAndArgs: Conv2DDerBiasInputConfig;
  output: Array1D;
}

export interface Conv2DDerBiasInputConfig extends KernelInputConfig {
  inputs: Conv2DDerBiasInputArrays;
}

export interface Conv2DDerBiasInputArrays extends KernelInputArrays {
  dy: Array4D;
}

// DepthwiseConv2D
export interface DepthwiseConv2DNode extends KernelNode {
  inputAndArgs: DepthwiseConv2DInputConfig;
  output: Array4D;
}

export interface DepthwiseConv2DInputConfig extends KernelInputConfig {
  inputs: DepthwiseConv2DInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface DepthwiseConv2DInputArrays extends KernelInputArrays {
  x: Array4D;
  filter: Array4D;
}
