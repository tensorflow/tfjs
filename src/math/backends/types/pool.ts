import {Conv2DInfo} from '../../conv_util';
import {Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// Pool
export interface PoolNode extends KernelNode {
  inputAndArgs: PoolInputConfig;
  output: Array4D;
}

export interface PoolInputConfig extends KernelInputConfig {
  inputs: PoolInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface PoolInputArrays extends KernelInputArrays { x: Array4D; }

// PoolBackprop
export interface PoolBackpropNode extends KernelNode {
  inputAndArgs: PoolInputConfig;
  output: Array4D;
}

export interface PoolBackpropInputConfig extends KernelInputConfig {
  inputs: PoolBackpropInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface PoolBackpropInputArrays extends KernelInputArrays {
  dy: Array4D;
  x: Array4D;
}
