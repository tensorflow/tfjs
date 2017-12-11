import {NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

export interface ArgMaxNode extends KernelNode {
  inputAndArgs: ArgMaxInputConfig;
  output: NDArray<'int32'>;
}

export interface ArgMaxInputConfig extends KernelInputConfig {
  inputs: ArgMaxInputArrays;
  args: {axes: number[];};
}

export interface ArgMaxInputArrays extends KernelInputArrays { x: NDArray; }

export interface ArgMinNode extends KernelNode {
  inputAndArgs: ArgMinInputConfig;
  output: NDArray<'int32'>;
}

export interface ArgMinInputConfig extends KernelInputConfig {
  inputs: ArgMinInputArrays;
  args: {axes: number[];};
}

export interface ArgMinInputArrays extends KernelInputArrays { x: NDArray; }
