import {DataTypes, NDArray} from '../../ndarray';
import {SumTypes} from '../../types';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

export interface SumNode<T extends keyof DataTypes> extends KernelNode {
  inputAndArgs: SumInputConfig<T>;
  output: NDArray<SumTypes[T]>;
}

export interface SumInputConfig<T extends keyof DataTypes> extends
    KernelInputConfig {
  inputs: SumInputArrays<T>;
  args: {axes: number[];};
}

export interface SumInputArrays<T extends keyof DataTypes> extends
    KernelInputArrays {
  x: NDArray<T>;
}
