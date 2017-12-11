import {NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

export interface PowNode<T extends NDArray> extends KernelNode {
  inputAndArgs: PowInputConfig<T>;
  output: T;
}

export interface PowInputConfig<T extends NDArray> extends KernelInputConfig {
  inputs: PowInputArrays<T>;
}

export interface PowInputArrays<T extends NDArray> extends KernelInputArrays {
  a: T;
  b: NDArray<'int32'>;
}
