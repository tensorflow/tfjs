import {Array1D, DataTypes, NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputArrays, KernelInputConfig, KernelNode} from '../kernel_config';

// Values
export interface TopKValuesNode<D extends keyof DataTypes, T extends NDArray<D>>
    extends KernelNode {
  inputAndArgs: TopKValuesInputConfig<T>;
  output: Array1D<D>;
}

export interface TopKValuesInputConfig<T extends NDArray> extends
    KernelInputConfig {
  inputs: TopKValuesInputArrays<T>;
  args: {k: number};
}

export interface TopKValuesInputArrays<T extends NDArray> extends
    KernelInputArrays {
  x: T;
}

// Indices
export interface TopKIndicesNode extends KernelNode {
  inputAndArgs: TopKIndicesInputConfig;
  output: Array1D<'int32'>;
}

export interface TopKIndicesInputConfig extends KernelInputConfig {
  inputs: TopKIndicesInputArrays;
  args: {k: number};
}

export interface TopKIndicesInputArrays extends KernelInputArrays {
  x: NDArray;
}
