import {registerKernel} from '../../';
import {kernelConfigs} from './all_kernels';

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
