import {registerKernel} from '../../kernel_registry';
import {kernelConfigs} from './all_kernels';

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
