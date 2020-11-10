 # How to modularize an op

_This document generally describes the world that we are moving towards as we modularise tfjs. Some of its descriptions hold for the current state of the world._ It's primarily a pragmatic workflow guide, you don't have to follow the steps in this exact order, but it can be a helpful starting place/checklist.

Glossary

**Op**: In TensorFlow.js an op is a backend agnostic function that is generally exposed as public API to end users. These are implemented in `tfjs-core`

**Kernel**: In TensorFlow.js a kernel is a backend specific low level implementation of functionality that is used by one or more ops. The kernels and their interfaces that are available in tfjs are defined in `tfjs-core/src/kernel_names.ts`. Kernels should not call other kernels nor call back into the public API of tfjs. Kernels may share code as regular function imports.

**Gradient**: The definition of backward mode operation for a given **Kernel**. These are implemented in tfjs-core and are also backend agnostic (i.e. they call other ops or kernels).

**runKernelFunc**: A function in tfjs-core's engine that executes functions. It can handle both modular and non-modular kernels (non-modular kernels are kernels called through the backend object rather than the kernel registry). Will be replaced with **runKernel** once all kernels are modular across all backends.

## Steps in tfjs-core

Note: We will be modularising **all the ops** before modularizing **any of** the kernels in the various backends.

**Before you start:** Go to [this issue](https://github.com/tensorflow/tfjs/issues/2822) and tell us which op you want to work on by leaving a comment.

- Add necessary kernel names and interfaces to **tfjs-core/src/kernel_names.ts**

    This **must** include an identifier for the kernel and optionally types for the `Inputs` and `Attrs`. Use these identifiers in Op and Kernel definitions. As closely as possible we want to match the interface defined by the [C++ API](https://www.tensorflow.org/api_docs/cc). We cannot always match exactly, so reach out for guidance if you are unsure.

    ```ts
    export const SquaredDifference = 'SquaredDifference';
    export type SquaredDifferenceInputs = Pick<NamedTensorInfoMap, 'a'|'b'>;
    ```

- Create `src/ops/op_name.ts`

    Move op definition into this file. e.g. `tfjs-core/src/ops/squared_difference.ts`

    Note that we still use runKernelFunc to support backends that haven't yet modularized their kernels. This the forward and backward function will be defined here as well as in the modular kernel(s)/gradient(s).

    Generally ops should only do input validation and data transformations to make the parameters **match the kernel interface.** Any other data transformation should be done by kernels. The guiding principle here is that the work of the kernel (as defined by its interface) should not be split between an op and a kernel definition. Note: in some cases when you move responsibility for data manipulation from ops to kernels, _older_ modualized kernels will break (i.e. fail their tests, e.g. kernels in the wasm backend). In these cases you also need to go in and adjust those kernels to match the new input.

    ```ts
    import {ENGINE, ForwardFunc} from '../engine';
    import {SquaredDifference, SquaredDifferenceInputs} from '../kernel_names';
    import {Tensor} from '../tensor';
    import {NamedTensorMap} from '../tensor_types';
    import {makeTypesMatch} from '../tensor_util';
    import {convertToTensor} from '../tensor_util_env';
    import {TensorLike} from '../types';

    import {assertAndGetBroadcastShape} from './broadcast_util';
    import {op} from './operation';
    import {scalar} from './tensor_ops';


    function squaredDifference_<T extends Tensor>(
        a: Tensor|TensorLike, b: Tensor|TensorLike): T {
      let $a = convertToTensor(a, 'a', 'squaredDifference');
      let $b = convertToTensor(b, 'b', 'squaredDifference');
      [$a, $b] = makeTypesMatch($a, $b);

      assertAndGetBroadcastShape($a.shape, $b.shape);
      // ****************
      // Modularization note: this gradient definition should be removed from
      // here once the modular gradient is implemented in the steps below.
      //*****************
      const der = (dy: Tensor, saved: Tensor[]) => {
        const [$a, $b] = saved;
        const two = scalar(2);
        const derA = () => dy.mul($a.sub($b).mul(two));
        const derB = () => dy.mul($b.sub($a).mul(two));
        return {a: derA, b: derB};
      };
      // ****************
      // END Modularization note
      //*****************


      const forward: ForwardFunc<Tensor> = (backend, save) => {
        const res = backend.squaredDifference($a, $b);
        save([$a, $b]);
        return res;
      };

      const inputs: SquaredDifferenceInputs = {a: $a, b: $b};
      const attrs = {};

      const inputsToSave = [$a, $b];
      const outputToSave: boolean[] = [];
      return ENGINE.runKernelFunc(
                 forward, inputs as unknown as NamedTensorMap, der,
                 SquaredDifference, attrs, inputsToSave, outputToSave) as T;
    }

    export const squaredDifference = op({squaredDifference_});
    ```



- Export modularized op from `src/ops/ops.ts`

    ```ts
    export {squaredDifference} from './squared_difference';
    ```

- Make chained op augmentor in `src/public/chained_ops/op_name.ts`

    `src/public/chained_ops/squared_difference.ts`

    ```ts

    import {squaredDifference} from '../../ops/squared_difference';
    import {Tensor} from '../../tensor';
    import {Rank, TensorLike} from '../../types';

    declare module '../../tensor' {
      interface Tensor<R extends Rank = Rank> {
        squaredDifference<T extends Tensor>(b: Tensor|TensorLike): T;
      }
    }

    Tensor.prototype.squaredDifference = function<T extends Tensor>(b: Tensor|
                                                                    TensorLike): T {
      this.throwIfDisposed();
      return squaredDifference(this, b);
    };
    ```

- Add augmentor to `src/public/chained_ops/register_all_chained_ops.ts`
- Add chained op test to `src/public/chained_ops/register_all_chained_ops_test.ts`
- Remove op from `src/tensor.ts`
  - Remove it from the `Tensor` class and from the `OpHandler` interface

- Create `src/gradients/kernel_name` for any kernels used that do not already have a modular gradient

    e.g. `src/gradients/SquaredDifference_grad.ts`

    Note that we use **directly imported ops**. Avoid using the chained API

    ```ts
    import {SquaredDifference} from '../kernel_names';
    import {GradConfig} from '../kernel_registry';
    import {mul, sub} from '../ops/binary_ops';
    import {scalar} from '../ops/tensor_ops';
    import {Tensor} from '../tensor';

    export const squaredDifferenceGradConfig: GradConfig = {
      kernelName: SquaredDifference,
      gradFunc: (dy: Tensor, saved: Tensor[]) => {
        const [a, b] = saved;
        const two = scalar(2);
        const derA = () => mul(dy, mul(two, sub(a, b)));
        const derB = () => mul(dy, mul(two, sub(b, a)));
        return {a: derA, b: derB};
      }
    };
    ```

- Add gradient config to `src/register_all_gradients.ts`

    ```ts
    import {squaredDifferenceGradConfig} from './gradients/SquaredDifference_grad';

    const gradConfigs: GradConfig[] = [
      // add the gradient config to this list.
      squaredDifferenceGradConfig,
    ];
    ```

## Submit a PR

At this stage you should be able to submit a PR for review. Don't forget to run `yarn test` locally within `tfjs-core` before you do!
