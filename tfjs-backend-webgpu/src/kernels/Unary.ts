// tslint:disable-next-line: no-imports-from-dist
import { SimpleUnaryImpl as ImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/utils/unary_types';
import { DataType, KernelFunc, TensorInfo, TypedArray, UnaryInputs, registerKernel } from '@tensorflow/tfjs-core';
import { Abs, Ceil, Cos, Cosh, Elu, Exp, Expm1, Floor, Log, LogicalNot, Neg, NegInputs, Relu, Relu6, Rsqrt, Sigmoid, Sin, Sinh, Sqrt, Square, Tanh } from '@tensorflow/tfjs-core';

import { WebGPUBackend } from '../backend_webgpu';
import { simpleAbsImplCPU, ceilImplCPU, expImplCPU, expm1ImplCPU, floorImplCPU, logImplCPU, negImplCPU, rsqrtImplCPU } from '../kernel_utils/shared';
import { getMainHeaderAndGlobalIndexString } from '../shader_preprocessor';
import { computeDispatch, flatDispatchLayout } from '../webgpu_util';

import { WebGPUProgram } from './webgpu_program';

export const kernelInfo: { [key: string]: { 'shader': string, 'implCPU'?: ImplCPU, 'dtype'?: DataType } } = {
  [Abs]: {
    'shader': 'return abs(a);',
    'implCPU': simpleAbsImplCPU
  },
  [Ceil]: {
    'shader': 'return ceil(a);',
    'implCPU': ceilImplCPU
  },
  [Cos]: {
    'shader': 'return cos(a);'
  },
  [Cosh]: {
    'shader': 'let e2x = exp(-a); return (e2x + 1.0 / e2x) / 2.0;'
  },
  [Elu]: {
    'shader': 'if (a >= 0.0) { return a; } return (exp(a) - 1.0);'
  },
  [Exp]: {
    'shader': 'return exp(a);',
    'implCPU': expImplCPU,
    'dtype': 'float32'
  },
  [Expm1]: {
    'shader': 'return exp(a) - 1.0;',
    'implCPU': expm1ImplCPU
  },
  [Floor]: {
    'shader': 'return floor(a);',
    'implCPU': floorImplCPU
  },
  [Log]: {
    'shader': `if (a < 0.0) { return 1.0/0.0; }
    return log(a);`,
    'implCPU': logImplCPU
  },
  [LogicalNot]: {
    'shader': 'return f32(!(a >= 1.0));'
  },
  [Neg]: {
    'shader': 'return -a;'
  },
  [Relu]: {
    'shader': 'return max(a, 0.0);'
  },
  [Relu6]: {
    'shader': 'return clamp(a, 0.0, 6.0);'
  },
  [Rsqrt]: {
    'shader': 'return 1.0/sqrt(a);',
    'implCPU': rsqrtImplCPU
  },
  [Sigmoid]: {
    'shader': 'return 1.0 / (1.0 + exp(-1.0 * a));'
  },
  [Sin]: {
    'shader': 'return sin(a);'
  },
  [Sinh]: {
    'shader': `let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;`
  },
  [Sqrt]: {
    'shader': 'return sqrt(a);'
  },
  [Square]: {
    'shader': 'return a * a;'
  },
  [Tanh]: {
    'shader': `let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);`
  },
};

const shaders: { [key: string]: string } = {};
for (const kernelName in kernelInfo) {
  shaders[kernelName] = kernelInfo[kernelName]['shader'];
}
shaders['Elu_vec4'] = `var resFloat = exp(a) - vec4<f32>(1.0);
  if (a.r >= 0.0) {
    resFloat.r = a.r;
  }
  if (a.g >= 0.0) {
    resFloat.g = a.g;
  }
  if (a.b >= 0.0) {
    resFloat.b = a.b;
  }
  if (a.a >= 0.0) {
    resFloat.a = a.a;
  }
  return resFloat;`;
shaders['Int'] = 'return f32(i32((a)));';
shaders['Linear'] = 'return a;';
shaders['Relu_vec4'] = `var resFloat = a * vec4<f32>(a >= vec4<f32>(0.0));
  let isNaN = isNan(a);

  if (isNaN.r) {
    resFloat.r = a.r;
  }
  if (isNaN.g) {
    resFloat.g = a.g;
  }
  if (isNaN.b) {
    resFloat.b = a.b;
  }
  if (isNaN.a) {
    resFloat.a = a.a;
  }
  return resFloat;`;
shaders['Relu6_vec4'] = 'return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));';

class UnaryProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: { x: number[] };
  dispatch: [number, number, number];
  variableNames = ['A'];
  workGroupSize: [number, number, number];
  kernelName: string;
  size = true;

  constructor(outputShape: number[], kernelName: string) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
      this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.kernelName = kernelName;
    this.shaderKey = `unary_${kernelName}`;
  }

  getUserCode(): string {
    return `
      fn unaryKernel(a : f32) -> f32 {
        ${shaders[this.kernelName]}
      }
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let a = getAAtOutCoordsByGlobalIndex(index);
          setOutputFlat(index, unaryKernel(a));
        }
      }
      `;
  }
}

type UnaryConfig = {
  kernelName: string,
  implCPU?: ImplCPU,
  dtype?: DataType
};

/**
 * Template that creates a `KernelFunc` for unary kernels.
 * @param kernelName kernel name to create `UnaryProgram`.
 * @param implCPU Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function unaryFunc(
  { kernelName, implCPU, dtype }: UnaryConfig): KernelFunc {
  return ({ inputs, backend }) => {
    const { x } = inputs as UnaryInputs;
    const webgpuBackend = backend as WebGPUBackend;

    const $dtype = dtype || x.dtype;
    if (webgpuBackend.shouldExecuteOnCPU([x]) && implCPU != null) {
      const xData = webgpuBackend.tensorMap.get(x.dataId);
      const outValues = implCPU(xData.values as TypedArray, $dtype);
      return webgpuBackend.makeTensorInfo(x.shape, $dtype, outValues);
    }

    const program: UnaryProgram = new UnaryProgram(x.shape, kernelName);
    return webgpuBackend.runWebGPUProgram(program, [x], $dtype);
  };
}

// This doesn't use unaryFunc because negImplCPU is not of type ImplCPU.
function neg(args: { inputs: NegInputs, backend: WebGPUBackend }):
  TensorInfo {
  const { inputs, backend } = args;
  const { x } = inputs;

  if (backend.shouldExecuteOnCPU([x])) {
    const xData = backend.tensorMap.get(x.dataId);
    const [outValues, newShape] =
      negImplCPU(xData.values as TypedArray, x.shape, x.dtype);
    return backend.makeTensorInfo(newShape, x.dtype, outValues);
  }

  const program = new UnaryProgram(x.shape, Neg);

  return backend.runWebGPUProgram(program, [x], x.dtype);
}

export function registerKernelsUnary() {
  for (const kernelName in kernelInfo) {
    if (kernelName === Neg) {
      continue;
    }
    registerKernel({
      kernelName,
      backendName: 'webgpu',
      kernelFunc: unaryFunc({
        kernelName,
        implCPU: kernelInfo[kernelName]['implCPU'],
        dtype: kernelInfo[kernelName]['dtype']
      }),
    });
  }
  registerKernel({
    kernelName: Neg,
    backendName: 'webgpu',
    kernelFunc: neg,
  });
}

export function getShaderUnary(kernelName: string) {
  return shaders[kernelName];
}

export const exp = unaryFunc({
  kernelName: Exp,
  implCPU: expImplCPU,
  dtype: 'float32',
});

export function int(input: TensorInfo, backend: WebGPUBackend): TensorInfo {
  const program = new UnaryProgram(input.shape, 'Int');
  const output = backend.runWebGPUProgram(program, [input], 'int32');
  return { dataId: output.dataId, shape: output.shape, dtype: output.dtype };
}
