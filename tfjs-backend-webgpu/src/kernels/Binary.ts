
// tslint:disable-next-line: no-imports-from-dist
import { SimpleBinaryKernelImpl as ImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/utils/binary_types';
import { BinaryInputs, DataType, KernelFunc, TensorInfo, TypedArray, backend_util, registerKernel, upcastType } from '@tensorflow/tfjs-core';
import { Add, Complex, ComplexInputs, Equal, FloorDiv, Greater, GreaterEqual, Less, LessEqual, LogicalAnd, Maximum, Minimum, Multiply, NotEqual, Pow, Prelu, PreluInputs, RealDiv, SquaredDifference, Sub } from '@tensorflow/tfjs-core';

import { getMainHeaderAndGlobalIndexString } from '../shader_preprocessor';
import { WebGPUBackend } from '../backend_webgpu';
import { addImplCPU, multiplyImplCPU, subImplCPU, equalImplCPU, greaterImplCPU, greaterEqualImplCPU, lessImplCPU, lessEqualImplCPU, maximumImplCPU, minimumImplCPU, notEqualImplCPU } from '../kernel_utils/shared';
import { computeDispatch, flatDispatchLayout } from '../webgpu_util';

import { identity } from './Identity';
import { WebGPUProgram } from './webgpu_program';

const CHECK_NAN_SNIPPET = `
if (isNanCustom(a)) { return a; }
if (isNanCustom(b)) { return b; }`;

const CHECK_NAN_SNIPPET_VEC4 = `
if (isNaN.r > 0.) {
  resultTemp.r = uniforms.NAN;
}
if (isNaN.g > 0.) {
  resultTemp.g = uniforms.NAN;
}
if (isNaN.b > 0.) {
  resultTemp.b = uniforms.NAN;
}
if (isNaN.a > 0.) {
  resultTemp.a = uniforms.NAN;
}`;

function getMinMaxString(kernelName: string, useVec4: boolean) {
  const checkNanSnippet = useVec4 ? CHECK_NAN_SNIPPET_VEC4 : CHECK_NAN_SNIPPET;
  return useVec4 ? `
    var resultTemp = vec4<f32>(${kernelName}(a, b));
    let isNaN = min(vec4<f32>(isNanCustomVec4F32(a)) + vec4<f32>(isNanCustomVec4F32(b)), vec4<f32>(1.0));
    ` + checkNanSnippet +
    `
    return resultTemp;
  ` :
    checkNanSnippet + `
    return ${kernelName}(a, b);
  `;
}

const kernelInfo: { [key: string]: { 'shader': string, 'implCPU'?: ImplCPU, 'dtype'?: DataType, 'supportsComplex'?: boolean } } = {
  [Add]: {
    'shader': 'return a + b;',
    'implCPU': addImplCPU,
    'supportsComplex': true
  },
  [Equal]: {
    'shader': 'return f32(a == b);',
    'implCPU': equalImplCPU,
    'dtype': 'bool'
  },
  [FloorDiv]: {
    'shader': `
  let s = sign(a) * sign(b);
  let ia = i32(round(a));
  let ib = i32(round(b));
  return f32(idiv(ia, ib, s));`,
    'dtype': 'int32'
  },
  [Greater]: {
    'shader': 'return f32(a > b);',
    'implCPU': greaterImplCPU,
    'dtype': 'bool'
  },
  [GreaterEqual]: {
    'shader': 'return f32(a >= b);',
    'implCPU': greaterEqualImplCPU,
    'dtype': 'bool'
  },
  [Less]: {
    'shader': 'return f32(a < b);',
    'implCPU': lessImplCPU,
    'dtype': 'bool'
  },
  [LessEqual]: {
    'shader': 'return f32(a <= b);',
    'implCPU': lessEqualImplCPU,
    'dtype': 'bool'
  },
  [LogicalAnd]: {
    'shader': 'return f32(f32(a) >= 1.0 && f32(b) >= 1.0);',
    'dtype': 'bool'
  },
  [Maximum]: {
    'shader': `${getMinMaxString('max', false)}`,
    'implCPU': maximumImplCPU
  },
  [Minimum]: {
    'shader': `${getMinMaxString('min', false)}`,
    'implCPU': minimumImplCPU
  },
  [Multiply]: {
    'shader': 'return a * b;',
    'implCPU': multiplyImplCPU,
    'supportsComplex': true
  },
  [NotEqual]: {
    'shader': 'return f32(a != b);',
    'implCPU': notEqualImplCPU,
    'dtype': 'bool'
  },
  [Pow]: {
    'shader': `
  if(a < 0.0 && floor(b) < b) {
    return uniforms.NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  if (round(abs(b) % 2.0) != 1.0) {
    return pow(abs(a), b);
  }
  return sign(a) * pow(abs(a), b);`
  },
  [Prelu]: {
    'shader': 'if (a < 0.0) { return b * a; }  return a;'
  },
  [RealDiv]: {
    'shader': 'return a / b;'
  },
  [SquaredDifference]: {
    'shader': 'return (a - b) * (a - b);'
  },
  [Sub]: {
    'shader': 'return a - b;',
    'implCPU': subImplCPU,
    'supportsComplex': true
  },
};

const shaders: { [key: string]: string } = {};
for (const kernelName in kernelInfo) {
  shaders[kernelName] = kernelInfo[kernelName]['shader'];
}
shaders['Complex_multiply_real'] = 'return areal * breal - aimag * bimag;';
shaders['Complex_multiply_imag'] = 'return areal * bimag + aimag * breal;';
shaders['Equal_vec4'] = 'return vec4<f32>(a == b);';
shaders['Floor_div_vec4'] = `
  let ia = vec4<i32>(round(a));
  let ib = vec4<i32>(round(b));
  let cond = ib != vec4<i32>(0);
  var resultTemp = vec4<i32>(0);
  let s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    resultTemp[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    resultTemp[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    resultTemp[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    resultTemp[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4<f32>(resultTemp);`;
shaders['Greater_vec4'] = 'return vec4<f32>(a > b);';
shaders['Greater_equal_vec4'] = 'return vec4<f32>(a >= b);';
shaders['Less_vec4'] = 'return vec4<f32>(a < b);';
shaders['Less_equal_vec4'] = 'return vec4<f32>(a <= b);';
shaders['Logial_and_vec4'] = 'return (vec4<f32>(a >= vec4<f32>(1.0)) * vec4<f32>(b >= vec4<f32>(1.0)));';
shaders['Maximum_vec4'] = `${getMinMaxString('max', true)}`;
shaders['Minimum_vec4'] = `${getMinMaxString('min', true)}`;
shaders['Not_equal_vec4'] = 'return vec4<f32>(a != b);';
shaders['Pow_vec4'] = `
let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
let isModRound1 = vec4<f32>(isModRound1Bool);
let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
var resultTemp = multiplier * pow(abs(a), b);

// Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
let isExpZero = b == vec4<f32>(0.0);
if (isExpZero.r) {
  resultTemp.r = 1.0;
}
if (isExpZero.g) {
  resultTemp.g = 1.0;
}
if (isExpZero.b) {
  resultTemp.b = 1.0;
}
if (isExpZero.a) {
  resultTemp.a = 1.0;
}
let isNaN = vec4<f32>(a < vec4<f32>(0.0)) * vec4<f32>(floor(b) < b);
${CHECK_NAN_SNIPPET_VEC4}
return resultTemp;`;
shaders['Prelu_vec4'] = `
let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);`;

export function getShaderBinary(kernelName: string) {
  return shaders[kernelName];
}

class BinaryProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: { x: number[] };
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number];
  kernelName: string;
  size = true;

  constructor(kernelName: string, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
      this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `binary_${kernelName}`;
    this.kernelName = kernelName;
  }

  getUserCode(): string {

    const userCode = `
      fn binaryOperation(a : f32, b : f32) -> f32 {
        ${shaders[this.kernelName]}
      }
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let a = getAAtOutCoordsByGlobalIndex(index);
          let b = getBAtOutCoordsByGlobalIndex(index);
          setOutputFlat(index, binaryOperation(a, b));
        }
      }
      `;
    return userCode;
  }
}

type BinaryConfig = {
  kernelName?: string,
  shader?: string,
  implCPU?: ImplCPU,
  dtype?: DataType,
  supportsComplex?: boolean,
};

/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param kernelName Kernel name to create `BinaryProgram`.
 * @param implCPU Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function binaryFunc(
  { kernelName, implCPU, dtype, supportsComplex = false }:
    BinaryConfig): KernelFunc {
  return ({ inputs, backend }) => {
    const { a, b } = inputs as BinaryInputs;
    const webgpuBackend = backend as WebGPUBackend;

    if (supportsComplex && a.dtype === 'complex64') {
      const aData = webgpuBackend.tensorMap.get(a.dataId);
      const bData = webgpuBackend.tensorMap.get(b.dataId);
      let real: TensorInfo, imag: TensorInfo;
      if (kernelName !== Multiply) {
        [real, imag] = [
          [aData.complexTensorInfos.real, bData.complexTensorInfos.real],
          [aData.complexTensorInfos.imag, bData.complexTensorInfos.imag]
        ].map(complexParts => {
          const [aPart, bPart] = complexParts;

          const aHandle = {
            dataId: aPart.dataId,
            dtype: aPart.dtype,
            shape: a.shape
          };
          const bHandle = {
            dataId: bPart.dataId,
            dtype: bPart.dtype,
            shape: b.shape
          };

          const program = new BinaryProgram(kernelName, a.shape, b.shape);
          return webgpuBackend.runWebGPUProgram(
            program, [aHandle, bHandle],
            upcastType(aPart.dtype, bPart.dtype));
        });
      } else {
        const realProgram = new BinaryComplexProgram(
          'Complex_multiply_real', a.shape, b.shape);
        const imagProgram = new BinaryComplexProgram(
          'Complex_multiply_imag', a.shape, b.shape);

        const inputs = [
          {
            dataId: aData.complexTensorInfos.real.dataId,
            dtype: aData.complexTensorInfos.real.dtype,
            shape: a.shape
          },
          {
            dataId: aData.complexTensorInfos.imag.dataId,
            dtype: aData.complexTensorInfos.imag.dtype,
            shape: a.shape
          },
          {
            dataId: bData.complexTensorInfos.real.dataId,
            dtype: bData.complexTensorInfos.real.dtype,
            shape: b.shape
          },
          {
            dataId: bData.complexTensorInfos.imag.dataId,
            dtype: bData.complexTensorInfos.imag.dtype,
            shape: b.shape
          }
        ];

        real = webgpuBackend.runWebGPUProgram(realProgram, inputs, 'float32');
        imag = webgpuBackend.runWebGPUProgram(imagProgram, inputs, 'float32');
      }

      const complexOutput =
        complex({ inputs: { real, imag }, backend: webgpuBackend });

      webgpuBackend.disposeData(real.dataId);
      webgpuBackend.disposeData(imag.dataId);

      // TODO: Implement CPU forwarding for complex inputs.

      return complexOutput;
    }

    const $dtype = dtype || upcastType(a.dtype, b.dtype);
    if ((a.dtype === 'string' || b.dtype === 'string' ||
      webgpuBackend.shouldExecuteOnCPU([a, b])) &&
      implCPU != null) {
      const aData = webgpuBackend.tensorMap.get(a.dataId).values as TypedArray;
      const bData = webgpuBackend.tensorMap.get(b.dataId).values as TypedArray;
      const decodedAVals = a.dtype === 'string' ?
        // tslint:disable-next-line: no-any
        backend_util.fromUint8ToStringArray(aData as any as Uint8Array[]) :
        aData;
      const decodedBVals = a.dtype === 'string' ?
        // tslint:disable-next-line: no-any
        backend_util.fromUint8ToStringArray(bData as any as Uint8Array[]) :
        bData;
      const [outValues, outShape] =
        implCPU(a.shape, b.shape, decodedAVals, decodedBVals, $dtype);

      return webgpuBackend.makeTensorInfo(outShape, $dtype, outValues);
    }
    const program = new BinaryProgram(kernelName, a.shape, b.shape);
    return webgpuBackend.runWebGPUProgram(program, [a, b], $dtype);
  };
}

class BinaryComplexProgram implements WebGPUProgram {
  variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: { x: number[] };
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [128, 1, 1];
  kernelName: string;
  size = true;

  constructor(kernelName: string, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
      this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = `binaryComplex_${kernelName}`;
    this.kernelName = kernelName;
  }

  getUserCode(): string {

    const userCode = `
      fn binaryComplex(
          areal : f32, aimag : f32, breal : f32, bimag : f32) -> f32 {
        ${shaders[this.kernelName]}
      }

      ${getMainHeaderAndGlobalIndexString()}
        if(index < uniforms.size) {
          let areal = getARealAtOutCoordsByGlobalIndex(index);
          let aimag = getAImagAtOutCoordsByGlobalIndex(index);
          let breal = getBRealAtOutCoordsByGlobalIndex(index);
          let bimag = getBImagAtOutCoordsByGlobalIndex(index);
          setOutputFlat(index, binaryComplex(areal, aimag, breal, bimag));
        }
      }
    `;
    return userCode;
  }
}

/**
 * Complex tensors share data with their real and imaginary components. Complex
 * tensors' reference to the components is tracked by refCount on the individual
 * component. The refCounts are increased by the identity call.
 *
 * When a complex tensor is disposed, it will reduce the refCount on the
 * components by calling disposeData on each.
 */
export function complex(args: { inputs: ComplexInputs, backend: WebGPUBackend }):
  TensorInfo {
  const { inputs, backend } = args;
  const { real, imag } = inputs;

  const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
  const complex = backend.tensorMap.get(complexInfo.dataId);

  const realTensorInfo = identity({ inputs: { x: real }, backend });

  const imagTensorInfo = identity({ inputs: { x: imag }, backend });

  complex.complexTensorInfos = { real: realTensorInfo, imag: imagTensorInfo };

  return complexInfo;
}

export function prelu(args: { inputs: PreluInputs, backend: WebGPUBackend }):
  TensorInfo {
  const { inputs, backend } = args;
  const { x, alpha } = inputs;

  const program = new BinaryProgram(Prelu, x.shape, alpha.shape);
  return backend.runWebGPUProgram(program, [x, alpha], 'float32');
}

export function registerKernelsBinary() {
  for (const kernelName in kernelInfo) {
    if ([Complex, Prelu].includes(kernelName)) {
      continue;
    }
    registerKernel({
      kernelName,
      backendName: 'webgpu',
      kernelFunc: binaryFunc({
        kernelName,
        implCPU: kernelInfo[kernelName]['implCPU'],
        dtype: kernelInfo[kernelName]['dtype'],
        supportsComplex: kernelInfo[kernelName]['supportsComplex']
      }),
    });
  }
  registerKernel({
    kernelName: Complex,
    backendName: 'webgpu',
    kernelFunc: complex,
  });
  registerKernel({
    kernelName: Prelu,
    backendName: 'webgpu',
    kernelFunc: prelu,
  });
}

export const multiply = binaryFunc({
  kernelName: Multiply,
  implCPU: multiplyImplCPU,
  supportsComplex: true,
});

export const notEqual = binaryFunc({
  kernelName: NotEqual,
  implCPU: notEqualImplCPU,
  dtype: 'bool',
});

export const realDiv = binaryFunc({
  kernelName: RealDiv,
});

export const sub = binaryFunc({
  kernelName: Sub,
  implCPU: subImplCPU,
  supportsComplex: true,
});
