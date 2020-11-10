/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import {backend_util, BackendTimingInfo, DataId, DataType, KernelBackend, ModelTensorInfo, Rank, Scalar, scalar, ScalarLike, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorInfo, tidy, util} from '@tensorflow/tfjs';
import {isArray, isNullOrUndefined} from 'util';

import {encodeInt32ArrayAsInt64, Int64Scalar} from './int64_tensors';
import {TensorMetadata, TFEOpAttr, TFJSBinding} from './tfjs_binding';

type TensorData = {
  shape: number[],
  dtype: number,
  values: backend_util.BackendValues,
  id: number
};

export class NodeJSKernelBackend extends KernelBackend {
  binding: TFJSBinding;
  isGPUPackage: boolean;
  isUsingGpuDevice: boolean;
  private tensorMap: tf.DataStorage<TensorData>;

  constructor(binding: TFJSBinding, packageName: string) {
    super();
    this.binding = binding;
    this.isGPUPackage = packageName === '@tensorflow/tfjs-node-gpu';
    this.isUsingGpuDevice = this.binding.isUsingGpuDevice();
    this.tensorMap = new tf.DataStorage<TensorData>(this, tf.engine());
  }

  getDTypeInteger(dtype: DataType): number {
    switch (dtype) {
      case 'float32':
        return this.binding.TF_FLOAT;
      case 'int32':
        return this.binding.TF_INT32;
      case 'bool':
        return this.binding.TF_BOOL;
      case 'complex64':
        return this.binding.TF_COMPLEX64;
      case 'string':
        return this.binding.TF_STRING;
      default:
        throw new Error(`Unsupported DType: ${dtype}`);
    }
  }

  private typeAttributeFromTensor(value: Tensor): number {
    return this.getDTypeInteger(value.dtype);
  }

  // Creates a new Tensor and maps the dataId to the passed in ID.
  private createOutputTensor(metadata: TensorMetadata): Tensor {
    const newId = {};

    this.tensorMap.set(newId, {
      shape: metadata.shape,
      dtype: metadata.dtype,
      id: metadata.id,
      values: null
    });

    let dtype: DataType;
    switch (metadata.dtype) {
      case this.binding.TF_FLOAT:
        dtype = 'float32';
        break;
      case this.binding.TF_INT32:
        dtype = 'int32';
        break;
      case this.binding.TF_BOOL:
        dtype = 'bool';
        break;
      case this.binding.TF_COMPLEX64:
        dtype = 'complex64';
        break;
      case this.binding.TF_STRING:
        dtype = 'string';
        break;
      case this.binding.TF_RESOURCE:
        // NOTE(cais): We currently represent resource-type Tensors
        // as string of ubytes.
        dtype = 'string';
        break;
      case this.binding.TF_UINT8:
        // TensorFlow uses UINT8 as dtype for image tensor. UINT8 is not
        // supported in TFJS yet, cast it to int32.
        dtype = 'int32';
        break;
      default:
        throw new Error(`Unknown dtype enum ${metadata.dtype}`);
    }

    // TODO(yassogba) Enable this once all the kernels are removed from backend.
    // We can then change the return type from Tensor to TensorInfo.
    // return {dataId: newId, shape: metadata.shape, dtype};

    return tf.engine().makeTensorFromDataId(newId, metadata.shape, dtype);
  }

  // Prepares Tensor instances for Op execution.
  private getInputTensorIds(tensors: Array<TensorInfo|Int64Scalar>): number[] {
    const ids: number[] = [];
    for (let i = 0; i < tensors.length; i++) {
      if (tensors[i] instanceof Int64Scalar) {
        // Then `tensors[i]` is a Int64Scalar, which we currently represent
        // using an `Int32Array`.
        const value = (tensors[i] as Int64Scalar).valueArray;
        const id = this.binding.createTensor([], this.binding.TF_INT64, value);
        ids.push(id);
      } else {
        const info = this.tensorMap.get((tensors[i] as TensorInfo).dataId);
        // TODO - what about ID in this case? Handle in write()??
        if (info.values != null) {
          // Values were delayed to write into the TensorHandle. Do that before
          // Op execution and clear stored values.
          info.id =
              this.binding.createTensor(info.shape, info.dtype, info.values);
          info.values = null;
        }
        ids.push(info.id);
      }
    }
    return ids;
  }

  createReductionOpAttrs(tensor: TensorInfo, keepDims = false): TFEOpAttr[] {
    return [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: keepDims},
      createTensorsTypeOpAttr('T', tensor.dtype),
      createTensorsTypeOpAttr('Tidx', 'int32')
    ];
  }

  floatPrecision(): 16|32 {
    return 32;
  }

  epsilon(): number {
    return super.epsilon();
  }

  /**
   * Executes an op that has a single input and output.
   *
   * Helper function to wrap executeSingleOutput in a particular case.
   * @param name The name of the Op to execute.
   * @param input The input Tensor for the Op.
   */
  executeSingleInput(name: string, input: TensorInfo): Tensor {
    const opAttrs = [createTensorsTypeOpAttr('T', input.dtype)];
    return this.executeSingleOutput(name, opAttrs, [input]);
  }

  /**
   * Executes a TensorFlow Eager Op that provides one output Tensor.
   * @param name The name of the Op to execute.
   * @param opAttrs The list of Op attributes required to execute.
   * @param inputs The list of input Tensors for the Op.
   * @return A resulting Tensor from Op execution.
   */
  executeSingleOutput(name: string, opAttrs: TFEOpAttr[], inputs: TensorInfo[]):
      Tensor {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), 1);
    return this.createOutputTensor(outputMetadata[0]);
  }

  /**
   * Executes a TensorFlow Eager Op that provides multiple output Tensors.
   * @param name The name of the Op to execute.
   * @param opAttrs The list of Op attributes required to execute.
   * @param inputs The list of input Tensors for the Op.
   * @param numOutputs The number of output Tensors for Op execution.
   * @return A resulting Tensor array from Op execution.
   */
  executeMultipleOutputs(
      name: string, opAttrs: TFEOpAttr[], inputs: TensorInfo[],
      numOutputs: number): Tensor[] {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), numOutputs);
    return outputMetadata.map(m => this.createOutputTensor(m));
  }

  numDataIds(): number {
    return this.tensorMap.numDataIds();
  }

  dispose(): void {}

  async read(dataId: DataId): Promise<backend_util.BackendValues> {
    return this.readSync(dataId);
  }

  readSync(dataId: DataId): backend_util.BackendValues {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    if (info.values != null) {
      return info.values;
    } else {
      return this.binding.tensorDataSync(info.id);
    }
  }

  disposeData(dataId: DataId): void {
    // No-op if already disposed.
    if (!this.tensorMap.has(dataId)) {
      return;
    }
    const id = this.tensorMap.get(dataId).id;
    if (id != null && id >= 0) {
      this.binding.deleteTensor(id);
    }
    this.tensorMap.delete(dataId);
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType): void {
    this.tensorMap.set(
        dataId, {shape, dtype: getTFDType(dtype), values, id: -1});
  }

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    const dataId = {};
    this.move(dataId, values, shape, dtype);
    return dataId;
  }

  applyActivation<T extends Tensor>(
      input: T, activation: string, preluActivationWeights?: Tensor): T {
    let result = input;
    if (activation != null) {
      if (activation === 'linear') {
        // No-op
      } else if (activation === 'relu') {
        result = tf.relu(result);
      } else if (activation === 'prelu') {
        result = tf.prelu(result, preluActivationWeights) as T;
      } else if (activation === 'elu') {
        result = tf.elu(result);
      } else if (activation === 'relu6') {
        result = tf.relu6(result);
      } else {
        throw new Error(`Activation: ${
            activation} has not been implemented for the Node.js backend`);
      }
    }
    return result;
  }

  divide(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [createTensorsTypeOpAttr(
        'T', backend_util.upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Div', opAttrs, [a, b]);
  }

  divNoNan(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [createTensorsTypeOpAttr(
        'T', backend_util.upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('DivNoNan', opAttrs, [a, b]);
  }

  where(condition: Tensor): Tensor2D {
    return this.executeSingleOutput('Where', [], [condition]) as Tensor2D;
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  int<T extends Tensor>(x: T): T {
    throw new Error('Method not implemented.');
  }

  // todo(yassogba) consider removing. core does not call this directly
  complexAbs<T extends Tensor>(x: T): T {
    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Tout', 'float32')
    ];
    return this.executeSingleOutput('ComplexAbs', opAttrs, [x]) as T;
  }

  fft(x: Tensor<Rank.R2>): Tensor<Rank.R2> {
    const opAttrs = [createTensorsTypeOpAttr('Tcomplex', x.dtype)];
    return this.executeSingleOutput('FFT', opAttrs, [x]) as Tensor<Rank.R2>;
  }

  ifft(x: Tensor2D): Tensor2D {
    const opAttrs = [createTensorsTypeOpAttr('Tcomplex', x.dtype)];
    return this.executeSingleOutput('IFFT', opAttrs, [x]) as Tensor2D;
  }

  complex<T extends Tensor>(real: T, imag: T): T {
    const opAttrs = [
      createTensorsTypeOpAttr('T', real),
      {
        name: 'Tout',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_COMPLEX64
      },
    ];
    const inputs = [real, imag];
    return this.executeSingleOutput('Complex', opAttrs, inputs) as T;
  }

  real<T extends Tensor>(input: T): T {
    const opAttrs = [
      createTensorsTypeOpAttr('T', input), {
        name: 'Tout',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_FLOAT
      }
    ];
    const inputs = [input];
    return this.executeSingleOutput('Real', opAttrs, inputs) as T;
  }

  imag<T extends Tensor>(input: T): T {
    const opAttrs = [
      {
        name: 'T',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_COMPLEX64
      },
      {
        name: 'Tout',
        type: this.binding.TF_ATTR_TYPE,
        value: this.binding.TF_FLOAT
      }
    ];
    const inputs = [input];
    return this.executeSingleOutput('Imag', opAttrs, inputs) as T;
  }

  decodeJpeg(
      contents: Uint8Array, channels: number, ratio: number,
      fancyUpscaling: boolean, tryRecoverTruncated: boolean,
      acceptableFraction: number, dctMethod: string): Tensor3D {
    const opAttrs = [
      {name: 'channels', type: this.binding.TF_ATTR_INT, value: channels},
      {name: 'ratio', type: this.binding.TF_ATTR_INT, value: ratio}, {
        name: 'fancy_upscaling',
        type: this.binding.TF_ATTR_BOOL,
        value: fancyUpscaling
      },
      {
        name: 'try_recover_truncated',
        type: this.binding.TF_ATTR_BOOL,
        value: tryRecoverTruncated
      },
      {
        name: 'acceptable_fraction',
        type: this.binding.TF_ATTR_FLOAT,
        value: acceptableFraction
      },
      {name: 'dct_method', type: this.binding.TF_ATTR_STRING, value: dctMethod}
    ];
    const inputArgs = [scalar(contents, 'string')];
    return this.executeSingleOutput('DecodeJpeg', opAttrs, inputArgs) as
        Tensor<Rank.R3>;
  }

  decodePng(contents: Uint8Array, channels: number): Tensor3D {
    const opAttrs =
        [{name: 'channels', type: this.binding.TF_ATTR_INT, value: channels}];
    const inputArgs = [scalar(contents, 'string')];

    return this.executeSingleOutput('DecodePng', opAttrs, inputArgs) as
        Tensor<Rank.R3>;
  }

  decodeBmp(contents: Uint8Array, channels: number): Tensor3D {
    const opAttrs =
        [{name: 'channels', type: this.binding.TF_ATTR_INT, value: channels}];
    const inputArgs = [scalar(contents, 'string')];
    return this.executeSingleOutput('DecodeBmp', opAttrs, inputArgs) as
        Tensor<Rank.R3>;
  }

  decodeGif(contents: Uint8Array): Tensor4D {
    const inputArgs = [scalar(contents, 'string')];
    return this.executeSingleOutput('DecodeGif', [], inputArgs) as
        Tensor<Rank.R4>;
  }

  executeEncodeImageOp(
      name: string, opAttrs: TFEOpAttr[], imageData: Uint8Array,
      imageShape: number[]): Tensor {
    const inputTensorId =
        this.binding.createTensor(imageShape, this.binding.TF_UINT8, imageData);
    const outputMetadata =
        this.binding.executeOp(name, opAttrs, [inputTensorId], 1);
    const outputTensorInfo = outputMetadata[0];
    // prevent the tensor data from being converted to a UTF8 string, since
    // the encoded data is not valid UTF8
    outputTensorInfo.dtype = this.binding.TF_UINT8;
    return this.createOutputTensor(outputTensorInfo);
  }

  encodeJpeg(
      imageData: Uint8Array, imageShape: number[], format: ''|'grayscale'|'rgb',
      quality: number, progressive: boolean, optimizeSize: boolean,
      chromaDownsampling: boolean, densityUnit: 'in'|'cm', xDensity: number,
      yDensity: number, xmpMetadata: string): Tensor {
    const opAttrs = [
      {name: 'format', type: this.binding.TF_ATTR_STRING, value: format},
      {name: 'quality', type: this.binding.TF_ATTR_INT, value: quality}, {
        name: 'progressive',
        type: this.binding.TF_ATTR_BOOL,
        value: progressive
      },
      {
        name: 'optimize_size',
        type: this.binding.TF_ATTR_BOOL,
        value: optimizeSize
      },
      {
        name: 'chroma_downsampling',
        type: this.binding.TF_ATTR_BOOL,
        value: chromaDownsampling
      },
      {
        name: 'density_unit',
        type: this.binding.TF_ATTR_STRING,
        value: densityUnit
      },
      {name: 'x_density', type: this.binding.TF_ATTR_INT, value: xDensity},
      {name: 'y_density', type: this.binding.TF_ATTR_INT, value: yDensity}, {
        name: 'xmp_metadata',
        type: this.binding.TF_ATTR_STRING,
        value: xmpMetadata
      }
    ];
    return this.executeEncodeImageOp(
        'EncodeJpeg', opAttrs, imageData, imageShape);
  }

  encodePng(imageData: Uint8Array, imageShape: number[], compression: number):
      Tensor {
    const opAttrs = [
      {name: 'compression', type: this.binding.TF_ATTR_INT, value: compression}
    ];
    return this.executeEncodeImageOp(
        'EncodePng', opAttrs, imageData, imageShape);
  }

  deleteSavedModel(id: number): void {
    this.binding.deleteSavedModel(id);
  }

  loadSavedModelMetaGraph(path: string, tags: string): number {
    return this.binding.loadSavedModel(path, tags);
  }

  private getMappedInputTensorIds(
      inputs: Tensor[], inputTensorInfos: ModelTensorInfo[]) {
    const tensorIds = this.getInputTensorIds(inputs);
    for (let i = 0; i < inputs.length; i++) {
      if (inputTensorInfos[i] != null) {
        if (inputTensorInfos[i].tfDtype === 'DT_UINT8') {
          const data = Uint8Array.from(inputs[i].dataSync());
          const inputTensorId = this.binding.createTensor(
              inputs[i].shape, this.binding.TF_UINT8, data);
          tensorIds[i] = inputTensorId;
        } else if (inputTensorInfos[i].tfDtype === 'DT_INT64') {
          const data =
              encodeInt32ArrayAsInt64(inputs[i].dataSync() as Int32Array);
          const inputTensorId = this.binding.createTensor(
              inputs[i].shape, this.binding.TF_INT64, data);
          tensorIds[i] = inputTensorId;
        }
      }
    }
    return tensorIds;
  }

  runSavedModel(
      id: number, inputs: Tensor[], inputTensorInfos: ModelTensorInfo[],
      outputOpNames: string[]): Tensor[] {
    const outputMetadata = this.binding.runSavedModel(
        id, this.getMappedInputTensorIds(inputs, inputTensorInfos),
        inputTensorInfos.map(info => info.name).join(','),
        outputOpNames.join(','));
    return outputMetadata.map(m => this.createOutputTensor(m));
  }

  // ------------------------------------------------------------
  // TensorBoard-related (tfjs-node-specific) backend kernels.

  summaryWriter(logdir: string): Tensor1D {
    const opAttrs = [
      {
        name: 'shared_name',
        type: this.binding.TF_ATTR_STRING,
        value: `logdir:${logdir}`
      },
      {name: 'container', type: this.binding.TF_ATTR_STRING, value: ''}
    ];
    const writerResource =
        this.executeSingleOutput('SummaryWriter', opAttrs, []);
    return writerResource as Tensor1D;
  }

  createSummaryFileWriter(
      resourceHandle: Tensor, logdir: string, maxQueue?: number,
      flushMillis?: number, filenameSuffix?: string): void {
    const inputArgs = [
      resourceHandle, scalar(logdir),
      scalar(maxQueue == null ? 10 : maxQueue, 'int32'),
      scalar(flushMillis == null ? 2 * 60 * 1000 : flushMillis, 'int32'),
      scalar(filenameSuffix == null ? '.v2' : filenameSuffix)
    ];
    this.executeMultipleOutputs('CreateSummaryFileWriter', [], inputArgs, 0);
  }

  writeScalarSummary(
      resourceHandle: Tensor, step: number, name: string,
      value: Scalar|number): void {
    tidy(() => {
      util.assert(
          Number.isInteger(step),
          () => `step is expected to be an integer, but is instead ${step}`);
      const inputArgs: Array<Tensor|Int64Scalar> =
          [resourceHandle, new Int64Scalar(step), scalar(name, 'string')];

      let typeAttr: number;
      if (typeof value === 'number') {
        inputArgs.push(scalar(value));
        typeAttr = this.binding.TF_FLOAT;
      } else {
        // `value` is a Scalar.
        util.assert(
            value.rank === 0,
            () => `A non-scalar tensor (rank ${value.rank}) is passed to ` +
                `writeScalarSummary()`);
        inputArgs.push(value);
        typeAttr = this.typeAttributeFromTensor(value);
      }
      const opAttrs: TFEOpAttr[] =
          [{name: 'T', type: this.binding.TF_ATTR_TYPE, value: typeAttr}];

      this.binding.executeOp(
          'WriteScalarSummary', opAttrs, this.getInputTensorIds(inputArgs), 0);
    });
  }

  flushSummaryWriter(resourceHandle: Tensor): void {
    const inputArgs: Tensor[] = [resourceHandle];
    this.executeMultipleOutputs('FlushSummaryWriter', [], inputArgs, 0);
  }

  // ~ TensorBoard-related (tfjs-node-specific) backend kernels.
  // ------------------------------------------------------------

  memory() {
    // Due to automatic garbage collection, the numbers are unreliable.
    // TODO(kreeger): Since there is finalization in C, count the true
    // number of undisposed tensors.
    return {unreliable: true};
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = process.hrtime();
    f();
    // hrtime() returns tuple of [seconds, nanoseconds], and we need to return
    // milliseconds.
    const elapsed = process.hrtime(start);
    return {kernelMs: elapsed[0] * 1000 + elapsed[1] / 1000000};
  }

  getNumOfSavedModels() {
    return this.binding.getNumOfSavedModels();
  }
}

/** Returns an instance of the Node.js backend. */
export function nodeBackend(): NodeJSKernelBackend {
  return tf.findBackend('tensorflow') as NodeJSKernelBackend;
}

/** Returns the TF dtype for a given DataType. */
export function getTFDType(dataType: tf.DataType): number {
  const binding = nodeBackend().binding;
  switch (dataType) {
    case 'float32':
      return binding.TF_FLOAT;
    case 'int32':
      return binding.TF_INT32;
    case 'bool':
      return binding.TF_BOOL;
    case 'complex64':
      return binding.TF_COMPLEX64;
    case 'string':
      return binding.TF_STRING;
    // tslint:disable-next-line:no-any
    case 'int64' as any:
      // int64 is not a generally supported dtype in TensorFlow.js
      // (tfjs-core). However, it needs to be included here for the purpose of
      // writing the `step` value to TensorBoard via WriteScalarSummary and
      // other op kernels.
      return binding.TF_INT64;
    default:
      const errorMessage = `Unknown dtype: ${dataType}`;
      throw new Error(errorMessage);
  }
}

/**
 * Creates a TFEOpAttr for a 'type' OpDef attribute from a Tensor or list of
 * Tensors.
 */
export function createTensorsTypeOpAttr(
    attrName: string,
    tensorsOrDtype: tf.Tensor|tf.Tensor[]|tf.DataType): TFEOpAttr {
  if (isNullOrUndefined(tensorsOrDtype)) {
    throw new Error('Invalid input tensors value.');
  }
  return {
    name: attrName,
    type: nodeBackend().binding.TF_ATTR_TYPE,
    value:
        (tensorsOrDtype instanceof tf.Tensor || Array.isArray(tensorsOrDtype)) ?
        getTFDTypeForInputs(tensorsOrDtype) :
        getTFDType(tensorsOrDtype)
  };
}

// TODO(yassogba) remove? who uses this?
export function createOpAttr(
    attrName: string, tensorsOrDtype: tf.Tensor|tf.Tensor[]|tf.DataType,
    value: ScalarLike): TFEOpAttr {
  if (isNullOrUndefined(tensorsOrDtype)) {
    throw new Error('Invalid input tensors value.');
  }
  return {name: attrName, type: nodeBackend().binding.TF_BOOL, value};
}

/** Returns the dtype number for a single or list of input Tensors. */
function getTFDTypeForInputs(tensors: tf.Tensor|tf.Tensor[]): number {
  if (isNullOrUndefined(tensors)) {
    throw new Error('Invalid input tensors value.');
  }
  if (isArray(tensors)) {
    for (let i = 0; i < tensors.length; i++) {
      return getTFDType(tensors[i].dtype);
    }
    return -1;
  } else {
    return getTFDType(tensors.dtype);
  }
}

export function ensureTensorflowBackend() {
  tf.util.assert(
      tf.getBackend() === 'tensorflow',
      () => `Expect the current backend to be "tensorflow", but got "${
          tf.getBackend()}"`);
}
