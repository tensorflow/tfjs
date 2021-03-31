declare interface TFWeb {
  tfweb: {setWasmPath(path: string): void;};
  TFWebModelRunner: typeof TFWebModelRunner;
}

export declare class TFWebModelRunner {
  static create(modelPath: string, options: TFWebModelRunnerOptions):
      Promise<TFWebModelRunner>;
  getInputs(): TFWebModelRunnerTensorInfo[]|undefined;
  getOutputs(): TFWebModelRunnerTensorInfo[]|undefined;
  infer(): boolean|undefined;
  cleanUp(): void;
}

export declare interface TFWebModelRunnerOptions {
  numThreads: number;
}

export declare interface TFWebModelRunnerTensorInfo {
  id: number;
  dataType: 'int8'|'uint8'|'bool'|'int16'|'int32'|'uint32'|'float32'|'float64';
  name: string;
  shape: string;
  data(): Int8Array|Uint8Array|Int16Array|Int32Array|Uint32Array|Float32Array
      |Float64Array;
}

export declare let tfweb: TFWeb;

export as namespace TFWebClient;
