
declare class VectorClass<T> {
  size(): number;
  get(i: number): T|undefined;
  push_back(value: T): void;
  delete(): void;
}

export declare class VectorIntClass extends VectorClass<number> {}
export declare class VectorStringClass extends VectorClass<string> {}

export type Vector<T> = VectorClass<T>;
export type VectorInt = VectorIntClass;
export type VectorString = VectorStringClass;

export declare interface EncodeStringResult {
  readonly valuesFlat: VectorInt;
  readonly splitsFlat: VectorInt;
}

export interface SentencePieceModule extends EmscriptenModule {
  RegisterModel: (modelBase64: string) => string;
  EncodeString:
      (modelKey: string, inputs: VectorString, addBos: boolean, addEos: boolean,
       reverse: boolean) => EncodeStringResult;
  DecodeString:
      (modelKey: string, valuesFlat: VectorInt, splitsFlat: VectorInt,
       addBos: boolean, addEos: boolean, reverse: boolean) => VectorString;
  VectorInt: typeof VectorIntClass;
  VectorString: typeof VectorStringClass;
}

export interface WasmFactoryConfig {
  mainScriptUrlOrBlob?: string|Blob;
  locateFile?(path: string, prefix: string): string;
  instantiateWasm?: Function;
  onRuntimeInitialized?: () => void;
  onAbort?: (msg: string) => void;
}

declare var sentencePieceModuleFactory: (settings: WasmFactoryConfig) =>
    Promise<SentencePieceModule>;
export default sentencePieceModuleFactory;
