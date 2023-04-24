export interface SentencePieceModule extends EmscriptenModule {
  NormalizeString: any;
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
