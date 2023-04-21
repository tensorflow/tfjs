export interface BackendWasmModule extends EmscriptenModule {
  NormalizeString: any;
}

export interface WasmFactoryConfig {
  mainScriptUrlOrBlob?: string|Blob;
  locateFile?(path: string, prefix: string): string;
  instantiateWasm?: Function;
  onRuntimeInitialized?: () => void;
  onAbort?: (msg: string) => void;
}

declare var moduleFactory: (settings: WasmFactoryConfig) =>
    Promise<BackendWasmModule>;
export {moduleFactory};
