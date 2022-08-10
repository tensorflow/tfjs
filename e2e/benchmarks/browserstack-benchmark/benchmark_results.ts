
class BenchmarkRecord {
  status: string;
  value: BenchmarkRecordValue;

  getTableName(benchmarkReocrd) {
    let benchmarkCategory =
        this.value.modelInfo.model !== 'codeSnippet' ? 'Model' : 'Code Snippet';
    return `${benchmarkCategory} (${
        benchmarkReocrd ?.value ?.modelInfo ?.backend})`;
  }

  getDeviceName() {
    let deviceInfo = this.value.deviceInfo || {};
    return `${deviceInfo.os}(${deviceInfo.os_version})  ${
        deviceInfo.device || deviceInfo.browser + deviceInfo.browser_version} `;
  }

  getBenchmarkTargetName(benchmarkReocrd) {
    let benchmarkTargetName = this.value.modelInfo.model
    if (benchmarkTargetName === 'codeSnippet') {
      benchmarkTargetName = this.value.modelInfo.codeSnippet || 'NA';
    }
    return `${benchmarkTargetName.replace('\n', '').replace('\t', '')}`;
  }
}

class BenchmarkRecordValue {
  timeInfo: TimeInfo;
  memoryInfo: MemoryInfo;
  deviceInfo: DeviceInfo;
  modelInfo: ModelInfo;
}

class TimeInfo {
  times: number[];
  averageTime: number;
  minTime: number;
  maxTime: number;
}

class MemoryInfo {
  newBytes: number;
  newTensors: number;
  peakBytes: number;
  kernels: any[];
  kernelNames: string[];
  aggregatedKernels: any[];
}

class DeviceInfo {
  browser: string;
  browser_version: string;
  os: string;
  os_version: string;
  device?: string;
}

class ModelInfo {
  model: string;
  numRuns: number;
  backend: string;
  codeSnippet?: string;
}
