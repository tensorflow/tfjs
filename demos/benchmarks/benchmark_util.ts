
import {ENV, NDArray, NDArrayMathGPU} from 'deeplearn';

export async function warmupAndBenchmarkGPU(
    math: NDArrayMathGPU, benchmark: () => NDArray): Promise<number> {
  const gpgpu = math.getGPGPUContext();

  let out: NDArray;
  const saveOutputBenchmark = () => {
    out = benchmark();
  };

  // Warmup.
  if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')) {
    await gpgpu.runQuery(saveOutputBenchmark);
  } else {
    saveOutputBenchmark();
    out.dataSync();
  }
  out.dispose();

  let totalTime: number;
  if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
    totalTime = await gpgpu.runQuery(saveOutputBenchmark);
  } else {
    const start = performance.now();

    saveOutputBenchmark();
    out.dataSync();

    totalTime = performance.now() - start;
  }
  out.dispose();
  return totalTime;
}
