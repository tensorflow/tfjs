
import * as dl from 'deeplearn';

export async function warmupAndBenchmarkGPU(benchmark: () => dl.Tensor):
    Promise<number> {
  // Warmup.
  const out = benchmark();
  await out.data();
  out.dispose();
  return (await dl.time(benchmark)).kernelMs;
}
