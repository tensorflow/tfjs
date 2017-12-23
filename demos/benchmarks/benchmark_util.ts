
import {NDArray, NDArrayMath} from 'deeplearn';

export async function warmupAndBenchmarkGPU(
    math: NDArrayMath, benchmark: () => NDArray): Promise<number> {
  // Warmup.
  const out = benchmark();
  await out.data();
  out.dispose();
  // Real timing.
  return math.time(benchmark);
}
