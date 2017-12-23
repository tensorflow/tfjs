import {Array1D, ENV, Scalar} from 'deeplearn';

async function runExample() {
  const math = ENV.math;
  const a = Array1D.new([1, 2, 3]);
  const b = Scalar.new(2);

  const result = math.add(a, b);

  // Option 1: With async/await.
  // Caveat: in non-Chrome browsers you need to put this in an async function.
  console.log(await result.data());  // Float32Array([3, 4, 5])

  // Option 2: With a Promise.
  result.data().then(data => console.log(data));

  // Option 3: Synchronous download of data.
  // This is simpler, but blocks the UI until the GPU is done.
  console.log(result.dataSync());
}

runExample();
