
const socket = io();
const state = {
  run: () => {
    // Disable the button.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = .5;

    socket.emit('run', true);
  }
};

socket.on('benchmarkComplete', benchmarkResult => {
  const {timeInfo, memoryInfo} = benchmarkResult;
  document.getElementById('results').innerHTML +=
      JSON.stringify(timeInfo, null, 2);

  // Enable the button.
  benchmarkButton.__li.style.pointerEvents = '';
  benchmarkButton.__li.style.opacity = 1;
});

const gui = new dat.gui.GUI();
const benchmarkButton = gui.add(state, 'run').name('Run benchmark');
