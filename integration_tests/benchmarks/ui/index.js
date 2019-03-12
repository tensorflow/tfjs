firebase.initializeApp({
  apiKey: 'AIzaSyBhlCUcKQYYTJs0AkH1z_49iSyDMrKAKx8',
  authDomain: 'jstensorflow.firebaseapp.com',
  databaseURL: 'https://tensorflowjs-benchmarks.firebaseio.com/',
  projectId: 'jstensorflow',
  storageBucket: 'jstensorflow.appspot.com',
  messagingSenderId: '433613381222'
});

const database = firebase.database();
const ref = database.ref();

const swatches =
    ['#2196f3', '#3f51b5', '#9c27b0', '#e91e63', '#ff5722', '#009688'];

ref.once('value', resp => {
  const val = resp.val();
  const keys = Object.keys(val).sort((a, b) => new Date(a) - new Date(b));
  const data = [];

  const state = {'activeTarget': 0, 'activeTest': 0};

  keys.forEach(date => {  // populate data
    const obj = val[date];

    Object.keys(obj).forEach(test => {
      const targets = Object.keys(obj[test]);
      targets.forEach((target, i) => {
        const {timestamp, runs, userAgent} = obj[test][target];
        const newEntry = {timestamp, params: []};

        Object.keys(runs).forEach(param => {
          newEntry.params.push({name: param, ms: runs[param].averageTimeMs});
        });

        let targetIndex = data.map(d => d.name).indexOf(target);
        if (targetIndex === -1) {
          data.push({name: target, userAgent: userAgent, tests: []});
          targetIndex = data.length - 1;
        }

        let testIndex = data[targetIndex].tests.map(d => d.name).indexOf(test);
        if (testIndex === -1) {
          data[targetIndex].tests.push({name: test, entries: []});
          testIndex = data[targetIndex].tests.length - 1;
        }

        data[targetIndex].tests[testIndex].entries.push(newEntry);
      });
    });
  });

  console.log(data);

  const chartHeight = 200;
  const chartWidth = document.querySelector('#container').offsetWidth;

  data.forEach((target, i) => {  // create DOM
    const name = target.name;
    const userAgent = target.userAgent;
    const tab = document.createElement('a');
    tab.setAttribute('href', '#' + name);
    tab.textContent = name;
    tab.classList.add('mdl-tabs__tab');

    const panel = document.createElement('div');
    panel.classList.add('mdl-tabs__panel');
    panel.id = `${name}-panel`;

    if (i === 0) {
      tab.classList.add('is-active');
      panel.classList.add('is-active');
    }

    panel.innerHTML += `<div class="user-agent-string">User agent: ${userAgent}</div>`;

    target.tests.forEach((test, i) => {
      const params = test.entries.reduce((acc, curr) => {
        curr.params.forEach(param => {
          if (typeof acc[param.name] === 'undefined') {
            acc[param.name] = [];
          }

          acc[param.name].push({ms: param.ms});
        });

        return acc;
      }, {});

      const msArray = test.entries.map(d => d.params.map(p => p.ms))
                          .reduce((acc, curr) => acc.concat(curr), []);
      const max = Math.max(...msArray);
      const min = Math.min(...msArray);

      const minWidthOfIncrement = 20;
      let increment = 1;
      while ((chartWidth / ((test.entries.length - 1) / increment)) <
             minWidthOfIncrement) {
        increment *= 2;
      }

      const xIncrement = chartWidth / (test.entries.length - 1);

      const template =  // template trendlines
          `<div class='test'>
        <h4 class='test-name'>${test.name}</h4>
        <div class='legend'>${
              Object.keys(params)
                  .map((param, i) => `<div class='swatch'>
            <div class='color' style='background-color: ${swatches[i]}'></div>
            <div class='label'>${param}</div>
          </div>`).join(' ')}</div>
        <div class='graph-container'>
          <div style='height:${chartHeight}px' class='y-axis-labels'>
            <div class='y-max'>${max.toFixed(2)}ms</div>
            <div class='y-min'>${min.toFixed(2)}ms</div>
          </div>
          <svg data-index=${i} class='graph' width='${chartWidth}' height='${
              chartHeight}'>${
              Object.keys(params).map(
                  (param, i) => `<path stroke='${swatches[i]}' d='M${
                      params[param]
                          .map(
                              (d, i) => `${i * xIncrement},${
                                  chartHeight *
                                  (1 - ((d.ms - min) / (max - min)))}`)
                          .join('L')}'></path>`)}</svg>
          <div class='x-axis-labels'>${
              test.entries
                  .map((d, i) => {
                    if (i % increment === 0) {
                      const date = new Date(d.timestamp);
                      return `<div class='x-label' style='left:${
                          (i / increment) *
                          (chartWidth /
                           ((test.entries.length - 1) / increment))}px'>${
                          date.getMonth() + 1} / ${date.getDate()}</div>`;
                    }
                    return '';
                  })
                  .join(' ')}</div>
          <div class='detail-panel'>
            <div class='line'></div>
            <div class='contents'></div>
          </div>
        </div>
      </div>`;

      panel.innerHTML += template;
    });

    document.querySelector('.mdl-tabs__tab-bar').appendChild(tab);
    document.querySelector('.mdl-tabs').appendChild(panel);

    let graphOffsetLeft = 0;

    function resize() {
      graphOffsetLeft = document.querySelector('.graph-container').offsetLeft;
    }

    window.addEventListener('resize', resize);
    resize();

    document.addEventListener('mousemove', e => {  // handle hovering
      if (e.target.classList.contains('graph')) {
        state.activeTest = +e.target.getAttribute('data-index');

        const entries =
            data[state.activeTarget].tests[state.activeTest].entries;

        const left = e.clientX - graphOffsetLeft;
        const entryIndex = Math.max(
            0,
            Math.min(
                entries.length - 1,
                Math.floor((left / chartWidth) * entries.length)));
        const parentNode = e.target.parentNode;
        parentNode.querySelector('.detail-panel').style.left = left + 'px';
        parentNode.querySelector('.detail-panel .contents').innerHTML = `${
            entries[entryIndex]
                .params
                .map(
                    (d, i) => `<div class='label-wrapper'>
                      <div class='color' style='background-color: ${
                        swatches[i]}'></div>
                      <div class='label'>${d.ms.toFixed(2)}</div>
                    </div>`)
                .join(' ')}
        `;
      }
    });
  });
});