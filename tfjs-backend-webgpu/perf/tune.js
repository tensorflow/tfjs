window.addEventListener("DOMContentLoaded", initPage);
// append row to the HTML table
function appendRow(result) {
  let tbl = document.getElementById('my-table'), // table reference
    rowNum = tbl.rows.length - 1,
    row = tbl.insertRow(tbl.rows.length);      // append table row
  // insert table cells to the new row
  // 'Kernel name', 'Input', 'WebGPU', 'WebGL', 'WebGLComp', 'WebGPUProgram', 'WebGLProgram', 'Scale',
  createCell(row.insertCell(0), result.name, result.name, `${result.name}-${rowNum}`);
  createCell(row.insertCell(1), result.input, result.input, `input-${rowNum}`);
  createCell(row.insertCell(2), result.webgpu, 'webgpu', `webgpu-${rowNum}`);
  createCell(row.insertCell(3), result.webgl, 'webgl', `webgl-${rowNum}`);
  createCell(row.insertCell(4), result.webglComp, 'webglComp', `webglComp-${rowNum}`);
  createCell(row.insertCell(5), result.webgpuProgram, 'webgpuProgram', `webgpuProgram-${rowNum}`);
  createCell(row.insertCell(6), result.webglProgram, 'webglProgram', `webglProgram-${rowNum}`);
  createCell(row.insertCell(7), result.scale, `scale-${result.scale}`, `scale-${rowNum}`);
  createCell(row.insertCell(8), 'Rerun', 'rerun', `rerun-${rowNum}`, rerun);
}

// Update result table by rerun
function updateRow(result, rowNum) {
  document.getElementById(`webgpu-${rowNum}`).innerHTML = result.webgpu;
  document.getElementById(`webgl-${rowNum}`).innerHTML = result.webgl;
  document.getElementById(`webglComp-${rowNum}`).innerHTML = result.webglComp;
}

// create DIV element and append to the table cell
function createCell(cell, text, classes, id, onclick) {
  let div = document.createElement('div'), // create DIV element
    txt = document.createTextNode(text); // create text node
  div.appendChild(txt);                    // append text node to the DIV
  div.setAttribute('class', classes);        // set DIV class attribute
  div.setAttribute('id', id);
  div.setAttribute('value', text);
  if (onclick) {
    div.setAttribute('type', 'button');
    let rowNum = id.split('-')[1];
    div.onclick = function () { onclick(rowNum) };
  }
  cell.appendChild(div);                   // append DIV to the table cell
}

function initPage() {
  // get the reference for the body
  const mybody = document.getElementsByTagName("body")[0];
  mybody.innerHTML = '';
  const myMessage = document.createElement("p");
  myMessage.id = 'message';
  mybody.appendChild(myMessage);

  // creates INFO labels
  for (let info of INFO) {
    const labelDiv = document.createElement('div');
    const label = document.createElement('label');
    label.innerHTML = `${info}`;
    labelDiv.appendChild(label);
    mybody.appendChild(labelDiv);
  }

  // creates Scales labels and checkbox
  for (let item of [SCALES]) {
    let backendDiv = document.createElement('div');
    let labelClass = document.createElement("label");
    labelClass.innerHTML = 'Scales';
    backendDiv.appendChild(labelClass);
    for (let i of item) {
      let checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      if (currentScale.includes(i)) {
        checkbox.checked = true;
      }
      checkbox.name = `scale-${i}`;
      checkbox.value = i;
      checkbox.className = `scaleCheckBox`;
      checkbox.addEventListener('change', (event) => hideOrPresent(event));
      let label = document.createElement("label");
      label.innerHTML = i;
      backendDiv.appendChild(checkbox);
      backendDiv.appendChild(label);
    }
    mybody.appendChild(backendDiv);
  }

  // creates run button
  let btn = document.createElement("button");
  btn.id = 'run';
  btn.innerHTML = "Run";
  btn.style.background = 'orange';
  btn.onclick = function () {
    run();
  };
  document.body.appendChild(btn);

  // creates <table> and <tbody> elements
  const mytable = document.createElement("table");
  mytable.id = 'my-table';
  mytablebody = document.createElement("tbody");

  // creates a <tr> element
  mycurrent_row = document.createElement("tr");
  mycurrent_row.style = 'background-color:#BDB76B;color:#ffffff;';
  // creating all cells
  ['Kernel name', 'Input', 'WebGPU', 'WebGL', 'WebGLComp %', 'WebGPUProgram', 'WebGLProgram', 'Scale(m*k*n)'].forEach((i) => {
    // creates a <td> element
    mycurrent_cell = document.createElement("th");
    // creates a Text Node
    currenttext = document.createTextNode(i);
    // appends the Text Node we created into the cell <td>
    mycurrent_cell.appendChild(currenttext);
    // appends the cell <td> into the row <tr>
    mycurrent_row.appendChild(mycurrent_cell);
  });
  // appends the row <tr> into <tbody>
  mytablebody.appendChild(mycurrent_row);

  // appends <tbody> into <table>
  mytable.appendChild(mytablebody);
  // appends <table> into <body>
  mybody.appendChild(mytable);
  // sets the border attribute of mytable to 2;
  mytable.setAttribute("border", "2");

  // Add sorting for tr
  document.querySelectorAll('th').forEach(th => th.addEventListener('click', (() => {
    const table = th.closest('table');
    Array.from(table.querySelectorAll('tr:nth-child(n+2)'))
      .sort(comparer(Array.from(th.parentNode.children).indexOf(th), this.asc = !this.asc))
      .forEach(tr => table.appendChild(tr));
  })));

  run();
}

async function timeMatmul(backend, rowNum) {
  await tf.setBackend(backend);
  let tensors = [];
  let tensorsWarmUp = [];
  let tensorsToDispose = [];
  let inputs = [];

  // Prepare data
  const warmA = tf.tensor2d(
    Array.from({ length: warmUpSize * warmUpSize }, () => Math.floor(Math.random())),
    [warmUpSize, warmUpSize]
  );
  const warmB = tf.tensor2d(
    Array.from({ length: warmUpSize * warmUpSize }, () => Math.floor(Math.random())),
    [warmUpSize, warmUpSize]
  );
  tensorsWarmUp = { tensorA: warmA, tensorB: warmB };
  tensorsToDispose.push(warmA);
  tensorsToDispose.push(warmB);
  if (rowNum !== undefined) {
    let input = document.getElementById(`input-${rowNum}`).getAttribute('value');
    let m = input.split('[')[1].split(',')[0];
    let k = input.split('[')[2].split(']')[0].split(',')[0];
    let n = input.split('[')[2].split(']')[0].split(',')[1];
    inputs = [`${m},${k},${n}`];
  }
  else {
    inputs = INPUTS;
  }
  for (let i = 0; i < inputs.length; i++) {
    let dimAOuter = parseInt(inputs[i].split(',')[0]);
    let dimInner = parseInt(inputs[i].split(',')[1]);
    let dimBOuter = parseInt(inputs[i].split(',')[2]);
    const tensorA = tf.tensor2d(
      Array.from({ length: dimAOuter * dimInner }, () => Math.floor(Math.random())),
      [dimAOuter, dimInner]
    );
    const tensorB = tf.tensor2d(
      Array.from({ length: dimInner * dimBOuter }, () => Math.floor(Math.random())),
      [dimInner, dimBOuter]
    );
    tensors.push({ tensorA, tensorB });
    tensorsToDispose.push(tensorA);
    tensorsToDispose.push(tensorB);
  }

  tf.env().set('CHECK_COMPUTATION_FOR_ERRORS', false);
  // Warmup, first run of each matmul shapes
  for (let i = 0; i < inputs.length; i++) {
    let result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
    await result.data();
    result.dispose();
  }
  const profile_data = await tf.profile(() => {
    // Warmup gpu and keep gpu frequency at a high level
    document.getElementById('message').innerHTML =
      `Warming up testing on ${backend} ...`;
    for (let i = 0; i < numWarmUp; i++) {
      let m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
      m.dispose();
    }
    // Collect result from here
    for (let i = 0; i < inputs.length; i++) {
      document.getElementById('message').innerHTML =
        `Testing on ${backend} ...`;
      for (let j = 0; j < numAvg; j++) {
        const result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
        // Insert large shape between each test to keep gpu high frequency
        const m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
        result.dispose();
        m.dispose();
      }
    }
  });
  const profile_kernels = profile_data.kernels;

  for (let tensor of tensorsToDispose) {
    tensor.dispose();
  }
  return profile_kernels;
}

function drawTable(webgpu_kernels, webgl_kernels, rowNum) {
  for (let i = numWarmUp; i < webgpu_kernels.length; i += numAvg * 2) {
    let inputInfo;
    webgpu_kernels[i].inputShapes.forEach((inputShape, index) => {
      if (inputInfo == null) {
        inputInfo = '';
      } else {
        inputInfo += '\n';
      }
      if (inputShape == null) {
        inputInfo += `input${index}: null`;
      } else {
        inputInfo += `input${index}: ${inputShape.length}D[${inputShape}]`;
      }
    });

    let mkn = webgpu_kernels[i].inputShapes[0][0] * webgpu_kernels[i].inputShapes[0][1] * webgpu_kernels[i].inputShapes[1][1];

    const avgWebgpu = getAvgKernelTime(webgpu_kernels.slice(i, i + 2 * numAvg));
    const avgWebgl = getAvgKernelTime(webgl_kernels.slice(i, i + 2 * numAvg));

    const result = {};
    result.name = webgpu_kernels[i].name;
    result.input = `${inputInfo}`;
    result.webgpu = `${parseFloat(avgWebgpu).toFixed(2)}`; // WebGPU
    result.webgl = `${parseFloat(avgWebgl).toFixed(2)}`; // WebGL
    result.webglComp = `${parseFloat((result.webgl / result.webgpu) * 100).toFixed(0)}`; // WebGLComp
    result.scale = `${parseInt(mkn)}`;  // Scale
    result.webgpuProgram = `${webgpu_kernels[i].extraInfo.split(',').map(x => x.split(':')[0])}`;  // WebGPUProgram
    result.webglProgram = `${webgl_kernels[i].extraInfo.split(',').map(x => x.split(':')[0])}`;  // WebGLProgram

    if (rowNum !== undefined) {
      updateRow(result, rowNum);
      // update gpu result
      webglCompResults[rowNum] = result.webglComp;
    } else {
      appendRow(result);
      // store gpu result
      webglCompResults.push(result.webglComp);
    }
  }
}

function updateTable(avgWebgpu, avgWebgl, rowNum) {
  const webglComp = `${parseFloat((avgWebgl / avgWebgpu) * 100).toFixed(0)}`;
  const result = {};
  result.webgpu = `${parseFloat(avgWebgpu).toFixed(2)}`;
  result.webgl = `${parseFloat(avgWebgl).toFixed(2)}`;
  result.webglComp = webglComp;
  updateRow(result, rowNum);
  webglCompResults[rowNum] = webglComp;

}

function updateColor() {
  for (let i = 0; i <= webglCompResults.length - 1; i++) {
    const webglCompValue = webglCompResults[i];
    let r, g, b;
    if (webglCompValue < 100) {
      r = 255;
      g = Math.floor(255 - 255 * ((100 - webglCompValue)) / 100);
      b = Math.floor(255 - 255 * ((100 - webglCompValue)) / 100);
    } else {
      g = 255;
      r = Math.floor(255 - 255 * ((webglCompValue - 100)) / 100);
      b = Math.floor(255 - 255 * ((webglCompValue - 100)) / 100);
    }
    document.getElementById(`webglComp-${i}`).style = `background-color: rgb(${r}, ${g}, ${b})`;
  }
}

function hideOrPresent(event) {
  const value = event.target.name;
  const nodes = document.getElementsByClassName(value);
  for (let i = 0; i <= nodes.length - 1; i++) {
    if (event.target.checked) {
      nodes[i].parentNode.parentNode.style = ``;
    } else {
      nodes[i].parentNode.parentNode.style = `display:none`;
    }
  }
}

async function run() {
  document.getElementById('run').disabled = true;
  // remove results
  let tableHeaderRowCount = 1;
  let table = document.getElementById('my-table');
  let rowCount = table.rows.length;
  for (let i = rowCount; i > tableHeaderRowCount; i--) {
    table.deleteRow(i - 1);
  }
  const scaleSelected = [];

  // clear compared result
  webglCompResults = [];

  // define scale suite
  let scales = document.querySelectorAll('.scaleCheckBox:checked');
  for (let s of scales) {
    scaleSelected.push(s.value);
  }

  INPUTS = defaultInputs;
  for (let scale of scaleSelected) {
    let mySet = getTestSet(scale);
    INPUTS = INPUTS.concat([...mySet]);
  }
  const profile_webgl = await timeMatmul('webgl');
  const profile_webgpu = await timeMatmul('webgpu');
  drawTable(profile_webgpu, profile_webgl);
  document.getElementById('message').innerHTML = 'Done!';
  updateColor();
  document.getElementById('run').disabled = false;
}

async function rerun(rowNum) {
  const profile_webgl = await timeMatmul('webgl', rowNum);
  const profile_webgpu = await timeMatmul('webgpu', rowNum);
  const avgWebgl = getAvgKernelTime(profile_webgl.slice(-numAvg * 2));
  const avgWebgpu = getAvgKernelTime(profile_webgpu.slice(-numAvg * 2));
  updateTable(avgWebgpu, avgWebgl, rowNum);
  document.getElementById('message').innerHTML = 'Done!';
  updateColor();
}

const getCellValue = (tr, idx) => tr.children[idx].innerText || tr.children[idx].textContent;

const comparer = (idx, asc) => (a, b) => ((v1, v2) =>
  v1 !== '' && v2 !== '' && !isNaN(v1) && !isNaN(v2) ? v1 - v2 : v1.toString().localeCompare(v2)
)(getCellValue(asc ? a : b, idx), getCellValue(asc ? b : a, idx));

function getFactors(c) {
  const y = [];
  let min = 1;
  let max = c;
  while (min < max && min * min <= c) {
    if (c % min === 0) {
      y.push(min, c / min);
      max = c / min;
    }
    min++;
  }
  return y;
}

function getTestSet(num) {
  const factors = getFactors(num);
  const mySet = new Set();
  for (let i = 0; i < factors.length; i++) {
    for (let j = 0; j < factors.length; j++) {
      if (factors[i] * factors[j] < num && num % (factors[i] * factors[j]) === 0) {
        let m = factors[i];
        let n = factors[j];
        let k = num / factors[i] / factors[j];
        if (m > 15 && n > 15 && k > 15
          && n < 2048 && k < 2048
          && Math.sqrt(m) % 1 === 0) {
          mySet.add(`${m},${k},${n}`);
        }
      }
    }
  }
  return mySet;
}

function getPrimeFactor(num) {
  const prime_factor = [];
  for (let i = 2; i < num; i++) {
    if (num % i == 0) {
      prime_factor.push(i);
      num /= i;
      i -= 1;
    }
  }
  prime_factor.push(num);
  return prime_factor;
}

function getAvgKernelTime(kernels) {
  const avg = kernels.reduce(
    (a, b, index) => {
      if (index % 2 === 0) {
        return a + b.kernelTimeMs;
      }
      return a;
    },
    0,
  ) / numAvg;
  return avg;
}

let webglCompResults = [];

let SCALES = [1016064, 5013504, 10838016, 33554432];
let currentScale = [];
let defaultInputs = [
  '1, 1280, 1001',
  '12544, 16, 64',
  '196, 672, 112',
  '1, 960, 1280',
  '196, 112, 672',
  '196, 480, 112',
  '49, 960, 160',
  '784, 40, 240',
  '49, 672, 160',
  '3136, 24, 72',
  '3136, 72, 24',
  '49, 160, 960',
  '196, 80, 480',
  '784, 120, 40',
  '784, 40, 120',
  '3136, 64, 24',
  '784, 72, 40',
  '1, 240, 960',
  '196, 80, 200',
  '1, 960, 240',
  '196, 240, 80',
  '12544, 16, 16',
  '196, 200, 80',
  '196, 80, 184',
  '196, 184, 80',
  '1, 672, 168',
  '1, 168, 672',
  '1, 480, 120',
  '1, 32, 120',
  '1, 120, 480',
  '1, 120, 32',
  '1, 24, 72',
  '1, 72, 24',
];

let INPUTS = [];
const numWarmUp = 1000;
const numAvg = 20;
const warmUpSize = 512;
const INFO = [
  '0. Run under flag: --enable-unsafe-webgpu --disable-dawn-features=disallow_unsafe_apis',
  '1. Sortable by clicking table column title',
  '2. Rerunable for every single line',
  '3. Set checkBox below to define new test suite',
  '4. Default workloads are used for MobileNetV3',
];
