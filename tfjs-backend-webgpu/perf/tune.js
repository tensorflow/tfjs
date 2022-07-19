window.addEventListener("DOMContentLoaded", initPage);
// append row to the HTML table
function appendRow(result) {
  let tbl = document.getElementById('my-table'), // table reference
    rowNum = tbl.rows.length - 1,
    row = tbl.insertRow(tbl.rows.length);      // append table row
  // insert table cells to the new row
  // 'Kernel name', 'Input', 'WebGPU', 'WebGL', 'WebGLComp', 'Program', 'Scale',
  createCell(row.insertCell(0), result.name, result.name, `${result.name}-${rowNum}`);
  createCell(row.insertCell(1), result.input, result.input, `input-${rowNum}`);
  createCell(row.insertCell(2), result.webgpu, 'webgpu', `webgpu-${rowNum}`);
  createCell(row.insertCell(3), result.webgl, 'webgl', `webgl-${rowNum}`);
  createCell(row.insertCell(4), result.webglComp, 'webglComp', `webglComp-${rowNum}`);
  createCell(row.insertCell(5), result.program, 'program', `program-${rowNum}`);
  createCell(row.insertCell(6), result.scale, `scale-${result.scale}`, `scale-${rowNum}`);
  createCell(row.insertCell(7), 'Rerun', 'rerun', `rerun-${rowNum}`, rerun);
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
  let mybody = document.getElementsByTagName("body")[0];
  mybody.innerHTML = '';
  myMessage = document.createElement("p");
  myMessage.id = 'message';
  mybody.appendChild(myMessage);

  // creates INFO labels
  for (let info of INFO) {
    let labelDiv = document.createElement('div');
    let label = document.createElement('label');
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
  mytable = document.createElement("table");
  mytable.id = 'my-table';
  mytablebody = document.createElement("tbody");

  // creates a <tr> element
  mycurrent_row = document.createElement("tr");
  mycurrent_row.style = 'background-color:#BDB76B;color:#ffffff;';
  // creating all cells
  ['Kernel name', 'Input', 'WebGPU', 'WebGL', 'WebGLComp %', 'Program', 'Scale(m*k*n)'].forEach((i) => {
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

async function timeMatmul(rowNum) {
  await tf.setBackend('webgpu');
  let tensors = [];
  let tensorsWarmUp = [];
  let tensorsToDispose = [];
  let inputs = [];
  const numWarmUp = 500;

  // Prepare data
  const warmA = tf.tensor2d(
    Array.from({ length: 12544 * 16 }, () => Math.floor(Math.random())),
    [12544, 16]
  );
  const warmB = tf.tensor2d(
    Array.from({ length: 16 * 96 }, () => Math.floor(Math.random())),
    [16, 96]
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

  const profile_webgpu = await tf.profile(() => {
    // Warmup model
    for (let j = 0; j < 50; j++) {
      for (let i = 0; i < inputs.length; i++) {
        document.getElementById('message').innerHTML =
          `Testing on webgpu ...`;
        let result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
        result.dispose();
      }
    }
    // Warmup gpu, keep gpu frequency at a high level
    document.getElementById('message').innerHTML =
      `Warming up testing on webgpu ...`;
    //for (let i = 0; i < numWarmUp; i++) {
    //  let m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
    //  m.dispose();
    //}
    // Collect result from here
    for (let i = 0; i < inputs.length; i++) {
      document.getElementById('message').innerHTML =
        `Testing on webgpu ...`;
      for (let i = 0; i < numWarmUp; i++) {
        let m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
        m.dispose();
      }
      let result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
      result.dispose();
    }
  });
  const webgpu_kernels = profile_webgpu.kernels;

  await tf.setBackend('webgl');
  const profile_webgl = await tf.profile(() => {
    // Warmup model
    for (let j = 0; j < 50; j++) {
      for (let i = 0; i < inputs.length; i++) {
        document.getElementById('message').innerHTML =
          `Testing on webgl ...`;
        let result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
        result.dispose();
      }
    }
    // Warmup gpu, keep gpu frequency at a high level
    document.getElementById('message').innerHTML =
      `Warming up testing on webgl ...`;
    //for (let i = 0; i < numWarmUp; i++) {
    //  let m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
    //  m.dispose();
    //}
    // Collect result from here
    for (let i = 0; i < inputs.length; i++) {
      document.getElementById('message').innerHTML =
        `Testing on webgl ...`;
      for (let i = 0; i < numWarmUp; i++) {
        let m = tf.matMul(tensorsWarmUp.tensorA, tensorsWarmUp.tensorB);
        m.dispose();
      }
      let result = tf.matMul(tensors[i].tensorA, tensors[i].tensorB);
      result.dispose();
    }
  });
  const webgl_kernels = profile_webgl.kernels;

  //for (let i = numWarmUp + inputs.length * 50; i < profile_webgpu.kernels.length; i++) {
  for (let i = numWarmUp + inputs.length * 50; i < profile_webgpu.kernels.length; i = i + 1 + numWarmUp) {
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

    const result = {};
    result.name = webgpu_kernels[i].name;
    result.input = `${inputInfo}`;
    result.webgpu = `${parseFloat(webgpu_kernels[i].kernelTimeMs).toFixed(2)}`; // WebGPU
    result.webgl = `${parseFloat(webgl_kernels[i].kernelTimeMs).toFixed(2)}`; // WebGL
    result.webglComp = `${parseFloat((result.webgl / result.webgpu) * 100).toFixed(0)}`; // WebGLComp
    result.scale = `${parseInt(mkn)}`;  // Scale
    result.program = `${webgpu_kernels[i].extraInfo.split(':')[0]}`;  // Program

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

  for (let tensor of tensorsToDispose) {
    tensor.dispose();
  }
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
  const iterationSelected = [];

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
  await timeMatmul();
  document.getElementById('message').innerHTML = 'Done!';
  updateColor();
  document.getElementById('run').disabled = false;
}

async function rerun(rowNum) {
  await timeMatmul(rowNum);
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

let webglCompResults = [];

let SCALES = [1016064, 5013504, 10838016, 33554432];
let currentScale = [];
let defaultInputs = [
  '12544, 8, 24',
  '12544, 16, 96',
  '49, 320, 1280',
  '49, 960, 320',
  '3136, 24, 144',
  '12544, 32, 16',
  '3136, 144, 24',
  '49, 960, 160',
];

let INPUTS = [];
const INFO = [
  '0. Run under flag: --enable-unsafe-webgpu --disable-dawn-features=disallow_unsafe_apis',
  '1. Sortable by clicking table column title',
  '2. Rerunable for every single line',
  '3. Set checkBox below to define new test suite',
];
