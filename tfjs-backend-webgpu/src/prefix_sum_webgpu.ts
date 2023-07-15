/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ScanSingleBlockProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [128, 1, 1];
  uniforms = 'inputOffset: i32';
  nextPowOfTwo: number;

  constructor(shape: number, inputLength: number) {
    this.outputShape = [shape];
    this.nextPowOfTwo =
        Math.pow(2, Math.ceil(Math.log(inputLength) / Math.log(2)));
    this.workgroupSize[0] = this.nextPowOfTwo === 1 ? 1 : this.nextPowOfTwo / 2;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.shaderKey = `scanSingleBlock_${this.nextPowOfTwo}`;
  }

  getUserCode(): string {
    const userCode = `
    var<workgroup> temp : array<f32, ${this.nextPowOfTwo}>;
     ${main('index')} {
       if (2 * index + uniforms.inputOffset < uniforms.xShape) {
         temp[2 * index] = getX(2 * index + uniforms.inputOffset);
       }
       if (2 * index + 1 + uniforms.inputOffset < uniforms.xShape) {
         temp[2 * index + 1] = getX(2 * index + 1 + uniforms.inputOffset);
       }

       var offset = 1;
       for (var d = ${this.nextPowOfTwo} >> 1; d > 0; d >>= 1) {
         workgroupBarrier();

         if (index < d) {
           let ai = offset * (2 * index + 1) - 1;
           let bi = offset * (2 * index + 2) - 1;

           temp[bi] += temp[ai];
         }
         offset *= 2;
       }

       workgroupBarrier();
       if (index == 0) {
         setOutputAtIndex(uniforms.outShape - 1, temp[${
        this.nextPowOfTwo} - 1]);
         temp[${this.nextPowOfTwo} - 1] = 0.0;
       }

       for (var d = 1; d < ${this.nextPowOfTwo}; d *= 2) {
         offset >>= 1;
         workgroupBarrier();

         if (index < d) {
           let ai = offset * (2 * index + 1) - 1;
           let bi = offset * (2 * index + 2) - 1;

           let t = temp[ai];
           temp[ai] = temp[bi];
           temp[bi] += t;
         }
       }

       workgroupBarrier();
       if (2 * index < uniforms.outShape - 1) {
         setOutputAtIndex(2 * index, temp[2 * index]);
       }
       if (2 * index + 1 < uniforms.outShape - 1) {
        setOutputAtIndex(2 * index + 1, temp[2 * index + 1]);
      }
     }
   `;
    return userCode;
  }
}

export class ScanMultipleBlocksProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [128, 1, 1];

  constructor(shape: number) {
    this.outputShape = [shape];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.shaderKey = `scanMultipleBlocks`;
  }

  getUserCode(): string {
    const userCode = `
    var<workgroup> temp : array<f32, 256>;
     ${main('index')} {
       let threadId = i32(localId.x);
       let blockId = i32(workgroupId.x);
       let blockOffset = blockId * 256;

       temp[2 * threadId] = getX(blockOffset + 2 * threadId);
       temp[2 * threadId + 1] = getX(blockOffset + 2 * threadId + 1);

       var offset = 1;
       for (var d = 128; d > 0; d >>= 1) {
         workgroupBarrier();

         if (threadId < d) {
           let ai = offset * (2 * threadId + 1) - 1;
           let bi = offset * (2 * threadId + 2) - 1;

           temp[bi] += temp[ai];
         }
         offset *= 2;
       }

       workgroupBarrier();
       if (threadId == 0) {
         temp[255] = 0.0;
       }

       for (var d = 1; d < 256; d *= 2) {
         offset >>= 1;
         workgroupBarrier();

         if (threadId < d) {
           let ai = offset * (2 * threadId + 1) - 1;
           let bi = offset * (2 * threadId + 2) - 1;

           let t = temp[ai];
           temp[ai] = temp[bi];
           temp[bi] += t;
         }
       }

       workgroupBarrier();
       setOutputAtIndex(blockOffset + 2 * threadId, temp[2 * threadId]);
       setOutputAtIndex(blockOffset + 2 * threadId + 1, temp[2 * threadId + 1]);
     }
   `;
    return userCode;
  }
}

export class ScanEachBlockSumProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [128, 1, 1];
  size = true;
  nextPowOfTwo: number;

  constructor(shape: number) {
    this.outputShape = [shape];
    this.dispatchLayout = flatDispatchLayout([shape * this.workgroupSize[0]]);
    this.dispatch = computeDispatch(
        this.dispatchLayout, [shape * this.workgroupSize[0]],
        this.workgroupSize);

    this.shaderKey = `scanEachBlockSum`;
  }

  getUserCode(): string {
    const userCode = `
    var<workgroup> temp : array<f32, 256>;
     ${main('index')} {
       let threadId = i32(localId.x);
       let blockId = i32(workgroupId.x);
       let blockOffset = blockId * 256;

       temp[2 * threadId] = getX(blockOffset + 2 * threadId);
       temp[2 * threadId + 1] = getX(blockOffset + 2 * threadId + 1);

       var offset = 1;
       for (var d = 128; d > 0; d >>= 1) {
         workgroupBarrier();

         if (threadId < d) {
           let ai = offset * (2 * threadId + 1) - 1;
           let bi = offset * (2 * threadId + 2) - 1;

           temp[bi] += temp[ai];
         }
         offset *= 2;
       }

       workgroupBarrier();
       if (threadId == 0) {
         setOutputAtIndex(blockId, temp[255]);
       }
     }
   `;
    return userCode;
  }
}

export class MergeBlocksSumProgram implements WebGPUProgram {
  variableNames = ['x', 'blocks'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [256, 1, 1];
  nextPowOfTwo: number;

  constructor(shape: number, blocks: number) {
    this.outputShape = [blocks * this.workgroupSize[0] + 1];
    this.dispatchLayout = flatDispatchLayout([blocks * this.workgroupSize[0]]);
    this.dispatch = computeDispatch(
        this.dispatchLayout, [blocks * this.workgroupSize[0]],
        this.workgroupSize);

    this.shaderKey = `mergeBlocksSum`;
  }

  getUserCode(): string {
    const userCode = `
     ${main('index')} {
       if (index < uniforms.outShape - 1) {
         let threadId = i32(localId.x);
         let blockId = i32(workgroupId.x);
         let blockOffset = blockId * 128;

         if (index == 0) {
           setOutputAtIndex(uniforms.outShape - 1, getBlocks(uniforms.blocksShape - 1));
         }

         setOutputAtIndex(index, getX(index) + getBlocks(blockId));
       }
     }
   `;
    return userCode;
  }
}

export class MergeBlocksRemainderSumProgram implements WebGPUProgram {
  variableNames = ['x', 'remainder'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [256, 1, 1];
  size = true;
  nextPowOfTwo: number;

  constructor(shape: number, blocks: number) {
    this.outputShape = [shape];
    this.dispatchLayout =
        flatDispatchLayout([(blocks + 1) * this.workgroupSize[0]]);
    this.dispatch = computeDispatch(
        this.dispatchLayout, [(blocks + 1) * this.workgroupSize[0]],
        this.workgroupSize);

    this.shaderKey = `mergeBlocksRemainderSum`;
  }

  getUserCode(): string {
    const userCode = `
     ${main('index')} {
       if (index < uniforms.outShape) {
         if (index < uniforms.xShape - 1) {
           setOutputAtIndex(index, getX(index));
         } else {
          setOutputAtIndex(index, getRemainder(index - uniforms.xShape + 1) +
              getX(uniforms.xShape - 1));
         }
       }
     }
   `;
    return userCode;
  }
}
