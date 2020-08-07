
import {backend_util, util} from '@tensorflow/tfjs-core';
const {getBroadcastDims} = backend_util;
import {getGlslDifferences} from '../glsl_version';

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number],
  isUniform: boolean,
  isPacked: boolean,
  packCol?: boolean, flatOffset: number
};

export type InputInfo = {
  name: string,
  shapeInfo: ShapeInfo
};

function getPackedSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getPackedSamplerScalar(inInfo);
    case 1:
      return getPackedSampler1D(inInfo);
    case 2:
      return getPackedSampler2D(inInfo);
    case 3:
      return getPackedSampler3D(inInfo);
    default:
      return getPackedSamplerND(inInfo);
  }
}

export function getColPackedInputSamplingSnippet(
    inInfo: InputInfo,
    outShapeInfo: ShapeInfo,
    ): string {
  let res = '';
  res += getPackedSamplerFromInInfo(inInfo);

  const inShape = inInfo.shapeInfo.logicalShape;
  const outShape = outShapeInfo.logicalShape;
  if (inShape.length <= outShape.length) {
    res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
  }
  return res;
}

export function getColPackedOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutputPacked1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutputPacked2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutputPacked3DCoords(
          outShape as [number, number, number], outTexShape);
    default:
      return getOutputPackedNDCoords(outShape, outTexShape);
  }
}

export const SAMPLE_1D_CHANNEL_SNIPPET = `
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 4;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

export const SAMPLE_2D_CHANNEL_SNIPPET = `
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = row * texelsInLogicalRow + (col / 4);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

export const SAMPLE_3D_CHANNEL_SNIPPET = `
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + row * texelsInLogicalRow + (col / 4);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;

function getOutputPacked1DCoords(
    shape: [number], texShape: [number, number]): string {
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  if (packedTexShape[0] === 1) {
    return `
      int getOutputCoords() {
        return 4 * int(resultUV.x * ${packedTexShape[1]}.0);
      }
    `;
  }

  if (packedTexShape[1] === 1) {
    return `
      int getOutputCoords() {
        return int(resultUV.y * ${packedTexShape[0]}.0);
      }
    `;
  }

  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      return resTexRC.x * ${packedTexShape[1]} * 4 + resTexRC.y;
    }
  `;
}

function getOutputPacked3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  const texelsInLogicalRow = Math.ceil(shape[2] / 4);
  const texelsInBatch = texelsInLogicalRow * shape[1];

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = index / ${texelsInLogicalRow};
      int c = imod(index, ${texelsInLogicalRow}) * 4;

      return ivec3(b, r, c);
    }
  `;
}

function getOutputPackedNDCoords(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];

  const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 4);
  const texelsInBatch = texelsInLogicalRow * shape[shape.length - 2];
  let texelsInBatchN = texelsInBatch;
  let batches = ``;
  let coords = 'b, r, c';

  for (let b = 2; b < shape.length - 1; b++) {
    texelsInBatchN *= shape[shape.length - b - 1];
    batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
    coords = `b${b}, ` + coords;
  }

  return `
    ivec${shape.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      ${batches}

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = index / ${texelsInLogicalRow};
      int c = imod(index, ${texelsInLogicalRow}) * 4;

      return ivec${shape.length}(${coords});
    }
  `;
}

function getOutputPacked2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${packedTexShape[0]}, ${
        packedTexShape[1]} * 4));
      }
    `;
  }

  // texels needed to accommodate a logical row
  const texelsInLogicalRow = Math.ceil(shape[1] / 4);

  /**
   * getOutputCoords
   *
   * resTexRC: The rows and columns of the texels. If you move over one
   * texel to the right in the packed texture, you are moving over one column
   * (not two).
   *
   * index: The texel index
   */
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = index / ${texelsInLogicalRow};
      int c = imod(index, ${texelsInLogicalRow}) * 4;

      return ivec2(r, c);
    }
  `;
}

function getPackedSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const glsl = getGlslDifferences();
  return `
    vec4 ${funcName}() {
      return ${glsl.texture2D}(${texName}, halfCR);
    }
  `;
}

function getPackedSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  const glsl = getGlslDifferences();

  return `
    vec4 ${funcName}(int index) {
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  const texNumR = texShape[0];
  const texNumC = texShape[1];
  const glsl = getGlslDifferences();

  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
        return ${glsl.texture2D}(${texName}, uv);
      }
    `;
  }

  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  const valuesPerRow = Math.ceil(shape[1] / 4);

  return `
    vec4 ${funcName}(int row, int col) {
      vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];

  if (shape[0] === 1) {
    const squeezedShape = shape.slice(1);
    const keptDims = [1, 2];
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['b', 'row', 'col'];
    return `
        ${getPackedSamplerFromInInfo(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / 4);
  const texelsInBatch = valuesPerRow * shape[1];
  const glsl = getGlslDifferences();

  return `
    vec4 ${funcName}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getPackedSamplerND(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const rank = shape.length;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [texShape[0], Math.ceil(texShape[1] / 4)];
  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[rank - 1] / 4);
  let texelsInBatch = valuesPerRow * shape[rank - 2];
  let params = `int b, int row, int col`;
  let index = `b * ${texelsInBatch} + row * ${valuesPerRow} + (col / 4)`;
  for (let b = 2; b < rank - 1; b++) {
    params = `int b${b}, ` + params;
    texelsInBatch *= shape[rank - b - 1];
    index = `b${b} * ${texelsInBatch} + ` + index;
  }
  const glsl = getGlslDifferences();
  return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}

function getOutputScalarCoords() {
  return `
    int getOutputCoords() {
      return 0;
    }
  `;
}

function getPackedSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  const broadcastDims = getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);

  const type = getCoordsDataType(outRank);
  const rankDiff = outRank - inRank;
  let coordsSnippet: string;
  const fields = ['x', 'y', 'z', 'w', 'u', 'v'];

  if (inRank === 0) {
    coordsSnippet = '';
  } else if (outRank < 2 && broadcastDims.length >= 1) {
    coordsSnippet = 'coords = 0;';
  } else {
    coordsSnippet =
        broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
            .join('\n');
  }
  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                                .map((s, i) => `coords.${fields[i + rankDiff]}`)
                                .join(', ');
  }

  let output = `return outputValue;`;
  const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
  const isInputScalar = inSize === 1;
  const outSize = util.sizeFromShape(outShapeInfo.logicalShape);
  const isOutputScalar = outSize === 1;

  if (inRank === 1 && !isInputScalar && !isOutputScalar) {
    output = `
      return outputValue;
    `;
  } else if (isInputScalar && !isOutputScalar) {
    output = `
        return vec4(outputValue.x);
    `;
  } else if (broadcastDims.length) {
    const rows = inRank - 2;
    const cols = inRank - 1;

    if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.x);`;
    } else if (broadcastDims.indexOf(rows) > -1) {
      output = `return outputValue`;
    } else if (broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.x);`;
    }
  }

  return `
    vec4 ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
      ${output}
    }
  `;
}

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else if (rank === 5) {
    return 'ivec5';
  } else if (rank === 6) {
    return 'ivec6';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

/** Returns a new input info (a copy) that has a squeezed logical shape. */
function squeezeInputInfo(
    inInfo: InputInfo, squeezedShape: number[]): InputInfo {
  // Deep copy.
  const newInputInfo: InputInfo = JSON.parse(JSON.stringify(inInfo));
  newInputInfo.shapeInfo.logicalShape = squeezedShape;
  return newInputInfo;
}

function getSqueezedParams(params: string[], keptDims: number[]): string {
  return keptDims.map(d => params[d]).join(', ');
}
