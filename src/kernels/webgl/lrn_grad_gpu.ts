import {GPGPUProgram} from './gpgpu_math';

export class LRNGradProgram implements GPGPUProgram {
  variableNames = ['inputImage', 'outputImage', 'dy'];
  outputShape: number[] = [];
  userCode: string;
  depthRadius: number;
  bias: number;
  alpha: number;
  beta: number;
  depth: number;

  constructor(
      inputShape: number[], depthRadius: number, bias: number, alpha: number,
      beta: number) {
    this.outputShape = inputShape;
    this.depth = inputShape[3];
    this.depthRadius = depthRadius;
    this.bias = bias;
    this.alpha = alpha;
    this.beta = beta;
    this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${depthRadius})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${depthRadius} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${alpha}) * norm + float(${bias});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${alpha})
                * float(${beta})
                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${beta});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `;
  }
}
