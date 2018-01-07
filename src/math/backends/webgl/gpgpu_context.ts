/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {ENV} from '../../../environment';
import * as util from '../../../util';
import * as gpgpu_util from './gpgpu_util';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';
import {WebGLLoseContextExtension} from './webgl_util';

export class GPGPUContext {
  gl: WebGLRenderingContext;
  textureFloatExtension: {};
  colorBufferFloatExtension: {};
  getBufferSubDataAsyncExtension: {};
  loseContextExtension: WebGLLoseContextExtension;
  vertexBuffer: WebGLBuffer;
  indexBuffer: WebGLBuffer;
  framebuffer: WebGLFramebuffer;
  outputTexture: WebGLTexture|null = null;
  program: WebGLProgram|null = null;
  private disposed = false;
  private autoDebugValidate = false;

  constructor(gl?: WebGLRenderingContext) {
    if (gl != null) {
      this.gl = gl;
    } else {
      this.gl = gpgpu_util.createWebGLContext();
    }
    // WebGL 2.0 enables texture floats without an extension.
    if (ENV.get('WEBGL_VERSION') === 1) {
      this.textureFloatExtension =
          webgl_util.getExtensionOrThrow(this.gl, 'OES_texture_float');
      this.colorBufferFloatExtension =
          this.gl.getExtension('WEBGL_color_buffer_float');
    } else {
      this.colorBufferFloatExtension =
          webgl_util.getExtensionOrThrow(this.gl, 'EXT_color_buffer_float');
    }

    this.loseContextExtension =
        webgl_util.getExtensionOrThrow(this.gl, 'WEBGL_lose_context') as
        WebGLLoseContextExtension;

    if (ENV.get('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED')) {
      this.getBufferSubDataAsyncExtension =
          this.gl.getExtension('WEBGL_get_buffer_sub_data_async');
    }

    this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
    this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
    this.framebuffer = webgl_util.createFramebuffer(this.gl);
  }

  public dispose() {
    if (this.disposed) {
      return;
    }
    if (this.program != null) {
      console.warn(
          'Disposing a GPGPUContext that still has a bound WebGLProgram.' +
          ' This is probably a resource leak, delete the program with ' +
          'GPGPUContext.deleteProgram before disposing.');
    }
    if (this.outputTexture != null) {
      console.warn(
          'Disposing a GPGPUContext that still has a bound output matrix ' +
          'texture.  This is probably a resource leak, delete the output ' +
          'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
          'disposing.');
    }
    const gl = this.gl;
    webgl_util.callAndCheck(gl, () => gl.finish());
    webgl_util.callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
    webgl_util.callAndCheck(gl, () => gl.deleteFramebuffer(this.framebuffer));
    webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, null));
    webgl_util.callAndCheck(gl, () => gl.deleteBuffer(this.vertexBuffer));
    webgl_util.callAndCheck(
        gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null));
    webgl_util.callAndCheck(gl, () => gl.deleteBuffer(this.indexBuffer));
    this.loseContextExtension.loseContext();
    this.disposed = true;
  }

  public enableAutomaticDebugValidation(enabled: boolean) {
    this.autoDebugValidate = enabled;
    webgl_util.enableDebugWebGLErrorChecking(enabled);
  }

  public createMatrixTexture(rows: number, columns: number): WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createMatrixTexture(this.gl, rows, columns);
  }

  public uploadPixelDataToTexture(
      texture: WebGLTexture,
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement) {
    this.throwIfDisposed();
    gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
  }

  public createPackedMatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns);
  }

  public deleteMatrixTexture(texture: WebGLTexture) {
    this.throwIfDisposed();
    if (this.outputTexture === texture) {
      webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
      this.outputTexture = null;
    }
    webgl_util.callAndCheck(this.gl, () => this.gl.deleteTexture(texture));
  }

  public uploadMatrixToTexture(
      texture: WebGLTexture, rows: number, columns: number,
      matrix: Float32Array) {
    this.throwIfDisposed();
    const numChannels = 1;
    return gpgpu_util.uploadMatrixToTexture(
        this.gl, texture, rows, columns, matrix, numChannels);
  }

  public uploadMatrixToPackedTexture(
      texture: WebGLTexture, rows: number, columns: number,
      matrix: Float32Array) {
    this.throwIfDisposed();
    return gpgpu_util.uploadMatrixToPackedTexture(
        this.gl, texture, rows, columns, matrix);
  }

  public downloadMatrixFromTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () =>
            gpgpu_util.downloadMatrixFromOutputTexture(this.gl, rows, columns));
  }

  public async downloadMatrixFromTextureAsync(
      texture: WebGLTexture, rows: number,
      columns: number): Promise<Float32Array> {
    if (this.getBufferSubDataAsyncExtension == null) {
      throw new Error(
          `Cannot download matrix from output texture asynchronously, ` +
          `WEBGL_get_buffer_sub_data_async is not enabled.`);
    }

    return this.downloadMatrixDriverAsync(
        texture,
        () => gpgpu_util.downloadMatrixFromOutputTextureAsync(
            this.gl, this.getBufferSubDataAsyncExtension, rows, columns));
  }

  public downloadMatrixFromRGBAColorTexture(
      texture: WebGLTexture, rows: number, columns: number,
      channels: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadMatrixFromRGBAColorTexture(
            this.gl, rows, columns, channels));
  }

  public downloadMatrixFromPackedTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadMatrixFromPackedOutputTexture(
            this.gl, rows, columns));
  }

  public createProgram(fragmentShaderSource: string): WebGLProgram {
    this.throwIfDisposed();
    const gl = this.gl;
    const fragmentShader: WebGLShader =
        webgl_util.createFragmentShader(gl, fragmentShaderSource);
    const vertexShader: WebGLShader = gpgpu_util.createVertexShader(gl);
    const program: WebGLProgram = webgl_util.createProgram(gl);
    webgl_util.callAndCheck(gl, () => gl.attachShader(program, vertexShader));
    webgl_util.callAndCheck(gl, () => gl.attachShader(program, fragmentShader));
    webgl_util.linkProgram(gl, program);
    if (this.autoDebugValidate) {
      webgl_util.validateProgram(gl, program);
    }

    return program;
  }

  public deleteProgram(program: WebGLProgram) {
    this.throwIfDisposed();
    if (program === this.program) {
      this.program = null;
    }
    if (program != null) {
      webgl_util.callAndCheck(this.gl, () => this.gl.deleteProgram(program));
    }
  }

  public setProgram(program: WebGLProgram|null) {
    this.throwIfDisposed();
    this.program = program;
    if ((this.program != null) && this.autoDebugValidate) {
      webgl_util.validateProgram(this.gl, this.program);
    }
    webgl_util.callAndCheck(this.gl, () => this.gl.useProgram(program));
  }

  public getUniformLocation(program: WebGLProgram, uniformName: string):
      WebGLUniformLocation {
    this.throwIfDisposed();
    return webgl_util.getProgramUniformLocationOrThrow(
        this.gl, program, uniformName);
  }

  public getAttributeLocation(program: WebGLProgram, attribute: string):
      number {
    this.throwIfDisposed();
    return webgl_util.callAndCheck(
        this.gl, () => this.gl.getAttribLocation(program, attribute));
  }

  public getUniformLocationNoThrow(program: WebGLProgram, uniformName: string):
      WebGLUniformLocation {
    this.throwIfDisposed();
    return this.gl.getUniformLocation(program, uniformName);
  }

  public setInputMatrixTexture(
      inputMatrixTexture: WebGLTexture, uniformLocation: WebGLUniformLocation,
      textureUnit: number) {
    this.throwIfDisposed();
    this.throwIfNoProgram();
    webgl_util.bindTextureToProgramUniformSampler(
        this.gl, this.program, inputMatrixTexture, uniformLocation,
        textureUnit);
  }

  public setOutputMatrixTexture(
      outputMatrixTexture: WebGLTexture, rows: number, columns: number) {
    this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
  }

  public setOutputPackedMatrixTexture(
      outputPackedMatrixTexture: WebGLTexture, rows: number, columns: number) {
    this.throwIfDisposed();
    const [width, height] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
    this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
  }

  public setOutputMatrixWriteRegion(
      startRow: number, numRows: number, startColumn: number,
      numColumns: number) {
    this.setOutputMatrixWriteRegionDriver(
        startColumn, startRow, numColumns, numRows);
  }

  public setOutputPackedMatrixWriteRegion(
      startRow: number, numRows: number, startColumn: number,
      numColumns: number) {
    throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
  }

  public debugValidate() {
    if (this.program != null) {
      webgl_util.validateProgram(this.gl, this.program);
    }
    webgl_util.validateFramebuffer(this.gl);
  }

  public executeProgram(attribLocations?: {[name: string]: number}) {
    this.throwIfDisposed();
    this.throwIfNoProgram();
    const gl = this.gl;
    gpgpu_util.bindVertexProgramAttributeStreams(
        gl, this.program, this.vertexBuffer, attribLocations);
    if (this.autoDebugValidate) {
      this.debugValidate();
    }
    webgl_util.callAndCheck(
        gl, () => gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0));
  }

  public blockUntilAllProgramsCompleted() {
    this.throwIfDisposed();
    webgl_util.callAndCheck(this.gl, () => this.gl.finish());
  }

  /**
   * Executes a query function which contains GL commands and resolves when
   * the command buffer has finished executing the query.
   * @param queryFn The query function containing GL commands to execute.
   * @return a promise that resolves with the ellapsed time in milliseconds.
   */
  public runQuery(queryFn: () => void): Promise<number> {
    if (ENV.get('WEBGL_VERSION') === 2) {
      return this.runQueryWebGL2(queryFn);
    }
    return this.runQueryWebGL1(queryFn);
  }

  private runQueryWebGL2(benchmark: () => void): Promise<number> {
    const ext = webgl_util.getExtensionOrThrow(
        this.gl, 'EXT_disjoint_timer_query_webgl2');
    // tslint:disable-next-line:no-any
    const query = (this.gl as any).createQuery();

    // tslint:disable-next-line:no-any
    (this.gl as any).beginQuery((ext as any).TIME_ELAPSED_EXT, query);

    benchmark();

    // tslint:disable-next-line:no-any
    (this.gl as any).endQuery((ext as any).TIME_ELAPSED_EXT);

    return new Promise<number>((resolve, reject) => {
      const queryGPU = () => {
        const available =
            // tslint:disable-next-line:no-any
            (this.gl as any)
                .getQueryParameter(
                    // tslint:disable-next-line:no-any
                    query, (this.gl as any).QUERY_RESULT_AVAILABLE);

        const disjoint =
            // tslint:disable-next-line:no-any
            this.gl.getParameter((ext as any).GPU_DISJOINT_EXT);
        return available && !disjoint;
      };

      const getTimeElapsed = () => {
        const timeElapsedNanos =
            // tslint:disable-next-line:no-any
            (this.gl as any)
                // tslint:disable-next-line:no-any
                .getQueryParameter(query, (this.gl as any).QUERY_RESULT);
        // Return milliseconds.
        resolve(timeElapsedNanos / 1000000);
      };

      const resolveWithWarning = () => {
        console.warn('Disjoint query timer never available.');
        resolve(-1);
      };

      util.repeatedTry(queryGPU).then(getTimeElapsed).catch(resolveWithWarning);
    });
  }

  private runQueryWebGL1(benchmark: () => void): Promise<number> {
    const ext = webgl_util.getExtensionOrThrow(
                    // tslint:disable-next-line:no-any
                    this.gl, 'EXT_disjoint_timer_query') as any;
    const query = ext.createQueryEXT();

    ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);

    benchmark();

    ext.endQueryEXT(ext.TIME_ELAPSED_EXT);

    return new Promise<number>((resolve, reject) => {
      const queryGPU = () => {
        const available =
            ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);

        const disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);

        return available && !disjoint;
      };

      const getTimeElapsed = () => {
        const timeElapsedNanos =
            ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
        // Return milliseconds.
        resolve(timeElapsedNanos / 1000000);
      };

      const resolveWithWarning = () => {
        console.warn('Disjoint query timer never available.');
        resolve(-1);
      };

      util.repeatedTry(queryGPU).then(getTimeElapsed).catch(resolveWithWarning);
    });
  }

  private downloadMatrixDriverSetup(texture: WebGLTexture) {
    this.throwIfDisposed();
    webgl_util.bindColorTextureToFramebuffer(
        this.gl, texture, this.framebuffer);
    if (this.autoDebugValidate) {
      webgl_util.validateFramebuffer(this.gl);
    }
  }

  private downloadMatrixDriverTeardown() {
    if (this.outputTexture != null) {
      webgl_util.bindColorTextureToFramebuffer(
          this.gl, this.outputTexture, this.framebuffer);
      if (this.autoDebugValidate) {
        webgl_util.validateFramebuffer(this.gl);
      }
    } else {
      webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
    }
  }

  private downloadMatrixDriver(
      texture: WebGLTexture,
      downloadAndDecode: () => Float32Array): Float32Array {
    this.downloadMatrixDriverSetup(texture);
    const result = downloadAndDecode();
    this.downloadMatrixDriverTeardown();

    return result;
  }

  private async downloadMatrixDriverAsync(
      texture: WebGLTexture,
      downloadAndDecode: () => Promise<Float32Array>): Promise<Float32Array> {
    this.downloadMatrixDriverSetup(texture);
    const result = await downloadAndDecode();
    this.downloadMatrixDriverTeardown();

    return result;
  }

  private setOutputMatrixTextureDriver(
      outputMatrixTextureMaybePacked: WebGLTexture, width: number,
      height: number) {
    this.throwIfDisposed();
    const gl = this.gl;
    webgl_util.bindColorTextureToFramebuffer(
        gl, outputMatrixTextureMaybePacked, this.framebuffer);
    if (this.autoDebugValidate) {
      webgl_util.validateFramebuffer(gl);
    }
    this.outputTexture = outputMatrixTextureMaybePacked;
    webgl_util.callAndCheck(gl, () => gl.viewport(0, 0, width, height));
    webgl_util.callAndCheck(gl, () => gl.scissor(0, 0, width, height));
  }

  private setOutputMatrixWriteRegionDriver(
      x: number, y: number, width: number, height: number) {
    this.throwIfDisposed();
    webgl_util.callAndCheck(
        this.gl, () => this.gl.scissor(x, y, width, height));
  }

  private throwIfDisposed() {
    if (this.disposed) {
      throw new Error('Attempted to use disposed GPGPUContext.');
    }
  }

  private throwIfNoProgram() {
    if (this.program == null) {
      throw new Error('No GPU program is currently set.');
    }
  }
}
