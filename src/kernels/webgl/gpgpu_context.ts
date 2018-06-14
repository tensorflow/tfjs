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

import {ENV} from '../../environment';
import * as util from '../../util';

import * as gpgpu_util from './gpgpu_util';
import {TextureConfig} from './gpgpu_util';
import * as tex_util from './tex_util';
// tslint:disable-next-line:max-line-length
import {WebGL1DisjointQueryTimerExtension, WebGL2DisjointQueryTimerExtension, WebGL2RenderingContext, WebGLLoseContextExtension, WebGLQuery} from './webgl_types';
import * as webgl_util from './webgl_util';

export class GPGPUContext {
  gl: WebGLRenderingContext;
  textureFloatExtension: {};
  textureHalfFloatExtension: {};
  colorBufferFloatExtension: {};
  colorBufferHalfFloatExtension: {};
  getBufferSubDataAsyncExtension: {};
  loseContextExtension: WebGLLoseContextExtension;
  disjointQueryTimerExtension: WebGL2DisjointQueryTimerExtension|
      WebGL1DisjointQueryTimerExtension;
  vertexBuffer: WebGLBuffer;
  indexBuffer: WebGLBuffer;
  framebuffer: WebGLFramebuffer;
  outputTexture: WebGLTexture|null = null;
  program: WebGLProgram|null = null;
  private disposed = false;
  private autoDebugValidate = false;
  private disjoint: boolean;

  private textureConfig: TextureConfig;

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

      if (!ENV.get('WEBGL_RENDER_FLOAT32_ENABLED')) {
        this.textureHalfFloatExtension =
            webgl_util.getExtensionOrThrow(this.gl, 'OES_texture_half_float');
        this.colorBufferHalfFloatExtension =
            this.gl.getExtension('EXT_color_buffer_half_float');
      }
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

    this.textureConfig =
        gpgpu_util.getTextureConfig(this.gl, this.textureHalfFloatExtension);
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

  public createFloat32MatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createFloat32MatrixTexture(
        this.gl, rows, columns, this.textureConfig);
  }

  public createFloat16MatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createFloat16MatrixTexture(
        this.gl, rows, columns, this.textureConfig);
  }

  public createUnsignedBytesMatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createUnsignedBytesMatrixTexture(
        this.gl, rows, columns, this.textureConfig);
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
    return gpgpu_util.createPackedMatrixTexture(
        this.gl, rows, columns, this.textureConfig);
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
    const numChannels = webgl_util.getNumChannels();
    return gpgpu_util.uploadMatrixToTexture(
        this.gl, texture, rows, columns, matrix, numChannels,
        this.textureConfig);
  }

  public uploadMatrixToPackedTexture(
      texture: WebGLTexture, rows: number, columns: number,
      matrix: Float32Array) {
    this.throwIfDisposed();
    return gpgpu_util.uploadMatrixToPackedTexture(
        this.gl, texture, rows, columns, matrix, this.textureConfig);
  }

  public downloadFloat32MatrixFromOutputTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadFloat32MatrixFromOutputTexture(
            this.gl, rows, columns, this.textureConfig));
  }

  public downloadByteEncodedFloatMatrixFromOutputTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadByteEncodedFloatMatrixFromOutputTexture(
            this.gl, rows, columns, this.textureConfig));
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
            this.gl, this.getBufferSubDataAsyncExtension, rows, columns,
            this.textureConfig));
  }

  public downloadMatrixFromPackedTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadMatrixFromPackedOutputTexture(
            this.gl, rows, columns, this.textureConfig));
  }

  private vertexAttrsAreBound = false;

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
    if (!this.vertexAttrsAreBound) {
      this.setProgram(program);
      this.vertexAttrsAreBound = gpgpu_util.bindVertexProgramAttributeStreams(
          gl, this.program, this.vertexBuffer);
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

  public getUniformLocation(
      program: WebGLProgram, uniformName: string,
      shouldThrow = true): WebGLUniformLocation {
    this.throwIfDisposed();
    if (shouldThrow) {
      return webgl_util.getProgramUniformLocationOrThrow(
          this.gl, program, uniformName);
    } else {
      return webgl_util.getProgramUniformLocation(
          this.gl, program, uniformName);
    }
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

  public executeProgram() {
    this.throwIfDisposed();
    this.throwIfNoProgram();
    const gl = this.gl;
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

  private getQueryTimerExtension(): WebGL1DisjointQueryTimerExtension
      |WebGL2DisjointQueryTimerExtension {
    if (this.disjointQueryTimerExtension == null) {
      this.disjointQueryTimerExtension =
          webgl_util.getExtensionOrThrow(
              this.gl,
              ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2 ?
                  'EXT_disjoint_timer_query_webgl2' :
                  'EXT_disjoint_timer_query') as
              WebGL1DisjointQueryTimerExtension |
          WebGL2DisjointQueryTimerExtension;
    }
    return this.disjointQueryTimerExtension;
  }

  private getQueryTimerExtensionWebGL2(): WebGL2DisjointQueryTimerExtension {
    return this.getQueryTimerExtension();
  }

  private getQueryTimerExtensionWebGL1(): WebGL1DisjointQueryTimerExtension {
    return this.getQueryTimerExtension() as WebGL1DisjointQueryTimerExtension;
  }

  /**
   * Executes a query function which contains GL commands and resolves when
   * the command buffer has finished executing the query.
   * @param queryFn The query function containing GL commands to execute.
   * @return a promise that resolves with the ellapsed GPU time in milliseconds.
   */
  public runQuery(queryFn: () => void): Promise<number> {
    const query = this.beginQuery();
    queryFn();
    this.endQuery();
    return this.pollQueryTime(query);
  }

  beginQuery(): WebGLQuery {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;
      const ext = this.getQueryTimerExtensionWebGL2();

      const query = gl2.createQuery();
      gl2.beginQuery(ext.TIME_ELAPSED_EXT, query);
      return query;
    }
    const ext = this.getQueryTimerExtensionWebGL1();
    const query = ext.createQueryEXT();
    ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
    return query;
  }

  endQuery() {
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;
      const ext = this.getQueryTimerExtensionWebGL2();
      gl2.endQuery(ext.TIME_ELAPSED_EXT);
      return;
    }
    const ext = this.getQueryTimerExtensionWebGL1();
    ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
  }

  private isQueryAvailable(query: WebGLQuery, queryTimerVersion: number):
      boolean {
    if (queryTimerVersion === 0) {
      return true;
    }

    if (queryTimerVersion === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;
      const ext = this.getQueryTimerExtensionWebGL2();

      const available =
          gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE);
      if (this.disjoint == null) {
        this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
      }

      return available && !this.disjoint;
    } else {
      const ext = this.getQueryTimerExtensionWebGL1();

      const available =
          ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);
      if (this.disjoint == null) {
        this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
      }

      return available && !this.disjoint;
    }
  }

  pollQueryTime(query: WebGLQuery): Promise<number> {
    return new Promise<number>(resolve => {
      const queryTimerVersion =
          ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION');
      this.addItemToPoll(
          () => this.isQueryAvailable(query, queryTimerVersion),
          () => resolve(this.getQueryTime(query, queryTimerVersion)));
    });
  }

  private itemsToPoll: PollItem[] = [];

  pollItems(): void {
    // Find the last query that has finished using binary search.
    // All other queries before it are also done.
    const index = binSearchLastTrue(this.itemsToPoll.map(x => x.isDoneFn));
    for (let i = 0; i <= index; ++i) {
      const {resolveFn} = this.itemsToPoll[i];
      resolveFn();
    }
    this.itemsToPoll = this.itemsToPoll.slice(index + 1);
  }

  private addItemToPoll(isDoneFn: () => boolean, resolveFn: () => void) {
    this.itemsToPoll.push({isDoneFn, resolveFn});
    if (this.itemsToPoll.length > 1) {
      // We already have a running loop that polls.
      return;
    }
    // Start a new loop that polls.
    util.repeatedTry(() => {
      this.pollItems();
      // End the loop if no more items to poll.
      return this.itemsToPoll.length === 0;
    });
  }

  private getQueryTime(query: WebGLQuery, queryTimerVersion: number): number {
    if (queryTimerVersion === 0) {
      return null;
    }

    if (queryTimerVersion === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;

      const timeElapsedNanos = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
      // Return milliseconds.
      return timeElapsedNanos / 1000000;
    } else {
      const ext = this.getQueryTimerExtensionWebGL1();

      const timeElapsedNanos =
          ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
      // Return milliseconds.
      return timeElapsedNanos / 1000000;
    }
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

type PollItem = {
  isDoneFn: () => boolean,
  resolveFn: () => void
};

/**
 * Finds the index of the last true element using binary search where
 * evaluation of an entry is expensive.
 */
export function binSearchLastTrue(arr: Array<() => boolean>): number {
  let start = 0;
  let end = arr.length - 1;
  let best = -1;
  while (start <= end) {
    const mid = (start + end) >> 1;
    const isDone = arr[mid]();
    if (isDone) {
      best = mid;
      start = mid + 1;
    } else {
      end = mid - 1;
    }
  }
  return best;
}
