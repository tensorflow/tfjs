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
import {PixelData, TypedArray} from '../../types';
import * as util from '../../util';

import {getWebGLContext, setWebGLContext} from './canvas_util';
import * as gpgpu_util from './gpgpu_util';
import * as tex_util from './tex_util';
import {TextureConfig} from './tex_util';
import {WebGL1DisjointQueryTimerExtension, WebGL2DisjointQueryTimerExtension} from './webgl_types';
import * as webgl_util from './webgl_util';

export interface FenceContext {
  query: WebGLQuery|WebGLSync;
  isFencePassed(): boolean;
}

export class GPGPUContext {
  gl: WebGLRenderingContext;
  textureFloatExtension: {};
  textureHalfFloatExtension: {};
  colorBufferFloatExtension: {};
  colorBufferHalfFloatExtension: {};
  disjointQueryTimerExtension: WebGL2DisjointQueryTimerExtension|
      WebGL1DisjointQueryTimerExtension;
  vertexBuffer: WebGLBuffer;
  indexBuffer: WebGLBuffer;
  framebuffer: WebGLFramebuffer;
  outputTexture: WebGLTexture|null = null;
  program: WebGLProgram|null = null;
  private disposed = false;
  private disjoint: boolean;
  private textureConfig: TextureConfig;

  constructor(gl?: WebGLRenderingContext) {
    const glVersion = ENV.getNumber('WEBGL_VERSION');
    if (gl != null) {
      this.gl = gl;
      setWebGLContext(glVersion, gl);
    } else {
      this.gl = getWebGLContext(glVersion);
    }
    // WebGL 2.0 enables texture floats without an extension.
    if (ENV.getNumber('WEBGL_VERSION') === 1) {
      this.textureFloatExtension = webgl_util.getExtensionOrThrow(
          this.gl, this.debug, 'OES_texture_float');
      this.colorBufferFloatExtension =
          this.gl.getExtension('WEBGL_color_buffer_float');

      if (!ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')) {
        this.textureHalfFloatExtension = webgl_util.getExtensionOrThrow(
            this.gl, this.debug, 'OES_texture_half_float');
        this.colorBufferHalfFloatExtension =
            this.gl.getExtension('EXT_color_buffer_half_float');
      }
    } else {
      const COLOR_BUFFER_FLOAT = 'EXT_color_buffer_float';
      const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
      if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_FLOAT)) {
        this.colorBufferFloatExtension =
            this.gl.getExtension(COLOR_BUFFER_FLOAT);
      } else if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
        this.colorBufferHalfFloatExtension =
            this.gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
      } else {
        throw new Error('GL context does not support color renderable floats');
      }
    }

    this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl, this.debug);
    this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl, this.debug);
    this.framebuffer = webgl_util.createFramebuffer(this.gl, this.debug);

    this.textureConfig =
        tex_util.getTextureConfig(this.gl, this.textureHalfFloatExtension);
  }

  private get debug(): boolean {
    return ENV.getBool('DEBUG');
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
    webgl_util.callAndCheck(gl, this.debug, () => gl.finish());
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.deleteFramebuffer(this.framebuffer));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, null));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.deleteBuffer(this.indexBuffer));
    this.disposed = true;
  }

  public createFloat32MatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createFloat32MatrixTexture(
        this.gl, this.debug, rows, columns, this.textureConfig);
  }

  public createFloat16MatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createFloat16MatrixTexture(
        this.gl, this.debug, rows, columns, this.textureConfig);
  }

  public createUnsignedBytesMatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createUnsignedBytesMatrixTexture(
        this.gl, this.debug, rows, columns, this.textureConfig);
  }

  public uploadPixelDataToTexture(
      texture: WebGLTexture,
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement) {
    this.throwIfDisposed();
    gpgpu_util.uploadPixelDataToTexture(this.gl, this.debug, texture, pixels);
  }

  public uploadDenseMatrixToTexture(
      texture: WebGLTexture, width: number, height: number, data: TypedArray) {
    this.throwIfDisposed();
    gpgpu_util.uploadDenseMatrixToTexture(
        this.gl, this.debug, texture, width, height, data, this.textureConfig);
  }

  public createFloat16PackedMatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createFloat16PackedMatrixTexture(
        this.gl, this.debug, rows, columns, this.textureConfig);
  }

  public createPackedMatrixTexture(rows: number, columns: number):
      WebGLTexture {
    this.throwIfDisposed();
    return gpgpu_util.createPackedMatrixTexture(
        this.gl, this.debug, rows, columns, this.textureConfig);
  }

  public deleteMatrixTexture(texture: WebGLTexture) {
    this.throwIfDisposed();
    if (this.outputTexture === texture) {
      webgl_util.unbindColorTextureFromFramebuffer(
          this.gl, this.debug, this.framebuffer);
      this.outputTexture = null;
    }
    webgl_util.callAndCheck(
        this.gl, this.debug, () => this.gl.deleteTexture(texture));
  }

  public downloadByteEncodedFloatMatrixFromOutputTexture(
      texture: WebGLTexture, rows: number, columns: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadByteEncodedFloatMatrixFromOutputTexture(
            this.gl, this.debug, rows, columns, this.textureConfig));
  }

  public downloadPackedMatrixFromBuffer(
      buffer: WebGLBuffer, batch: number, rows: number, columns: number,
      physicalRows: number, physicalCols: number): Float32Array {
    return gpgpu_util.downloadPackedMatrixFromBuffer(
        this.gl, buffer, batch, rows, columns, physicalRows, physicalCols,
        this.textureConfig);
  }

  public downloadFloat32MatrixFromBuffer(buffer: WebGLBuffer, size: number):
      Float32Array {
    return gpgpu_util.downloadFloat32MatrixFromBuffer(this.gl, buffer, size);
  }

  public createBufferFromTexture(
      texture: WebGLTexture, rows: number, columns: number): WebGLBuffer {
    this.bindTextureToFrameBuffer(texture);
    const result = gpgpu_util.createBufferFromOutputTexture(
        this.gl as WebGL2RenderingContext, this.debug, rows, columns,
        this.textureConfig);
    this.unbindTextureToFrameBuffer();
    return result;
  }

  public createAndWaitForFence(): Promise<void> {
    const fenceContext = this.createFence(this.gl);
    return this.pollFence(fenceContext);
  }

  private createFence(gl: WebGLRenderingContext): FenceContext {
    let query: WebGLQuery|WebGLSync;
    let isFencePassed: () => boolean;

    if (ENV.getBool('WEBGL_FENCE_API_ENABLED')) {
      const gl2 = gl as WebGL2RenderingContext;

      const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();

      isFencePassed = () => {
        const status = gl2.clientWaitSync(sync, 0, 0);
        return status === gl2.ALREADY_SIGNALED ||
            status === gl2.CONDITION_SATISFIED;
      };

      query = sync;
    } else if (
        ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
      query = this.beginQuery();
      this.endQuery();
      isFencePassed = () => this.isQueryAvailable(
          query, ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
    } else {
      // If we have no way to fence, return true immediately. This will fire in
      // WebGL 1.0 when there is no disjoint query timer. In this case, because
      // the fence passes immediately, we'll immediately ask for a download of
      // the texture, which will cause the UI thread to hang.
      isFencePassed = () => true;
    }

    return {query, isFencePassed};
  }

  public downloadMatrixFromPackedTexture(
      texture: WebGLTexture, physicalRows: number,
      physicalCols: number): Float32Array {
    return this.downloadMatrixDriver(
        texture,
        () => gpgpu_util.downloadMatrixFromPackedOutputTexture(
            this.gl, this.debug, physicalRows, physicalCols));
  }

  private vertexAttrsAreBound = false;

  public createProgram(fragmentShaderSource: string): WebGLProgram {
    this.throwIfDisposed();
    const gl = this.gl;
    const fragmentShader: WebGLShader =
        webgl_util.createFragmentShader(gl, this.debug, fragmentShaderSource);
    const vertexShader: WebGLShader =
        gpgpu_util.createVertexShader(gl, this.debug);
    const program: WebGLProgram = webgl_util.createProgram(
        gl,
        this.debug,
    );
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.attachShader(program, vertexShader));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.attachShader(program, fragmentShader));
    webgl_util.linkProgram(gl, this.debug, program);
    if (this.debug) {
      webgl_util.validateProgram(gl, this.debug, program);
    }
    if (!this.vertexAttrsAreBound) {
      this.setProgram(program);
      this.vertexAttrsAreBound = gpgpu_util.bindVertexProgramAttributeStreams(
          gl, this.debug, this.program, this.vertexBuffer);
    }
    return program;
  }

  public deleteProgram(program: WebGLProgram) {
    this.throwIfDisposed();
    if (program === this.program) {
      this.program = null;
    }
    if (program != null) {
      webgl_util.callAndCheck(
          this.gl, this.debug, () => this.gl.deleteProgram(program));
    }
  }

  public setProgram(program: WebGLProgram|null) {
    this.throwIfDisposed();
    this.program = program;
    if ((this.program != null) && this.debug) {
      webgl_util.validateProgram(this.gl, this.debug, this.program);
    }
    webgl_util.callAndCheck(
        this.gl, this.debug, () => this.gl.useProgram(program));
  }

  public getUniformLocation(
      program: WebGLProgram, uniformName: string,
      shouldThrow = true): WebGLUniformLocation {
    this.throwIfDisposed();
    if (shouldThrow) {
      return webgl_util.getProgramUniformLocationOrThrow(
          this.gl, this.debug, program, uniformName);
    } else {
      return webgl_util.getProgramUniformLocation(
          this.gl, program, uniformName);
    }
  }

  public getAttributeLocation(program: WebGLProgram, attribute: string):
      number {
    this.throwIfDisposed();
    return webgl_util.callAndCheck(
        this.gl, this.debug,
        () => this.gl.getAttribLocation(program, attribute));
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
        this.gl, this.debug, this.program, inputMatrixTexture, uniformLocation,
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
      webgl_util.validateProgram(this.gl, this.debug, this.program);
    }
    webgl_util.validateFramebuffer(this.gl);
  }

  public executeProgram() {
    this.throwIfDisposed();
    this.throwIfNoProgram();
    const gl = this.gl;
    if (this.debug) {
      this.debugValidate();
    }
    webgl_util.callAndCheck(
        gl, this.debug,
        () => gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0));
  }

  public blockUntilAllProgramsCompleted() {
    this.throwIfDisposed();
    webgl_util.callAndCheck(this.gl, this.debug, () => this.gl.finish());
  }

  private getQueryTimerExtension(): WebGL1DisjointQueryTimerExtension
      |WebGL2DisjointQueryTimerExtension {
    if (this.disjointQueryTimerExtension == null) {
      this.disjointQueryTimerExtension =
          webgl_util.getExtensionOrThrow(
              this.gl, this.debug,
              ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') ===
                      2 ?
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

  beginQuery(): WebGLQuery {
    if (ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;
      const ext = this.getQueryTimerExtensionWebGL2();

      const query = gl2.createQuery();
      gl2.beginQuery(ext.TIME_ELAPSED_EXT, query);
      return query;
    }
    const ext = this.getQueryTimerExtensionWebGL1();
    const query = ext.createQueryEXT() as WebGLQuery;
    ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
    return query;
  }

  endQuery() {
    if (ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
      const gl2 = this.gl as WebGL2RenderingContext;
      const ext = this.getQueryTimerExtensionWebGL2();
      gl2.endQuery(ext.TIME_ELAPSED_EXT);
      return;
    }
    const ext = this.getQueryTimerExtensionWebGL1();
    ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
  }

  public async waitForQueryAndGetTime(query: WebGLQuery): Promise<number> {
    await util.repeatedTry(
        () => this.disposed ||  // while testing contexts are created / disposed
                                // in rapid succession, so without this check we
                                // may poll for the query timer indefinitely
            this.isQueryAvailable(
                query,
                ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') as
                    number));
    return this.getQueryTime(
        query, ENV.getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
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

  pollFence(fenceContext: FenceContext) {
    return new Promise<void>(resolve => {
      this.addItemToPoll(() => fenceContext.isFencePassed(), () => resolve());
    });
  }

  private itemsToPoll: PollItem[] = [];

  pollItems(): void {
    // Find the last query that has finished.
    const index = linearSearchLastTrue(this.itemsToPoll.map(x => x.isDoneFn));
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

  private bindTextureToFrameBuffer(texture: WebGLTexture) {
    this.throwIfDisposed();
    webgl_util.bindColorTextureToFramebuffer(
        this.gl, this.debug, texture, this.framebuffer);
    if (this.debug) {
      webgl_util.validateFramebuffer(this.gl);
    }
  }

  private unbindTextureToFrameBuffer() {
    if (this.outputTexture != null) {
      webgl_util.bindColorTextureToFramebuffer(
          this.gl, this.debug, this.outputTexture, this.framebuffer);
      if (this.debug) {
        webgl_util.validateFramebuffer(this.gl);
      }
    } else {
      webgl_util.unbindColorTextureFromFramebuffer(
          this.gl, this.debug, this.framebuffer);
    }
  }

  private downloadMatrixDriver(
      texture: WebGLTexture,
      downloadAndDecode: () => Float32Array): Float32Array {
    this.bindTextureToFrameBuffer(texture);
    const result = downloadAndDecode();
    this.unbindTextureToFrameBuffer();

    return result;
  }

  private setOutputMatrixTextureDriver(
      outputMatrixTextureMaybePacked: WebGLTexture, width: number,
      height: number) {
    this.throwIfDisposed();
    const gl = this.gl;
    webgl_util.bindColorTextureToFramebuffer(
        gl, this.debug, outputMatrixTextureMaybePacked, this.framebuffer);
    if (this.debug) {
      webgl_util.validateFramebuffer(gl);
    }
    this.outputTexture = outputMatrixTextureMaybePacked;
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.viewport(0, 0, width, height));
    webgl_util.callAndCheck(
        gl, this.debug, () => gl.scissor(0, 0, width, height));
  }

  private setOutputMatrixWriteRegionDriver(
      x: number, y: number, width: number, height: number) {
    this.throwIfDisposed();
    webgl_util.callAndCheck(
        this.gl, this.debug, () => this.gl.scissor(x, y, width, height));
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
 * Finds the index of the last true element using linear search.
 * Note: We can't do binary search because Chrome expects us to explicitly
 * test all fences before download:
 * https://github.com/tensorflow/tfjs/issues/1145
 */
export function linearSearchLastTrue(arr: Array<() => boolean>): number {
  let i = 0;
  for (; i < arr.length; ++i) {
    const isDone = arr[i]();
    if (!isDone) {
      break;
    }
  }
  return i - 1;
}
