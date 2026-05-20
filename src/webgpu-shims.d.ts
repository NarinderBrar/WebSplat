/**
 * WebGPU runtime constant shims.
 *
 * Babylon ships ambient interface declarations for WebGPU types in
 * `@babylonjs/core/Engines/engine.d.ts`, but it does NOT declare the
 * runtime-defined constants (`GPUBufferUsage`, `GPUShaderStage`,
 * `GPUTextureUsage`, `GPUColorWrite`). Installing `@webgpu/types` adds a
 * branded `__brand` discriminator that conflicts with Babylon's stubs.
 *
 * This file just patches in the constant tables we need - no interface
 * collisions, no extra dependency.
 */

declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly INDEX: number;
  readonly VERTEX: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
  readonly INDIRECT: number;
  readonly QUERY_RESOLVE: number;
};

declare const GPUTextureUsage: {
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly TEXTURE_BINDING: number;
  readonly STORAGE_BINDING: number;
  readonly RENDER_ATTACHMENT: number;
};

declare const GPUShaderStage: {
  readonly VERTEX: number;
  readonly FRAGMENT: number;
  readonly COMPUTE: number;
};

declare const GPUColorWrite: {
  readonly RED: number;
  readonly GREEN: number;
  readonly BLUE: number;
  readonly ALPHA: number;
  readonly ALL: number;
};
