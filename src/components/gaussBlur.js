import blurCode from './gaussBlur.wgsl';
import renderCode from './render.wgsl';


class Kernel {
  /** @param {Float32Array} values  */
  constructor(values) {
    this.values = values;
    this.sum = values.reduce((a, b) => a + b, 0);
  }

  /** @type {number}  */
  sum;
  /** @type {Float32Array} */
  values;

  packed_data() {
    return new Float32Array([this.sum, ...this.values]);
  }

  size() {
    return this.values.length;
  }

  /**
   * @param {number} sigma
   * @returns {Kernel}
   */
  static kernel(sigma) {
    const size = Kernel.kernel_size_for_sigma(sigma);
    const values = new Float32Array(size);
    const radius = Math.floor((size - 1) / 2);
    for (let i = 0; i < radius; i++) {
      const value = Kernel.normalized_probablility_density_function(i, sigma);
      values[radius + i] = value;
      values[radius - i] = value;
    }
    return new Kernel(values);
  }

  static kernel_size_for_sigma(sigma) {
    return 2 * Math.ceil(3 * sigma) + 1;
  }

  static normalized_probablility_density_function(x, sigma) {
    return 0.39894 * Math.exp(-0.5 * x * x / (sigma * sigma)) / sigma
  }

  static compute_work_group_count(
    [width, height],
    [workgroup_width, workgroup_height]
  ) {
    const w = Math.ceil((width + workgroup_width - 1) / workgroup_width);
    const h = Math.ceil((height + workgroup_height - 1) / workgroup_height);

    return [w, h]
  }
}

export async function gaussBlur() {
  const radius = 100;
  const sigma = 95;
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported on this browser.');
  }
  const source = fetch('./84*84.png');
  const img = await (await source).blob();

  const kernelInst = Kernel.kernel(sigma);
  const kernel_size = kernelInst.size();

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const imageBitmap = await createImageBitmap(img);
  const imgData = imageBitmap;

  const { width, height } = imgData;

  const size = [width, height, 1];

  const texture = device.createTexture({
    size,
    format: "rgba8unorm",
    dimension: '2d',
    sampleCount: 1,
    mipLevelCount: 1,
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
  })

  device.queue.copyExternalImageToTexture(
    { source: imageBitmap, },
    { texture, },
    { width, height, }
  );

  const vertical_pass_texture = device.createTexture({
    label: 'vertical_pass_texture',
    size,
    format: "rgba8unorm",
    dimension: '2d',
    sampleCount: 1,
    mipLevelCount: 1,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
  })

  const horizontal_pass_texture = device.createTexture({
    label: 'horizontal_pass_texture',
    size,
    format: "rgba8unorm",
    dimension: '2d',
    sampleCount: 1,
    mipLevelCount: 1,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });



  const shaderModule = device.createShaderModule({
    label: 'shaderModule',
    code: blurCode,
  });

  const computePipeline = device.createComputePipeline({
    label: 'pipeline',
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: "main"
    }
  });

  const settings = device.createBuffer({
    label: 'img info',
    size: 4, // u32
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(settings, 0, new Uint32Array([kernel_size]), 0);

  const kernel = device.createBuffer({
    label: 'packed data',
    size: (kernel_size + 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(kernel, 0, new Float32Array([...kernelInst.packed_data()]), 0);

  const compute_constants = device.createBindGroup({
    label: 'compute constants',
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: settings,
        }
      },
      {
        binding: 1,
        resource: {
          buffer: kernel,
        }
      },
    ],
  })

  const vertical = device.createBuffer({
    label: 'Orientation',
    size: 4, // u8
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vertical, 0, new Uint32Array([1]), 0);

  const horizontal = device.createBuffer({
    label: 'Orientation',
    size: 4, // u8
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(horizontal, 0, new Uint32Array([0]), 0);


  const vertical_bind_group = device.createBindGroup({
    label: 'Texture bind group',
    layout: computePipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 0,
        resource: texture.createView(),
      },
      {
        binding: 1,
        resource: vertical_pass_texture.createView(),
      },
      {
        binding: 2,
        resource: {
          buffer: vertical,
        }
      },
    ]
  })


  const horizontal_bind_group = device.createBindGroup({
    label: 'Texture bind group',
    layout: computePipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 0,
        resource: vertical_pass_texture.createView(),
      },
      {
        binding: 1,
        resource: horizontal_pass_texture.createView(),
      },
      {
        binding: 2,
        resource: {
          buffer: horizontal,
        }
      },
    ]
  })

  const commandEncoder = device.createCommandEncoder();

  const computePass = commandEncoder.beginComputePass({
    label: 'gaussian blur',
  });

  computePass.setPipeline(computePipeline);

  computePass.setBindGroup(0, compute_constants, []);
  let w, h;

  computePass.setBindGroup(1, vertical_bind_group, []);
  [w, h] = Kernel.compute_work_group_count([width, height], [128, 1]);
  computePass.dispatchWorkgroups(w, h, 1);

  computePass.setBindGroup(1, horizontal_bind_group, []);
  [h, w] = Kernel.compute_work_group_count([width, height], [1, 128]);
  computePass.dispatchWorkgroups(w, h, 1);

  computePass.end();

  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("webgpu");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
      format: "float32x2",
      offset: 0,
      shaderLocation: 0, // Position, see vertex shader
    }],
  };

  const pipeline = device.createRenderPipeline({
    label: 'render to canvas',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: renderCode,
      }),
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: device.createShaderModule({
        code: renderCode,
      }),
      entryPoint: "fragmentMain",
      targets: [
        {
          format: canvasFormat,
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const showResultBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: sampler,
      },
      {
        binding: 1,
        resource: horizontal_pass_texture.createView(),
      },
    ],
  });

  const renderEncoder = commandEncoder.beginRenderPass({
    label: 'canvas',
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0.1, a: 1 }, // New line
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  const vertices = new Float32Array([
    //   X,    Y,
    -0.6, -0.6,
    0.6, -0.6,
    0.6, 0.6,

    -0.6, -0.6,
    0.6, 0.6,
    -0.6, 0.6,
  ]);
  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
  renderEncoder.setVertexBuffer(0, vertexBuffer);
  renderEncoder.setPipeline(pipeline);
  renderEncoder.setBindGroup(0, showResultBindGroup);
  renderEncoder.draw(vertices.length / 2, 2);
  renderEncoder.end();

  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const buffer = device.createBuffer({
    label: 'output img',
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  commandEncoder.copyTextureToBuffer(
    {
      texture: horizontal_pass_texture,
    },
    {
      buffer: buffer,
      bytesPerRow,
      rowsPerImage: height,
    },
    {
      width,
      height,
      depthOrArrayLayers: 1,
    }
  );

  device.queue.submit([commandEncoder.finish()]);


  await buffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = buffer.getMappedRange();

  
  const copy = new Uint8Array(arrayBuffer);
  const data = new Uint8Array(width * height * 4);
  const textureWidth = bytesPerRow / 4;

  for (let h = 0; h < height; h++) {
    const dataToCopy = copy.slice(h * bytesPerRow, (h + 1) * bytesPerRow);
    for (let w = 0; w < width * 4; w++) {
      data[w + h * width * 4] = dataToCopy[w];
    }
  }

  const newImageData = new ImageData(new Uint8ClampedArray(data), width, height);

  {
    const canvas = document.getElementById('canvas2')
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.putImageData(newImageData, 0, 0);
  }

  buffer.unmap();


  return newImageData

}