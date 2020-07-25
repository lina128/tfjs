/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {Cast, CastAttrs, CastInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {complexConfig} from './Complex';
import {identityConfig} from './Identity';
import {intConfig} from './Int';
import {notEqualConfig} from './NotEqual';
import {realConfig} from './Real';
import {zerosConfig} from './Zeros';

const cast_: KernelFunc = ({inputs, backend, attrs}) => {
  const {x} = inputs as CastInputs;
  const {dtype} = attrs as {} as CastAttrs;
  const cpuBackend = backend as MathBackendCPU;

  if (dtype === 'complex64') {
    if (x.dtype === 'complex64') {
      return identityConfig.kernelFunc({inputs, backend, attrs});
    }

    const real =
        cast_({inputs, backend, attrs: {dtype: 'float32'}}) as TensorInfo;
    const imag =
        zerosConfig.kernelFunc(
            {inputs: {}, backend, attrs: {dtype: 'float32', shape: x.shape}}) as
        TensorInfo;

    const result = complexConfig.kernelFunc({inputs: {real, imag}, backend});

    cpuBackend.disposeData(real.dataId);
    cpuBackend.disposeData(imag.dataId);

    return result;
  }

  if (!util.hasEncodingLoss(x.dtype, dtype)) {
    // We don't change the underlying data, since we cast to higher
    // precision.
    const $x = {dataId: x.dataId, shape: x.shape, dtype};
    return identityConfig.kernelFunc({inputs: {x: $x}, backend});
  }

  if (x.dtype === 'complex64') {
    const real =
        realConfig.kernelFunc({inputs: {input: x}, backend}) as TensorInfo;

    const result = cast_({inputs: {x: real}, backend, attrs});

    cpuBackend.disposeData(real.dataId);

    return result;
  }

  if (dtype === 'int32') {
    return intConfig.kernelFunc({inputs: {x}, backend});
  }

  if (dtype === 'bool') {
    const zeroId = cpuBackend.write(new Float32Array(1), [], x.dtype);

    const result = notEqualConfig.kernelFunc({
      inputs: {a: x, b: {dataId: zeroId, shape: [], dtype: x.dtype}},
      backend
    });

    cpuBackend.disposeData(zeroId);

    return result;
  }

  throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
};

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'cpu',
  kernelFunc: cast_
};
