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

import {Complex, ComplexInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';
import {MathBackendCPU} from '../backend_cpu';
import {identityConfig} from './Identity';

export const complexConfig: KernelConfig = {
  kernelName: Complex,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend}) => {
    const {real, imag} = inputs as ComplexInputs;
    const cpuBackend = backend as MathBackendCPU;

    const dataId = cpuBackend.write(null, real.shape, 'complex64');
    const out = cpuBackend.data.get(dataId);

    const $real =
        identityConfig.kernelFunc({inputs: {x: real}, backend}) as TensorInfo;
    const $imag =
        identityConfig.kernelFunc({inputs: {x: imag}, backend}) as TensorInfo;

    out.complexTensors = {real: $real, imag: $imag};

    return {dataId, shape: real.shape, dtype: 'complex64'};
  }
};
