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

import {KernelConfig, KernelFunc, TensorInfo, util, Zeros, ZerosAttrs} from '@tensorflow/tfjs-core';
import {MathBackendCPU} from '../backend_cpu';
import {complexConfig} from './Complex';

const zeros_: KernelFunc = ({backend, attrs}) => {
  const {dtype, shape} = attrs as ZerosAttrs;
  const cpuBackend = backend as MathBackendCPU;

  if (dtype === 'complex64') {
    const real =
        zeros_({inputs: {}, backend, attrs: {dtype: 'float32', shape}}) as
        TensorInfo;
    const imag =
        zeros_({inputs: {}, backend, attrs: {dtype: 'float32', shape}}) as
        TensorInfo;

    const result =
        complexConfig.kernelFunc({inputs: {real, imag}, backend: cpuBackend});

    cpuBackend.disposeData(real.dataId);
    cpuBackend.disposeData(imag.dataId);

    return result;
  }

  const values = util.makeZerosTypedArray(util.sizeFromShape(shape), dtype);

  const dataId = cpuBackend.write(values, shape, dtype);

  return {dataId, shape, dtype};
};

const zeros = kernel({zeros_});

export const zerosConfig: KernelConfig = {
  kernelName: Zeros,
  backendName: 'cpu',
  kernelFunc: zeros_
};
