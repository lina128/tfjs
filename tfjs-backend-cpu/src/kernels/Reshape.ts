/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {Reshape, ReshapeAttrs, ReshapeInputs} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x} = inputs as ReshapeInputs;
    const {shape} = attrs as {} as ReshapeAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const values = cpuBackend.data.get(x.dataId).values;

    const dataId = cpuBackend.write(values, shape, x.dtype);

    return {dataId, shape, dtype: x.dtype};
  }
};
