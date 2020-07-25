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

import {Negate, TensorInfo, UnaryInputs} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {multiplyConfig} from './Multiply';

export const negateConfig: KernelConfig = {
  kernelName: Negate,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend}) => {
    const {x} = inputs as UnaryInputs;
    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex(x, 'neg');

    const minusOne = cpuBackend.write(Float32Array.from([-1]), [], 'float32');
    const $minusOne:
        TensorInfo = {dataId: minusOne, shape: [], dtype: 'float32'};

    const result =
        multiplyConfig.kernelFunc({inputs: {a: $minusOne, b: x}, backend});

    cpuBackend.disposeData(minusOne);

    return result;
  }
};
