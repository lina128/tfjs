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

import {Cast, CastAttrs, CastInputs, KernelConfig} from '@tensorflow/tfjs-core';

import {identityConfig} from './Identity';

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x} = inputs as CastInputs;
    const {dtype} = attrs as {} as CastAttrs;

    if (dtype === 'complex64') {
      if (x.dtype === 'complex64') {
        return identityConfig.kernelFunc({inputs, backend, attrs});
      }
      const zerosTensor = zeros(x.shape);
      const floatX = x.toFloat();
      const result = backend.complex(floatX, zerosTensor);
      zerosTensor.dispose();
      floatX.dispose();
      return result as T;
    }

    if (!util.hasEncodingLoss(x.dtype, dtype)) {
      // We don't change the underlying data, since we cast to higher
      // precision.
      return ENGINE.makeTensorFromDataId(x.dataId, x.shape, dtype) as T;
    }
    if (x.dtype === 'complex64') {
      const real = backend.real(x);
      const result = real.cast(dtype);
      real.dispose();
      return result as T;
    }
    if (dtype === 'int32') {
      return backend.int(x);
    } else if (dtype === 'bool') {
      const zero = scalar(0, x.dtype);
      const result = backend.notEqual(x, zero) as T;
      zero.dispose();
      return result;
    } else {
      throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
    }
  }
};
