////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////

#ifndef _MASKED_OCCLUSION_CULLING_H_
#error "Header should only be included from "MaskedOcclusionCulling.h"."
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common shared include file to hide compiler/os specific functions from the rest of the code.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER) // Windows: MSVC
#include <intrin.h>

#define MOC_FORCE_INLINE __forceinline

MOC_FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
{
	unsigned long idx;
	_BitScanForward(&idx, *mask);
	*mask &= *mask - 1;
	return idx;
}

#if defined(max) || defined(min)
#error NOMINMAX should be defined before include <windows.h>
#endif
#elif defined(__GNUC__)
#define MOC_FORCE_INLINE __attribute__((always_inline)) inline

MOC_FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
{
	unsigned long idx;
	idx = __builtin_ctzl(*mask);
	*mask &= *mask - 1;
	return idx;
}
#else
#error Unsupported compiler
#endif
