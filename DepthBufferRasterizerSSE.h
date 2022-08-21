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
#ifndef DEPTHBUFFERRASTERIZERSSE_H
#define DEPTHBUFFERRASTERIZERSSE_H

#include "DepthBufferRasterizer.h"
#include "TransformedModelSSE.h"
#include "HelperSSE.h"

union XY
{
	struct
	{
		short x, y;
	};
	unsigned int xy;
};

struct BinTriangle
{
	XY vert[3];
	float Z[3]; // Plane equation, not just values at the three verts
};

#endif // DEPTHBUFFERRASTERIZERSSE_H