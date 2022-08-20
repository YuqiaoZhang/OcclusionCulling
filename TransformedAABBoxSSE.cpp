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

#include "TransformedAABBoxSSE.h"

static const UINT sBBIndexList[AABB_INDICES] =
	{
		// index for top
		1,
		3,
		2,
		0,
		3,
		1,

		// index for bottom
		5,
		7,
		4,
		6,
		7,
		5,

		// index for left
		1,
		7,
		6,
		2,
		7,
		1,

		// index for right
		3,
		5,
		4,
		0,
		5,
		3,

		// index for back
		2,
		4,
		7,
		3,
		4,
		2,

		// index for front
		0,
		6,
		5,
		1,
		6,
		0,
};

// 0 = use min corner, 1 = use max corner
static const UINT sBBxInd[AABB_VERTICES] = {1, 0, 0, 1, 1, 1, 0, 0};
static const UINT sBByInd[AABB_VERTICES] = {1, 1, 1, 1, 0, 0, 0, 0};
static const UINT sBBzInd[AABB_VERTICES] = {1, 1, 0, 0, 0, 1, 1, 0};

//--------------------------------------------------------------------------
// Get the bounding box center and half vector
// Create the vertex and index list for the triangles that make up the bounding box
//--------------------------------------------------------------------------
void TransformedAABBoxSSE::CreateAABBVertexIndexList(CPUTModelDX11 *pModel)
{
	mWorldMatrix = *pModel->GetWorldMatrix();

	pModel->GetBoundsObjectSpace(&mBBCenter, &mBBHalf);
	mRadiusSq = mBBHalf.lengthSq();
	pModel->GetBoundsWorldSpace(&mBBCenterWS, &mBBHalfWS);
}

//----------------------------------------------------------------
// Determine is model is inside view frustum
//----------------------------------------------------------------
bool TransformedAABBoxSSE::IsInsideViewFrustum(CPUTCamera *pCamera)
{
	return pCamera->mFrustum.IsVisible(mBBCenterWS, mBBHalfWS);
}

//----------------------------------------------------------------------------
// Determine if the occluddee size is too small and if so avoid drawing it
//----------------------------------------------------------------------------
bool TransformedAABBoxSSE::IsTooSmall(const BoxTestSetupSSE &setup, __m128 cumulativeMatrix[4])
{
	__m128 worldMatrix[4];
	worldMatrix[0] = _mm_loadu_ps(mWorldMatrix.r0.f);
	worldMatrix[1] = _mm_loadu_ps(mWorldMatrix.r1.f);
	worldMatrix[2] = _mm_loadu_ps(mWorldMatrix.r2.f);
	worldMatrix[3] = _mm_loadu_ps(mWorldMatrix.r3.f);
	MatrixMultiply(worldMatrix, setup.mViewProjViewport, cumulativeMatrix);

	float w = mBBCenter.x * cumulativeMatrix[0].m128_f32[3] +
			  mBBCenter.y * cumulativeMatrix[1].m128_f32[3] +
			  mBBCenter.z * cumulativeMatrix[2].m128_f32[3] +
			  cumulativeMatrix[3].m128_f32[3];

	if (w > 1.0f)
	{
		return mRadiusSq < w * setup.radiusThreshold;
	}
	return false;
}

void TransformedAABBoxSSE::Gather(vFloat4 pOut[3], UINT triId, const __m128 xformedPos[], UINT idx)
{
	for (int i = 0; i < 3; i++)
	{
		UINT ind0 = sBBIndexList[triId * 3 + i + 0];
		UINT ind1 = sBBIndexList[triId * 3 + i + 3];
		UINT ind2 = sBBIndexList[triId * 3 + i + 6];
		UINT ind3 = sBBIndexList[triId * 3 + i + 9];

		__m128 v0 = xformedPos[ind0]; // x0 y0 z0 w0
		__m128 v1 = xformedPos[ind1]; // x1 y1 z1 w1
		__m128 v2 = xformedPos[ind2]; // x2 y2 z2 w2
		__m128 v3 = xformedPos[ind3]; // x3 y3 z3 w3
		_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
		pOut[i].X = v0;
		pOut[i].Y = v1;
		pOut[i].Z = v2;
		pOut[i].W = v3;
	}
}
