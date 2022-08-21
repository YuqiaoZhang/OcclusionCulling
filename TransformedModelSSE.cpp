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
#include "TransformedModelSSE.h"

TransformedModelSSE::TransformedModelSSE()
	: mpCPUTModel(NULL),
	  mNumMeshes(0),
	  mWorldMatrix(NULL),
	  mNumVertices(0),
	  mNumTriangles(0),
	  mpMeshes(NULL)
{
	mInsideViewFrustum[0] = mInsideViewFrustum[1] = false;
	mTooSmall[0] = mTooSmall[1] = false;

	mpXformedPos[0] = mpXformedPos[1] = NULL;
	mWorldMatrix = (__m128 *)_aligned_malloc(sizeof(float) * 4 * 4, 16);
	mCumulativeMatrix[0] = (__m128 *)_aligned_malloc(sizeof(float) * 4 * 4, 16);
	mCumulativeMatrix[1] = (__m128 *)_aligned_malloc(sizeof(float) * 4 * 4, 16);
}

TransformedModelSSE::~TransformedModelSSE()
{
	SAFE_DELETE_ARRAY(mpMeshes);
	_aligned_free(mWorldMatrix);
	_aligned_free(mCumulativeMatrix[0]);
	_aligned_free(mCumulativeMatrix[1]);
}

//--------------------------------------------------------------------
// Create place holder for the transformed meshes for each model
//---------------------------------------------------------------------
void TransformedModelSSE::CreateTransformedMeshes(CPUTModelDX11 *pModel)
{
	mpCPUTModel = pModel;
	mNumMeshes = pModel->GetMeshCount();

	float *world = (float *)pModel->GetWorldMatrix();
	mWorldMatrix[0] = _mm_loadu_ps(world + 0);
	mWorldMatrix[1] = _mm_loadu_ps(world + 4);
	mWorldMatrix[2] = _mm_loadu_ps(world + 8);
	mWorldMatrix[3] = _mm_loadu_ps(world + 12);

	float3 center, half;
	pModel->GetBoundsObjectSpace(&center, &half);

	mBBCenterOS = center;
	mRadiusSq = half.lengthSq();
	mpMeshes = new TransformedMeshSSE[mNumMeshes];

	for (UINT i = 0; i < mNumMeshes; i++)
	{
		CPUTMeshDX11 *pMesh = (CPUTMeshDX11 *)pModel->GetMesh(i);
		ASSERT((pMesh != NULL), _L("pMesh is NULL"));

		mpMeshes[i].Initialize(pMesh);
		mNumVertices += mpMeshes[i].GetNumVertices();
		mNumTriangles += mpMeshes[i].GetNumTriangles();
	}
}

void TransformedModelSSE::TooSmall(const BoxTestSetupSSE &setup, UINT idx)
{
	if (mInsideViewFrustum[idx])
	{
		MatrixMultiply(mWorldMatrix, setup.mViewProjViewport, mCumulativeMatrix[idx]);

		mBBCenterW = mBBCenterOS.x * mCumulativeMatrix[idx][0].m128_f32[3] +
					 mBBCenterOS.y * mCumulativeMatrix[idx][1].m128_f32[3] +
					 mBBCenterOS.z * mCumulativeMatrix[idx][2].m128_f32[3] +
					 mCumulativeMatrix[idx][3].m128_f32[3];

		if (mBBCenterW > 1.0f)
		{
			mTooSmall[idx] = mRadiusSq < mBBCenterW * setup.radiusThreshold;
		}
		else
		{
			// BB center is behind the near clip plane, making screen-space radius meaningless.
			// Assume visible.  This should be a safe assumption, as the frustum test says the bbox is visible.
			mTooSmall[idx] = false;
		}
	}
}

//------------------------------------------------------------------
// Determine is the occluder model is inside view frustum
//------------------------------------------------------------------
void TransformedModelSSE::InsideViewFrustum(const BoxTestSetupSSE &setup, UINT idx)
{
	mpCPUTModel->GetBoundsWorldSpace(&mBBCenterWS, &mBBHalfWS);
	mInsideViewFrustum[idx] = setup.mpCamera->mFrustum.IsVisible(mBBCenterWS, mBBHalfWS);

	if (mInsideViewFrustum[idx])
	{
		MatrixMultiply(mWorldMatrix, setup.mViewProjViewport, mCumulativeMatrix[idx]);

		mBBCenterW = mBBCenterOS.x * mCumulativeMatrix[idx][0].m128_f32[3] +
					 mBBCenterOS.y * mCumulativeMatrix[idx][1].m128_f32[3] +
					 mBBCenterOS.z * mCumulativeMatrix[idx][2].m128_f32[3] +
					 mCumulativeMatrix[idx][3].m128_f32[3];

		if (mBBCenterW > 1.0f)
		{
			mTooSmall[idx] = mRadiusSq < mBBCenterW * setup.radiusThreshold;
		}
		else
		{
			// BB center is behind the near clip plane, making screen-space radius meaningless.
			// Assume visible.  This should be a safe assumption, as the frustum test says the bbox is visible.
			mTooSmall[idx] = false;
		}
	}
}

void TransformedModelSSE::TransformAndRasterizeTrianglesST(MaskedOcclusionCulling *moc, UINT idx)
{
	if (mInsideViewFrustum[idx] && !mTooSmall[idx])
	{
		for (UINT meshId = 0; meshId < mNumMeshes; meshId++)
			mpMeshes[meshId].TransformAndRasterizeTrianglesST(mCumulativeMatrix[idx], moc, idx);
	}
}

void TransformedModelSSE::TransformAndRasterizeTrianglesMT(MaskedOcclusionCulling* moc, UINT idx)
{
	if (mInsideViewFrustum[idx] && !mTooSmall[idx])
	{
		for (UINT meshId = 0; meshId < mNumMeshes; meshId++)
			mpMeshes[meshId].TransformAndRasterizeTrianglesMT(mCumulativeMatrix[idx], moc, idx);
	}
}
