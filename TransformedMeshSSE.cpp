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

#include "TransformedMeshSSE.h"
#include "DepthBufferRasterizerSSE.h"
#include "MaskedOcclusionCulling\MaskedOcclusionCulling.h"
#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#include <tbb/parallel_for.h>

TransformedMeshSSE::TransformedMeshSSE()
	: mNumVertices(0),
	  mNumIndices(0),
	  mNumTriangles(0),
	  mpVertices(NULL),
	  mpIndices(NULL)
{
	mpXformedPos[0] = mpXformedPos[1] = NULL;
}

TransformedMeshSSE::~TransformedMeshSSE()
{
}

void TransformedMeshSSE::Initialize(CPUTMeshDX11 *pMesh)
{
	mNumVertices = pMesh->GetDepthVertexCount();
	mNumIndices = pMesh->GetIndexCount();
	mNumTriangles = pMesh->GetTriangleCount();
	mpVertices = pMesh->GetDepthVertices();
	mpIndices = pMesh->GetDepthIndices();

	mIndicesTheOtherWayAround.clear();
}

// flip the winding <MaskedOcclusionCulling should be upgraded so it accepts both>
void TransformedMeshSSE::UpdateReversedWindingIndices()
{
	const int numTriangles = GetNumTriangles();
	mIndicesTheOtherWayAround.resize(numTriangles * 3);
	for (int i = 0; i < numTriangles; i++)
	{
		mIndicesTheOtherWayAround[i * 3 + 0] = mpIndices[i * 3 + 2];
		mIndicesTheOtherWayAround[i * 3 + 1] = mpIndices[i * 3 + 1];
		mIndicesTheOtherWayAround[i * 3 + 2] = mpIndices[i * 3 + 0];
	}
}

inline __m128d &operator+=(__m128d &v1, const __m128d &v2)
{
	return (v1 = _mm_add_pd(v1, v2));
}

inline __m128d &operator-=(__m128d &v1, const __m128d &v2)
{
	return (v1 = _mm_sub_pd(v1, v2));
}

inline __m128d &operator*=(__m128d &v1, const __m128d &v2)
{
	return (v1 = _mm_mul_pd(v1, v2));
}

inline __m128d &operator/=(__m128d &v1, const __m128d &v2)
{
	return (v1 = _mm_div_pd(v1, v2));
}

inline __m128d operator+(const __m128d &v1, const __m128d &v2)
{
	return _mm_add_pd(v1, v2);
}

inline __m128d operator-(const __m128d &v1, const __m128d &v2)
{
	return _mm_sub_pd(v1, v2);
}

inline __m128d operator*(const __m128d &v1, const __m128d &v2)
{
	return _mm_mul_pd(v1, v2);
}

inline __m128d operator/(const __m128d &v1, const __m128d &v2)
{
	return _mm_div_pd(v1, v2);
}

inline __m128 &operator+=(__m128 &v1, const __m128 &v2)
{
	return (v1 = _mm_add_ps(v1, v2));
}

inline __m128 &operator-=(__m128 &v1, const __m128 &v2)
{
	return (v1 = _mm_sub_ps(v1, v2));
}

inline __m128 &operator*=(__m128 &v1, const __m128 &v2)
{
	return (v1 = _mm_mul_ps(v1, v2));
}

inline __m128 &operator/=(__m128 &v1, const __m128 &v2)
{
	return (v1 = _mm_div_ps(v1, v2));
}

inline __m128 operator+(const __m128 &v1, const __m128 &v2)
{
	return _mm_add_ps(v1, v2);
}

inline __m128 operator-(const __m128 &v1, const __m128 &v2)
{
	return _mm_sub_ps(v1, v2);
}

inline __m128 operator*(const __m128 &v1, const __m128 &v2)
{
	return _mm_mul_ps(v1, v2);
}

inline __m128 operator/(const __m128 &v1, const __m128 &v2)
{
	return _mm_div_ps(v1, v2);
}

void TransformedMeshSSE::Gather(vFloat4 pOut[3], UINT triId, UINT numLanes, UINT idx)
{
	const UINT *pInd0 = &mpIndices[triId * 3];
	const UINT *pInd1 = pInd0 + (numLanes > 1 ? 3 : 0);
	const UINT *pInd2 = pInd0 + (numLanes > 2 ? 6 : 0);
	const UINT *pInd3 = pInd0 + (numLanes > 3 ? 9 : 0);

	for (UINT i = 0; i < 3; i++)
	{
		__m128 v0 = mpXformedPos[idx][pInd0[i]]; // x0 y0 z0 w0
		__m128 v1 = mpXformedPos[idx][pInd1[i]]; // x1 y1 z1 w1
		__m128 v2 = mpXformedPos[idx][pInd2[i]]; // x2 y2 z2 w2
		__m128 v3 = mpXformedPos[idx][pInd3[i]]; // x3 y3 z3 w3
		_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
		pOut[i].X = v0;
		pOut[i].Y = v1;
		pOut[i].Z = v2;
		pOut[i].W = v3;
	}
}

void TransformedMeshSSE::TransformAndRasterizeTrianglesST(__m128 *cumulativeMatrix, MaskedOcclusionCulling *moc, UINT idx)
{
	if (mIndicesTheOtherWayAround.size() != GetNumTriangles() * 3)
		UpdateReversedWindingIndices();
	assert(mIndicesTheOtherWayAround.size() == GetNumTriangles() * 3);

	// The 'modelToClipMatrix' should column major for post-multiplication(OGL) and row major for pre-multiplication(DX).
	moc->RenderTriangles((float *)mpVertices, mIndicesTheOtherWayAround.data(), GetNumTriangles(), (float *)cumulativeMatrix, MaskedOcclusionCulling::BACKFACE_CW, MaskedOcclusionCulling::CLIP_PLANE_ALL, MaskedOcclusionCulling::VertexLayout(12, 4, 8));
}

void TransformedMeshSSE::TransformAndRasterizeTrianglesMT(__m128 *cumulativeMatrix, MaskedOcclusionCulling *moc, UINT idx)
{
	if (mIndicesTheOtherWayAround.size() != GetNumTriangles() * 3)
		UpdateReversedWindingIndices();
	assert(mIndicesTheOtherWayAround.size() == GetNumTriangles() * 3);

	constexpr unsigned int const binsW = 4U;
	constexpr unsigned int const binsH = 4U;
	constexpr unsigned int const numBins = binsW * binsH;
	// assert(nNumBins >= nNumThreads);	// Having less bins than threads is a bad idea!

	MaskedOcclusionCulling::TriList triLists[numBins];

	// Compute worst case job size (we allocate memory for the worst case)
	constexpr unsigned int const TriSize = 9;		   // vec3 * 3
	constexpr unsigned int const MaxTrisPerBin = 8096; // Maximum number of triangles per job (bigger drawcalls are split), affects memory requirements
	assert(GetNumTriangles() <= MaxTrisPerBin);

	MaskedOcclusionCulling::pfnAlignedAlloc allocCallback = NULL;
	MaskedOcclusionCulling::pfnAlignedFree freeCallback = NULL;
	moc->GetAllocFreeCallback(allocCallback, freeCallback);

	float *trilistData = static_cast<float *>(allocCallback(16, sizeof(float) * TriSize * MaxTrisPerBin * numBins));

	// Setup trilist objects used for binning
	for (unsigned int binIdx = 0; binIdx < numBins; ++binIdx)
	{
		MaskedOcclusionCulling::TriList &tList = triLists[binIdx];
		tList.mNumTriangles = MaxTrisPerBin;
		tList.mTriIdx = 0U;
		tList.mPtr = trilistData + static_cast<uintptr_t>(TriSize) * static_cast<uintptr_t>(MaxTrisPerBin) * static_cast<uintptr_t>(binIdx);
	}

	unsigned int width;
	unsigned int height;
	moc->GetResolution(width, height);

	unsigned int binWidth;
	unsigned int binHeight;
	moc->ComputeBinWidthHeight(binsW, binsH, binWidth, binHeight);

	MaskedOcclusionCulling::ScissorRect scissors[numBins];
	for (unsigned int ty = 0; ty < binsH; ++ty)
	{
		for (unsigned int tx = 0; tx < binsW; ++tx)
		{
			unsigned int binIdx = tx + ty * binsW;

			// Adjust rects on final row / col to match resolution
			scissors[binIdx].mMinX = tx * binWidth;
			scissors[binIdx].mMaxX = tx + 1 == binsW ? width : (tx + 1) * binWidth;
			scissors[binIdx].mMinY = ty * binHeight;
			scissors[binIdx].mMaxY = ty + 1 == binsH ? height : (ty + 1) * binHeight;
		}
	}

	// The 'modelToClipMatrix' should column major for post-multiplication(OGL) and row major for pre-multiplication(DX).
	moc->BinTriangles((float *)mpVertices, mIndicesTheOtherWayAround.data(), GetNumTriangles(), triLists, binsW, binsH, (float *)cumulativeMatrix, MaskedOcclusionCulling::BACKFACE_CW, MaskedOcclusionCulling::CLIP_PLANE_ALL, MaskedOcclusionCulling::VertexLayout(12, 4, 8));

	// TODO: using "Continuation Passing Style"
	tbb::parallel_for(tbb::blocked_range<unsigned int>(0, numBins, 1),
					  [moc, &triLists, &scissors](tbb::blocked_range<unsigned int> const &r) -> void
					  {
						  for (unsigned int binIdx = r.begin(); binIdx < r.end(); ++binIdx)
						  {
							  // in my opinion, "scissor test" is expected to be efficient
							  moc->RenderTrilist(triLists[binIdx], &scissors[binIdx]);
						  }
					  }

	);

	freeCallback(trilistData);
}