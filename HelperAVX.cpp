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

#include "HelperAVX.h"
#include "CPUTCamera.h"
#include <DirectXMath.h>

HelperAVX::HelperAVX()
{
}

HelperAVX::~HelperAVX()
{
}


void HelperAVX::MatrixMultiply(const __m128 *m1, const __m128 *m2, __m128 *result)
{
	DirectX::XMMATRIX tmp_m1;
	tmp_m1.r[0] = m1[0];
	tmp_m1.r[1] = m1[1];
	tmp_m1.r[2] = m1[2];
	tmp_m1.r[3] = m1[3];

	DirectX::XMMATRIX tmp_m2;
	tmp_m2.r[0] = m2[0];
	tmp_m2.r[1] = m2[1];
	tmp_m2.r[2] = m2[2];
	tmp_m2.r[3] = m2[3];

	DirectX::XMMATRIX tmp_result = DirectX::XMMatrixMultiply(tmp_m1, tmp_m2);
	result[0] = tmp_result.r[0];
	result[1] = tmp_result.r[1];
	result[2] = tmp_result.r[2];
	result[3] = tmp_result.r[3];
}
//
// void BoxTestSetupAVX::Init(const __m128 viewMatrix[4], const __m128 projMatrix[4], const float4x4 &viewportMatrix, CPUTCamera *pCamera, float occludeeSizeThreshold)
//{
//	__m128 viewPortMatrix[4];
//	viewPortMatrix[0] = _mm_loadu_ps((float*)&viewportMatrix.r0);
//	viewPortMatrix[1] = _mm_loadu_ps((float*)&viewportMatrix.r1);
//	viewPortMatrix[2] = _mm_loadu_ps((float*)&viewportMatrix.r2);
//	viewPortMatrix[3] = _mm_loadu_ps((float*)&viewportMatrix.r3);
//
//	MatrixMultiply(viewMatrix, projMatrix, mViewProjViewport);
//	MatrixMultiply(mViewProjViewport, viewPortMatrix, mViewProjViewport);
//
//	mpCamera = pCamera;
//
//	float fov = pCamera->GetFov();
//	float tanOfHalfFov = tanf(fov * 0.5f);
//	radiusThreshold = occludeeSizeThreshold * occludeeSizeThreshold * tanOfHalfFov;
//}