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
#ifndef __CPUTMATERIALDX11_H__
#define __CPUTMATERIALDX11_H__

#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX 1
#include <sdkddkver.h>
#include <windows.h>
#include <d3d11.h>

#include "CPUTMaterial.h"
class CPUTPixelShaderDX11;
class CPUTComputeShaderDX11;
class CPUTVertexShaderDX11;
class CPUTGeometryShaderDX11;
class CPUTHullShaderDX11;
class CPUTDomainShaderDX11;
class CPUTModel;

class CPUTShaderParameters
{
public:
    UINT mTextureCount;
    cString *mpTextureParameterName;
    UINT *mpTextureParameterBindPoint;
    UINT mTextureParameterCount;

    // UINT                       mSamplerCount; // TODO: Why don't we need this? We should probably be rebinding samplers too
    cString *mpSamplerParameterName;
    UINT *mpSamplerParameterBindPoint;
    UINT mSamplerParameterCount;

    UINT mBufferCount;
    UINT mBufferParameterCount;
    cString *mpBufferParameterName;
    UINT *mpBufferParameterBindPoint;

    UINT mUAVCount;
    UINT mUAVParameterCount;
    cString *mpUAVParameterName;
    UINT *mpUAVParameterBindPoint;

    UINT mConstantBufferCount;
    UINT mConstantBufferParameterCount;
    cString *mpConstantBufferParameterName;
    UINT *mpConstantBufferParameterBindPoint;

    ID3D11ShaderResourceView *mppBindViews[CPUT_MATERIAL_MAX_SRV_SLOTS];
    ID3D11UnorderedAccessView *mppBindUAVs[CPUT_MATERIAL_MAX_UAV_SLOTS];
    ID3D11Buffer *mppBindConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];

    CPUTShaderParameters() : mTextureCount(0),
                             mTextureParameterCount(0),
                             mpTextureParameterName(NULL),
                             mpTextureParameterBindPoint(NULL),
                             mSamplerParameterCount(0),
                             mpSamplerParameterName(NULL),
                             mpSamplerParameterBindPoint(NULL),
                             mBufferCount(0),
                             mBufferParameterCount(0),
                             mpBufferParameterName(NULL),
                             mpBufferParameterBindPoint(NULL),
                             mUAVCount(0),
                             mUAVParameterCount(0),
                             mpUAVParameterName(NULL),
                             mpUAVParameterBindPoint(NULL),
                             mConstantBufferCount(0),
                             mConstantBufferParameterCount(0),
                             mpConstantBufferParameterName(NULL),
                             mpConstantBufferParameterBindPoint(NULL)
    {
        // initialize texture slot list to null
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++)
        {
            mppBindViews[ii] = NULL;
        }
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_UAV_SLOTS; ii++)
        {
            mppBindUAVs[ii] = NULL;
        }
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++)
        {
            mppBindConstantBuffers[ii] = NULL;
        }
    };
    ~CPUTShaderParameters()
    {
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++)
        {
            SAFE_RELEASE(mppBindViews[ii]);
        }
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_UAV_SLOTS; ii++)
        {
            SAFE_RELEASE(mppBindUAVs[ii]);
        }
        for (int ii = 0; ii < CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++)
        {
            SAFE_RELEASE(mppBindConstantBuffers[ii]);
        }
        SAFE_DELETE_ARRAY(mpTextureParameterName);
        SAFE_DELETE_ARRAY(mpTextureParameterBindPoint);
        SAFE_DELETE_ARRAY(mpSamplerParameterName);
        SAFE_DELETE_ARRAY(mpSamplerParameterBindPoint);
        SAFE_DELETE_ARRAY(mpBufferParameterName);
        SAFE_DELETE_ARRAY(mpBufferParameterBindPoint);
        SAFE_DELETE_ARRAY(mpUAVParameterName);
        SAFE_DELETE_ARRAY(mpUAVParameterBindPoint);
        SAFE_DELETE_ARRAY(mpConstantBufferParameterName);
        SAFE_DELETE_ARRAY(mpConstantBufferParameterBindPoint);
    }
    void CloneShaderParameters(CPUTShaderParameters *pShaderParameter);
};

static const int CPUT_NUM_SHADER_PARAMETER_LISTS = 7;
class CPUTMaterialDX11 : public CPUTMaterial
{
protected:
    static void *mpLastVertexShader;
    static void *mpLastPixelShader;
    static void *mpLastComputeShader;
    static void *mpLastGeometryShader;
    static void *mpLastHullShader;
    static void *mpLastDomainShader;

    static void *mpLastVertexShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastPixelShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastComputeShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastGeometryShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastHullShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastDomainShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];

    static void *mpLastVertexShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastPixelShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastComputeShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastGeometryShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastHullShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastDomainShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];

    static void *mpLastComputeShaderUAVs[CPUT_MATERIAL_MAX_UAV_SLOTS];

    static void *mpLastRenderStateBlock;

    // TODO: move texture to base class.  All APIs have textures.
    CPUTPixelShaderDX11 *mpPixelShader;
    CPUTComputeShaderDX11 *mpComputeShader; // TODO: Do Compute Shaders belong in material?
    CPUTVertexShaderDX11 *mpVertexShader;
    CPUTGeometryShaderDX11 *mpGeometryShader;
    CPUTHullShaderDX11 *mpHullShader;
    CPUTDomainShaderDX11 *mpDomainShader;
    int mRequiresPerModelPayload;

public:
    CPUTShaderParameters mPixelShaderParameters;
    CPUTShaderParameters mComputeShaderParameters;
    CPUTShaderParameters mVertexShaderParameters;
    CPUTShaderParameters mGeometryShaderParameters;
    CPUTShaderParameters mHullShaderParameters;
    CPUTShaderParameters mDomainShaderParameters;
    CPUTShaderParameters *mpShaderParametersList[CPUT_NUM_SHADER_PARAMETER_LISTS]; // Constructor initializes this as a list of pointers to the above shader parameters.

protected:
    ~CPUTMaterialDX11(); // Destructor is not public.  Must release instead of delete.

    void ReadShaderSamplersAndTextures(ID3DBlob *pBlob, CPUTShaderParameters *pShaderParameter);

    void BindTextures(CPUTShaderParameters &params, const CPUTModel *pModel = NULL, int meshIndex = -1);
    void BindBuffers(CPUTShaderParameters &params, const CPUTModel *pModel = NULL, int meshIndex = -1);
    void BindUAVs(CPUTShaderParameters &params, const CPUTModel *pModel = NULL, int meshIndex = -1);
    void BindConstantBuffers(CPUTShaderParameters &params, const CPUTModel *pModel = NULL, int meshIndex = -1);

public:
    CPUTMaterialDX11();

    CPUTResult LoadMaterial(const cString &fileName, const CPUTModel *pModel = NULL, int meshIndex = -1);
    void ReleaseTexturesAndBuffers();
    void RebindTexturesAndBuffers();
    CPUTVertexShaderDX11 *GetVertexShader() { return mpVertexShader; }
    CPUTPixelShaderDX11 *GetPixelShader() { return mpPixelShader; }
    CPUTGeometryShaderDX11 *GetGeometryShader() { return mpGeometryShader; }
    CPUTComputeShaderDX11 *GetComputeShader() { return mpComputeShader; }
    CPUTDomainShaderDX11 *GetDomainShader() { return mpDomainShader; }
    CPUTHullShaderDX11 *GetHullShader() { return mpHullShader; }

    // Tells material to set the current render state to match the properties, textures,
    //  shaders, state, etc that this material represents
    void SetRenderStates(CPUTRenderParameters &renderParams);
    bool MaterialRequiresPerModelPayload();
    CPUTMaterial *CloneMaterial(const cString &absolutePathAndFilename, const CPUTModel *pModel = NULL, int meshIndex = -1);
    static void ResetStateTracking()
    {
        mpLastVertexShader = (void *)-1;
        mpLastPixelShader = (void *)-1;
        mpLastComputeShader = (void *)-1;
        mpLastGeometryShader = (void *)-1;
        mpLastHullShader = (void *)-1;
        mpLastDomainShader = (void *)-1;
        mpLastRenderStateBlock = (void *)-1;
        for (UINT ii = 0; ii < CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++)
        {
            mpLastVertexShaderViews[ii] = (void *)-1;
            mpLastPixelShaderViews[ii] = (void *)-1;
            mpLastComputeShaderViews[ii] = (void *)-1;
            mpLastGeometryShaderViews[ii] = (void *)-1;
            mpLastHullShaderViews[ii] = (void *)-1;
            mpLastDomainShaderViews[ii] = (void *)-1;
        }
        for (UINT ii = 0; ii < CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++)
        {
            mpLastVertexShaderConstantBuffers[ii] = (void *)-1;
            mpLastPixelShaderConstantBuffers[ii] = (void *)-1;
            mpLastComputeShaderConstantBuffers[ii] = (void *)-1;
            mpLastGeometryShaderConstantBuffers[ii] = (void *)-1;
            mpLastHullShaderConstantBuffers[ii] = (void *)-1;
            mpLastDomainShaderConstantBuffers[ii] = (void *)-1;
        }
        for (UINT ii = 0; ii < CPUT_MATERIAL_MAX_UAV_SLOTS; ii++)
        {
            mpLastComputeShaderUAVs[ii] = (void *)-1;
        }
    }
};

#endif // #ifndef __CPUTMATERIALDX11_H__
