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
#include "CPUTModelDX11.h"
#include "CPUTMaterialDX11.h"
#include "CPUTRenderParamsDX.h"
#include "CPUTFrustum.h"
#include "CPUTTextureDX11.h"
#include "CPUTBufferDX11.h"

ID3D11Buffer *CPUTModelDX11::mpModelConstantBuffer = NULL;
UINT CPUTModelDX11::mFCullCount = 0;

// Return the mesh at the given index (cast to the GFX api version of CPUTMeshDX11)
//-----------------------------------------------------------------------------
CPUTMeshDX11 *CPUTModelDX11::GetMesh(const UINT index) const
{
    return (0 == mMeshCount || index > mMeshCount) ? NULL : (CPUTMeshDX11 *)mpMesh[index];
}

float3 gLightDir = float3(0.7f, -0.5f, -0.1f);

// Set the render state before drawing this object
//-----------------------------------------------------------------------------
void CPUTModelDX11::UpdateShaderConstants(CPUTRenderParameters &renderParams)
{
    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX *)&renderParams)->mpContext;
    D3D11_MAPPED_SUBRESOURCE mapInfo;
    pContext->Map(mpModelConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapInfo);
    {
        CPUTModelConstantBuffer *pCb = (CPUTModelConstantBuffer *)mapInfo.pData;

        // TODO: remove construction of XMM type
        DirectX::XMMATRIX world((float *)GetWorldMatrix());
        pCb->World = world;

        CPUTCamera *pCamera = renderParams.mpCamera;
        DirectX::XMVECTOR cameraPos = XMLoadFloat3(&DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f));
        if (pCamera)
        {
            DirectX::XMMATRIX view((float *)pCamera->GetViewMatrix());
            DirectX::XMMATRIX projection((float *)pCamera->GetProjectionMatrix());
            float *pCameraPos = (float *)&pCamera->GetPosition();
            cameraPos = XMLoadFloat3(&DirectX::XMFLOAT3(pCameraPos[0], pCameraPos[1], pCameraPos[2]));

            // Note: We compute viewProjection to a local to avoid reading from write-combined memory.
            // The constant buffer uses write-combined memory.  We read this matrix when computing WorldViewProjection.
            // It is very slow to read it directly from the constant buffer.
            DirectX::XMMATRIX viewProjection = view * projection;
            pCb->ViewProjection = viewProjection;
            pCb->WorldViewProjection = world * viewProjection;
            DirectX::XMVECTOR determinant = XMMatrixDeterminant(world);
            pCb->InverseWorld = XMMatrixInverse(&determinant, XMMatrixTranspose(world));
        }
        // TODO: Have the lights set their render states?

        DirectX::XMVECTOR lightDirection = DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(gLightDir.x, gLightDir.y, gLightDir.z));
        pCb->LightDirection = DirectX::XMVector3Normalize(lightDirection);
        pCb->EyePosition = cameraPos;
        float *bbCWS = (float *)&mBoundingBoxCenterWorldSpace;
        float *bbHWS = (float *)&mBoundingBoxHalfWorldSpace;
        float *bbCOS = (float *)&mBoundingBoxCenterObjectSpace;
        float *bbHOS = (float *)&mBoundingBoxHalfObjectSpace;
        pCb->BoundingBoxCenterWorldSpace = DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(bbCWS[0], bbCWS[1], bbCWS[2]));
        ;
        pCb->BoundingBoxHalfWorldSpace = DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(bbHWS[0], bbHWS[1], bbHWS[2]));
        ;
        pCb->BoundingBoxCenterObjectSpace = DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(bbCOS[0], bbCOS[1], bbCOS[2]));
        ;
        pCb->BoundingBoxHalfObjectSpace = DirectX::XMLoadFloat3(&DirectX::XMFLOAT3(bbHOS[0], bbHOS[1], bbHOS[2]));
        ;

        // Shadow camera
        DirectX::XMMATRIX shadowView, shadowProjection;
        CPUTCamera *pShadowCamera = gpSample->GetShadowCamera();
        if (pShadowCamera)
        {
            shadowView = DirectX::XMMATRIX((float *)pShadowCamera->GetViewMatrix());
            shadowProjection = DirectX::XMMATRIX((float *)pShadowCamera->GetProjectionMatrix());
            pCb->LightWorldViewProjection = world * shadowView * shadowProjection;
        }
    }
    pContext->Unmap(mpModelConstantBuffer, 0);
}

// Render - render this model (only)
//-----------------------------------------------------------------------------
void CPUTModelDX11::Render(CPUTRenderParameters &renderParams)
{
    CPUTRenderParametersDX *pParams = (CPUTRenderParametersDX *)&renderParams;
    CPUTCamera *pCamera = pParams->mpCamera;

#ifdef SUPPORT_DRAWING_BOUNDING_BOXES
    if (renderParams.mShowBoundingBoxes && (!pCamera || pCamera->mFrustum.IsVisible(mBoundingBoxCenterWorldSpace, mBoundingBoxHalfWorldSpace)))
    {
        DrawBoundingBox(renderParams);
    }
#endif
    if (!renderParams.mDrawModels)
    {
        return;
    }

    // Update the model's render states only once (and then iterate over materials)
    UpdateShaderConstants(renderParams);

    bool isVisible = true;
    isVisible = !pParams->mRenderOnlyVisibleModels || !pCamera || pCamera->mFrustum.IsVisible(mBoundingBoxCenterWorldSpace, mBoundingBoxHalfWorldSpace);
    if (isVisible)
    {
        // loop over all meshes in this model and draw them
        for (UINT ii = 0; ii < mMeshCount; ii++)
        {
            mpMaterial[ii]->SetRenderStates(renderParams);
            ((CPUTMeshDX11 *)mpMesh[ii])->Draw(renderParams, this);
        }
    }
    else
    {
        mFCullCount++;
    }
}

// Render - render this model (only)
//-----------------------------------------------------------------------------
void CPUTModelDX11::RenderShadow(CPUTRenderParameters &renderParams)
{
    CPUTRenderParametersDX *pParams = (CPUTRenderParametersDX *)&renderParams;
    CPUTCamera *pCamera = pParams->mpCamera;

#ifdef SUPPORT_DRAWING_BOUNDING_BOXES
    if (renderParams.mShowBoundingBoxes && (!pCamera || pCamera->mFrustum.IsVisible(mBoundingBoxCenterWorldSpace, mBoundingBoxHalfWorldSpace)))
    {
        DrawBoundingBox(renderParams);
    }
#endif
    if (!renderParams.mDrawModels)
    {
        return;
    }

    // TODO: add world-space bounding box to model so we don't need to do that work every frame
    if (!pParams->mRenderOnlyVisibleModels || !pCamera || pCamera->mFrustum.IsVisible(mBoundingBoxCenterWorldSpace, mBoundingBoxHalfWorldSpace))
    {
        // Update the model's render states only once (and then iterate over materials)
        UpdateShaderConstants(renderParams);

        CPUTMaterialDX11 *pMaterial = (CPUTMaterialDX11 *)(mpShadowCastMaterial);
        pMaterial->SetRenderStates(renderParams);

        // loop over all meshes in this model and draw them
        for (UINT ii = 0; ii < mMeshCount; ii++)
        {
            ((CPUTMeshDX11 *)mpMesh[ii])->DrawShadow(renderParams, this);
        }
    }
}

#ifdef SUPPORT_DRAWING_BOUNDING_BOXES
//-----------------------------------------------------------------------------
void CPUTModelDX11::DrawBoundingBox(CPUTRenderParameters &renderParams)
{
    CPUTRenderParametersDX *pParams = (CPUTRenderParametersDX *)&renderParams;

    UpdateShaderConstants(renderParams);
    CPUTMaterialDX11 *pMaterial = (CPUTMaterialDX11 *)mpBoundingBoxMaterial;

    pMaterial->SetRenderStates(renderParams);
    ((CPUTMeshDX11 *)mpBoundingBoxMesh)->Draw(renderParams, this);
}
#endif

//-----------------------------------------------------------------------------
void CPUTModelDX11::CreateModelConstantBuffer()
{
    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11 *)CPUTAssetLibrary::GetAssetLibrary();

    // Create the model constant buffer.
    HRESULT hr;
    D3D11_BUFFER_DESC bd = {0};
    bd.ByteWidth = sizeof(CPUTModelConstantBuffer);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = (CPUT_DX11::GetDevice())->CreateBuffer(&bd, NULL, &mpModelConstantBuffer);
    ASSERT(!FAILED(hr), _L("Error creating constant buffer."));
    CPUTSetDebugName(mpModelConstantBuffer, _L("Model Constant buffer"));
    cString name = _L("$cbPerModelValues");
    CPUTBufferDX11 *pBuffer = new CPUTBufferDX11(name, mpModelConstantBuffer);
    pAssetLibrary->AddConstantBuffer(name, pBuffer);
    pBuffer->Release(); // We're done with it.  We added it to the library.  Release our reference.
}

//-----------------------------------------------------------------------------
CPUTResult CPUTModelDX11::LoadModel(CPUTConfigBlock *pBlock, int *pParentID, CPUTModel *pMasterModel)
{
    CPUTResult result = CPUT_SUCCESS;
    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11 *)CPUTAssetLibrary::GetAssetLibrary();

    cString modelSuffix = ptoc(this);

    // set the model's name
    mName = pBlock->GetValueByName(_L("name"))->ValueAsString();
    mName = mName + _L(".mdl");

    // resolve the full path name
    cString modelLocation;
    cString resolvedPathAndFile;
    modelLocation = ((CPUTAssetLibraryDX11 *)CPUTAssetLibrary::GetAssetLibrary())->GetModelDirectory();
    modelLocation = modelLocation + mName;
    CPUTOSServices::GetOSServices()->ResolveAbsolutePathAndFilename(modelLocation, &resolvedPathAndFile);

    // Get the parent ID.  Note: the caller will use this to set the parent.
    *pParentID = pBlock->GetValueByName(_L("parent"))->ValueAsInt();

    LoadParentMatrixFromParameterBlock(pBlock);

    // Get the bounding box information
    float3 center(0.0f), half(0.0f);
    pBlock->GetValueByName(_L("BoundingBoxCenter"))->ValueAsFloatArray(center.f, 3);
    pBlock->GetValueByName(_L("BoundingBoxHalf"))->ValueAsFloatArray(half.f, 3);
    mBoundingBoxCenterObjectSpace = center;
    mBoundingBoxHalfObjectSpace = half;

    mMeshCount = pBlock->GetValueByName(_L("meshcount"))->ValueAsInt();
    mpMesh = new CPUTMesh *[mMeshCount];
    mpMaterial = new CPUTMaterial *[mMeshCount];
    memset(mpMaterial, 0, mMeshCount * sizeof(CPUTMaterial *));

    cString materialName;
    char pNumber[4];
    cString materialValueName;

    CPUTModelDX11 *pMasterModelDX = (CPUTModelDX11 *)pMasterModel;

    for (UINT ii = 0; ii < mMeshCount; ii++)
    {
        if (pMasterModelDX)
        {
            // Reference the master model's mesh.  Don't create a new one.
            mpMesh[ii] = pMasterModelDX->mpMesh[ii];
            mpMesh[ii]->AddRef();
        }
        else
        {
            mpMesh[ii] = new CPUTMeshDX11();
        }
    }
    if (!pMasterModelDX)
    {
        // Not a clone/instance.  So, load the model's binary payload (i.e., vertex and index buffers)
        // TODO: Change to use GetModel()
        result = LoadModelPayload(resolvedPathAndFile);
        ASSERT(CPUTSUCCESS(result), _L("Failed loading model"));
    }

#if 0
    cString assetSetDirectoryName = pAssetLibrary->GetAssetSetDirectoryName();
    cString modelDirectory        = pAssetLibrary->GetModelDirectory();
    cString materialDirectory     = pAssetLibrary->GetMaterialDirectory();
    cString textureDirectory      = pAssetLibrary->GetTextureDirectory();
    cString shaderDirectory       = pAssetLibrary->GetShaderDirectory();
    cString fontDirectory         = pAssetLibrary->GetFontDirectory();
    cString up2MediaDirName       = assetSetDirectoryName + _L("..\\..\\");
    pAssetLibrary->SetMediaDirectoryName( up2MediaDirName );
    mpShadowCastMaterial = pAssetLibrary->GetMaterial( _L("shadowCast"), false, this, -2 ); // -2 signifies shadow material.  TODO: find a clearer way (e.g., enum?)
    pAssetLibrary->SetAssetSetDirectoryName( assetSetDirectoryName );
    pAssetLibrary->SetModelDirectoryName( modelDirectory ); 
    pAssetLibrary->SetMaterialDirectoryName( materialDirectory );
    pAssetLibrary->SetTextureDirectoryName( textureDirectory );
    pAssetLibrary->SetShaderDirectoryName( shaderDirectory );
    pAssetLibrary->SetFontDirectoryName( fontDirectory );
#endif
    for (UINT ii = 0; ii < mMeshCount; ii++)
    {
        // get the right material number ('material0', 'material1', 'material2', etc)
        materialValueName = _L("material");
        _itoa_s(ii, pNumber, 4, 10);
        materialValueName.append(s2ws(pNumber));
        materialName = pBlock->GetValueByName(materialValueName)->ValueAsString();

        // Get/load material for this mesh
        CPUTMaterialDX11 *pMaterial = (CPUTMaterialDX11 *)pAssetLibrary->GetMaterial(materialName, false, this, ii);
        ASSERT(pMaterial, _L("Couldn't find material."));

        // set the material on this mesh
        // TODO: Model owns the materials.  That allows different models to share meshes (aka instancing) that have different materials
        SetMaterial(ii, pMaterial);

        // Release the extra refcount we're holding from the GetMaterial operation earlier
        // now the asset library, and this model have the only refcounts on that material
        pMaterial->Release();

        // Create two ID3D11InputLayout objects, one for each material.
        mpMesh[ii]->BindVertexShaderLayout(mpMaterial[ii], mpShadowCastMaterial);
        // mpShadowCastMaterial->Release()
    }
    return result;
}

// Set the material associated with this mesh and create/re-use a
//-----------------------------------------------------------------------------
void CPUTModelDX11::SetMaterial(UINT ii, CPUTMaterial *pMaterial)
{
    CPUTModel::SetMaterial(ii, pMaterial);

    // Can't bind the layout if we haven't loaded the mesh yet.
    CPUTMeshDX11 *pMesh = (CPUTMeshDX11 *)mpMesh[ii];
    D3D11_INPUT_ELEMENT_DESC *pDesc = pMesh->GetLayoutDescription();
    if (pDesc)
    {
        pMesh->BindVertexShaderLayout(pMaterial, mpMaterial[ii]);
    }
}

#ifdef SUPPORT_DRAWING_BOUNDING_BOXES
//-----------------------------------------------------------------------------
void CPUTModelDX11::CreateBoundingBoxMesh()
{
    CPUTResult result = CPUT_SUCCESS;
    if (!mpBoundingBoxMesh)
    {
        float3 pVertices[8] = {
            float3(1.0f, 1.0f, 1.0f),   // 0
            float3(1.0f, 1.0f, -1.0f),  // 1
            float3(-1.0f, 1.0f, 1.0f),  // 2
            float3(-1.0f, 1.0f, -1.0f), // 3
            float3(1.0f, -1.0f, 1.0f),  // 4
            float3(1.0f, -1.0f, -1.0f), // 5
            float3(-1.0f, -1.0f, 1.0f), // 6
            float3(-1.0f, -1.0f, -1.0f) // 7
        };
        USHORT pIndices[24] = {
            0, 1, 1, 3, 3, 2, 2, 0, // Top
            4, 5, 5, 7, 7, 6, 6, 4, // Bottom
            0, 4, 1, 5, 2, 6, 3, 7  // Verticals
        };
        CPUTVertexElementDesc pVertexElements[] = {
            {CPUT_VERTEX_ELEMENT_POSITON, tFLOAT, 12, 0},
        };

        mpBoundingBoxMesh = new CPUTMeshDX11();
        mpBoundingBoxMesh->SetMeshTopology(CPUT_TOPOLOGY_INDEXED_LINE_LIST);

        CPUTBufferInfo vertexElementInfo;
        vertexElementInfo.mpSemanticName = "POSITION";
        vertexElementInfo.mSemanticIndex = 0;
        vertexElementInfo.mElementType = CPUT_F32;
        vertexElementInfo.mElementComponentCount = 3;
        vertexElementInfo.mElementSizeInBytes = 12;
        vertexElementInfo.mElementCount = 8;
        vertexElementInfo.mOffset = 0;

        CPUTBufferInfo indexDataInfo;
        indexDataInfo.mElementType = CPUT_U16;
        indexDataInfo.mElementComponentCount = 1;
        indexDataInfo.mElementSizeInBytes = sizeof(UINT16);
        indexDataInfo.mElementCount = 24; // 12 lines, 2 verts each
        indexDataInfo.mOffset = 0;
        indexDataInfo.mSemanticIndex = 0;
        indexDataInfo.mpSemanticName = NULL;

        result = mpBoundingBoxMesh->CreateNativeResources(this, -1,
                                                          1, // vertexFormatDesc.mFormatDescriptorCount,
                                                          &vertexElementInfo,
                                                          pVertices, // (void*)vertexFormatDesc.mpVertices,
                                                          &indexDataInfo,
                                                          pIndices // &vertexFormatDesc.mpIndices[0]
        );
        ASSERT(CPUTSUCCESS(result), _L("Failed building bounding box mesh"));
    }
    if (!mpBoundingBoxMaterialMaster)
    {
        mpBoundingBoxMaterialMaster = CPUTAssetLibrary::GetAssetLibrary()->GetMaterial(_L("BoundingBox"), false, NULL, -3); // -1 == mesh independent.  -2 == shadow cast material.  -3 == bounding box.  TODO: how to formalize (enum?)
        mpBoundingBoxMesh->BindVertexShaderLayout(mpBoundingBoxMaterialMaster, NULL);
    }
    mpBoundingBoxMaterial = mpBoundingBoxMaterialMaster;
    mpBoundingBoxMaterial->AddRef();
}
#endif
