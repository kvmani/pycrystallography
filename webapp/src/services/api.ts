import axios from 'axios';

import {
  GenerateResponse,
  PowderPatternResponse,
  StructureModel,
  StructureResponse,
  TemPatternResponse,
  TemSettings,
  UiConfig,
  XrdSettings
} from '../types/structure';

const baseURL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';

const client = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export async function fetchUiConfig(): Promise<UiConfig> {
  const response = await client.get<UiConfig>('/ui/config');
  return response.data;
}

export async function uploadCif(file: File): Promise<StructureResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await client.post<StructureResponse>('/structures/from-cif', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
}

export async function generateDiffraction(
  structure: StructureModel,
  xrd: XrdSettings,
  tem: TemSettings
): Promise<GenerateResponse> {
  const response = await client.post<GenerateResponse>('/diffraction/generate', {
    structure,
    xrd,
    tem
  });
  return response.data;
}

export async function computePowderPattern(
  structure: StructureModel,
  settings: XrdSettings
): Promise<PowderPatternResponse> {
  const response = await client.post<PowderPatternResponse>('/diffraction/xrd', {
    structure,
    settings
  });
  return response.data;
}

export async function computeTemPattern(
  structure: StructureModel,
  settings: TemSettings
): Promise<TemPatternResponse> {
  const response = await client.post<TemPatternResponse>('/diffraction/tem', {
    structure,
    settings
  });
  return response.data;
}
