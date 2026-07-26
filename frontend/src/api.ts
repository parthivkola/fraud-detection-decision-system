export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
}

export interface ModelVersion {
  id: number;
  version_tag: string;
  description: string | null;
  file_path: string;
  scaler_path: string;
  metadata_path: string;
  is_active: boolean;
  ab_weight: number;
  created_at: string;
}

export interface PredictionRow {
  row_index: number;
  fraud_probability: number;
  is_fraud: boolean;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  decision: 'approve' | 'review' | 'block';
}

export interface PredictSummary {
  total_transactions: number;
  flagged_fraud: number;
  flagged_legitimate: number;
  fraud_rate: number;
  threshold_used: number;
  model_version: string | null;
}

export interface PredictResponse {
  batch_id: number;
  summary: PredictSummary;
  predictions: PredictionRow[];
}

export interface MetricsResponse {
  total_predictions: number;
  total_batches: number;
  flagged_fraud: number;
  flagged_legitimate: number;
  fraud_flag_rate: number;
  active_model_versions: string[];
  model_precision: number;
  model_recall: number;
  model_f1: number;
  model_accuracy: number;
  model_roc_auc: number;
  threshold: number;
  uptime_seconds: number;
  risk_distribution: Record<string, number>;
}

const BASE_URL = '/api/v1';

function getHeaders(token?: string | null): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  const t = token || localStorage.getItem('token');
  if (t) {
    headers['Authorization'] = `Bearer ${t}`;
  }
  return headers;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let err = 'An error occurred';
    try {
      const data = await res.json();
      err = data.detail || err;
    } catch {
      err = res.statusText;
    }
    throw new Error(err);
  }
  return res.json();
}

export const api = {
  async register(username: string, email: string, password: string): Promise<User> {
    const res = await fetch(`${BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    return handleResponse<User>(res);
  },

  async login(username: string, password: string): Promise<{ access_token: string }> {
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await handleResponse<{ access_token: string }>(res);
    localStorage.setItem('token', data.access_token);
    return data;
  },

  async getMe(): Promise<User> {
    const res = await fetch(`${BASE_URL}/auth/me`, {
      headers: getHeaders(),
    });
    return handleResponse<User>(res);
  },

  async getModels(): Promise<ModelVersion[]> {
    const res = await fetch(`${BASE_URL}/models`, {
      headers: getHeaders(),
    });
    return handleResponse<ModelVersion[]>(res);
  },

  async setModelActive(id: number, active: boolean): Promise<ModelVersion> {
    const endpoint = active ? 'activate' : 'deactivate';
    const res = await fetch(`${BASE_URL}/models/${id}/${endpoint}`, {
      method: 'PATCH',
      headers: getHeaders(),
    });
    return handleResponse<ModelVersion>(res);
  },

  async setModelWeight(id: number, ab_weight: number): Promise<ModelVersion> {
    const res = await fetch(`${BASE_URL}/models/${id}/weight`, {
      method: 'PATCH',
      headers: getHeaders(),
      body: JSON.stringify({ ab_weight }),
    });
    return handleResponse<ModelVersion>(res);
  },

  async getMetrics(modelTag?: string): Promise<MetricsResponse> {
    const url = modelTag ? `${BASE_URL}/metrics?model=${encodeURIComponent(modelTag)}` : `${BASE_URL}/metrics`;
    const res = await fetch(url, {
      headers: getHeaders(),
    });
    return handleResponse<MetricsResponse>(res);
  },

  async predict(file: File, modelTag?: string): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    let url = `${BASE_URL}/fraud/predict`;
    if (modelTag) {
      url += `?model_tag=${encodeURIComponent(modelTag)}`;
    }

    const headers: HeadersInit = {};
    const t = localStorage.getItem('token');
    if (t) {
      headers['Authorization'] = `Bearer ${t}`;
    }

    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
    });
    return handleResponse<PredictResponse>(res);
  },

  getSampleCsvUrl(): string {
    return `${BASE_URL}/sample-csv`;
  },
};
