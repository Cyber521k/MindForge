// REST API client for MindForge sidecar
const BASE_URL = "http://localhost:7878";

function getErrorDetail(respBody: any, fallback: string): string {
  const detail = respBody?.detail;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item: any) => item?.msg)
      .filter((msg: any): msg is string => typeof msg === "string" && msg.length > 0);
    return messages.length > 0 ? messages.join("; ") : fallback;
  }

  if (typeof detail === "string") {
    return detail;
  }

  return respBody?.error || respBody?.message || fallback;
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = getErrorDetail(body, detail);
    } catch {
      // response had no JSON body — use statusText
    }
    throw new Error(`API ${path}: ${detail}`);
  }
  return res.json();
}

export async function apiPost<T = any>(path: string, body?: any): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const respBody = await res.json();
      detail = getErrorDetail(respBody, detail);
    } catch {
      // response had no JSON body — use statusText
    }
    throw new Error(`API ${path}: ${detail}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

export interface HardwareInfo {
  chip: string;
  model: string;
  memory_gb: number;
  usable_memory_gb: number;
  tier: string;
  [key: string]: any;
}

export interface ModelEntry {
  name: string;
  repo: string;
  size_gb: number;
  tier: string;
  can_run: boolean;
  type: "local" | "cloud";
  badge?: string;
  [key: string]: any;
}

export interface ModelListResponse {
  hardware?: HardwareInfo;
  local?: ModelEntry[];
  cloud?: ModelEntry[];
  [key: string]: any;
}

export interface ProbeResult {
  job_id: string;
  status: string;
  total?: number;
  correct?: number;
  incorrect?: number;
  dpo_entries?: number;
  output_path?: string;
  error?: string;
}

export interface TrainingEntry {
  id: number;
  prompt: string;
  chosen: string;
  rejected: string;
  format: string;
  subject: string;
  status: string;
  [key: string]: any;
}

export interface ResponseEntry {
  id: number;
  question: string;
  model_response: string;
  correct_answer_letter: string;
  model_answer_letter: string;
  is_correct: number;
  confidence: number;
  subject: string;
  model: string;
  [key: string]: any;
}

export interface JobInfo {
  status: "running" | "completed" | "failed";
  type: string;
  result: any;
  error: string | null;
  progress: number;
  started_at?: number;
}

export interface Stats {
  total_questions: number;
  training_pairs: number;
  subjects: number;
  training_runs: number;
  accuracy: Record<string, number>;
}

export interface TaxonomyData {
  categories: Record<string, string[]>;
  [key: string]: any;
}
