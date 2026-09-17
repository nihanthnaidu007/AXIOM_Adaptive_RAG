import { API_BASE_URL } from '../config';

/**
 * API-key auth for the frontend. The backend fail-closes: every protected
 * /api endpoint returns 401 without a valid X-API-Key (503 when the server
 * has no key configured). The key is supplied at build time via
 * REACT_APP_API_KEY; left unset, headers carry no key and protected calls
 * surface the server's 401/503 error envelope in the UI.
 */
export function authHeaders(extra = {}) {
  const apiKey = process.env.REACT_APP_API_KEY || '';
  return apiKey ? { 'X-API-Key': apiKey, ...extra } : { ...extra };
}

/**
 * JSON fetch against the API with auth headers attached and error envelopes
 * surfaced. Throws an Error with `status` and the server's sanitized
 * `detail.error` message when the response is not ok.
 */
export async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}/api${path}`, {
    ...options,
    headers: authHeaders({ 'Content-Type': 'application/json', ...(options.headers || {}) }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload && payload.detail;
    const message = (detail && detail.error) || payload?.error || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}
