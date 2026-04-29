const BASE = '/api';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const fetchHealth    = ()      => get('/health');
export const fetchStates    = ()      => get('/states');
export const fetchPredict   = (state) => get(`/predict?state=${encodeURIComponent(state)}&horizon=8`);
export const fetchBreakdown = (state) => get(`/predict/breakdown?state=${encodeURIComponent(state)}`);
export const fetchMetrics   = (state) => get(`/metrics?state=${encodeURIComponent(state)}`);
export const fetchBacktest  = (state) => get(`/backtest?state=${encodeURIComponent(state)}`);
