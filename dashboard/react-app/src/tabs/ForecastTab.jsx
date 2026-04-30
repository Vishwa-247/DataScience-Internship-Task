import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts'
import { fetchPredict, fetchBreakdown } from '../api'

const fmt = (v) => {
  if (v == null || isNaN(v)) return '—'
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v.toFixed(0)
}

const MODEL_COLORS = { arima: '#3b82f6', prophet: '#f59e0b', xgboost: '#ef4444', lstm: '#a855f7' }

function KpiCard({ label, value, sub, color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={color ? { color } : {}}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

export default function ForecastTab({ state }) {
  const [data, setData] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setBreakdown(null)
    Promise.all([fetchPredict(state), fetchBreakdown(state).catch(() => null)])
      .then(([pred, bk]) => { setData(pred); setBreakdown(bk) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [state])

  if (loading) return <div className="loading">Loading forecast…</div>
  if (error) return <div className="error-box">{error}</div>
  if (!data) return null

  const f = data.forecast
  const w1 = f[0], w4 = f[3], w8 = f[7]
  const ciSpread8 = w8 ? w8.yhat_upper - w8.yhat_lower : 0

  const chartData = f.map((p, i) => ({
    date: p.date.slice(5),
    yhat: p.yhat,
    ci: [p.yhat_lower, p.yhat_upper],
    week: i + 1,
  }))

  const models = breakdown ? Object.keys(breakdown.models) : []

  return (
    <>
      {/* KPI row */}
      <div className="kpi-grid">
        <KpiCard label="Week 1 Forecast" value={fmt(w1?.yhat)} sub={`CI: ${fmt(w1?.yhat_lower)} – ${fmt(w1?.yhat_upper)}`} color="#60a5fa" />
        <KpiCard label="Week 4 Forecast" value={fmt(w4?.yhat)} sub={`CI: ${fmt(w4?.yhat_lower)} – ${fmt(w4?.yhat_upper)}`} />
        <KpiCard label="Week 8 Forecast" value={fmt(w8?.yhat)} sub={`CI: ${fmt(w8?.yhat_lower)} – ${fmt(w8?.yhat_upper)}`} />
        <KpiCard label="Week 8 CI Width" value={fmt(ciSpread8)} sub="Widens with horizon (√h)" color="#a855f7" />
      </div>

      {/* Main chart */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2>8-Week Ensemble Forecast — {state}</h2>
            <div className="subtitle">Confidence interval widens with horizon: width = 1.96 × CV-RMSE × √h</div>
          </div>
          <div className="badge-row">
            {Object.entries(data.ensemble_weights).map(([m, w]) => (
              <span key={m} className="model-badge" style={{ background: MODEL_COLORS[m] + '22', border: `1px solid ${MODEL_COLORS[m]}`, color: MODEL_COLORS[m] }}>
                {m.toUpperCase()} {(w * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
              labelStyle={{ color: '#e2e8f0' }}
              formatter={(v, name) => [Array.isArray(v) ? `${fmt(v[0])} – ${fmt(v[1])}` : fmt(v), name]}
            />
            <Legend />
            <Area dataKey="ci" stroke="none" fill="#3b82f6" fillOpacity={0.12} name="95% CI" />
            <Line dataKey="yhat" stroke="#3b82f6" strokeWidth={2.5} dot={{ fill: '#3b82f6', r: 5 }} name="Ensemble" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Per-model breakdown table */}
      {breakdown && models.length > 0 && (
        <div className="card">
          <h2>Per-Model Breakdown</h2>
          <div className="subtitle">Individual model forecasts vs ensemble — {state}</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Week</th>
                  {models.map(m => (
                    <th key={m} style={{ color: MODEL_COLORS[m] }}>{m.toUpperCase()}</th>
                  ))}
                  <th style={{ color: '#60a5fa' }}>ENSEMBLE</th>
                </tr>
              </thead>
              <tbody>
                {f.map((fp, i) => (
                  <tr key={fp.date}>
                    <td style={{ textAlign: 'left', color: '#94a3b8' }}>W{i + 1} {fp.date.slice(5)}</td>
                    {models.map(m => (
                      <td key={m}>{fmt(breakdown.models[m]?.[i]?.yhat)}</td>
                    ))}
                    <td style={{ color: '#60a5fa', fontWeight: 600 }}>{fmt(fp.yhat)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
