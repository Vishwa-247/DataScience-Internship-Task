import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  LineChart, Line, Legend
} from 'recharts'
import { fetchMetrics, fetchPredict } from '../api'

const fmt = (v) => {
  if (v == null || isNaN(v)) return '—'
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v.toFixed(1)
}

const COLORS = { arima: '#3b82f6', prophet: '#f59e0b', xgboost: '#ef4444', lstm: '#a855f7' }
const MEDALS = ['🥇', '🥈', '🥉', '']
const METRICS = ['rmse', 'mae', 'mape', 'smape']

export default function MetricsTab({ state }) {
  const [data, setData] = useState(null)
  const [selected, setSelected] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      fetchMetrics(state),
      fetchPredict(state).catch(() => null),
    ])
      .then(([m, p]) => {
        setData(m)
        setSelected(p?.selected_models || [])
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [state])

  if (loading) return <div className="loading">Loading metrics…</div>
  if (error) return <div className="error-box">{error}</div>
  if (!data?.models) return null

  const models = ['arima', 'prophet', 'xgboost', 'lstm'].filter(m => data.models[m])
  const rankedByRmse = [...models].sort((a, b) => data.models[a].mean_rmse - data.models[b].mean_rmse)

  const rmseData = rankedByRmse.map((m, i) => ({
    model: m.toUpperCase(),
    rmse: data.models[m].mean_rmse,
    fill: COLORS[m],
    rank: i,
  }))

  // Average across folds per metric
  const avgByModel = {}
  for (const m of models) {
    const folds = data.models[m].folds
    avgByModel[m] = {}
    for (const metric of METRICS) {
      const vals = folds.map(f => f[metric]).filter(v => isFinite(v) && !isNaN(v))
      avgByModel[m][metric] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN
    }
  }

  const bestPerMetric = {}
  for (const metric of METRICS) {
    bestPerMetric[metric] = models.reduce((best, m) =>
      avgByModel[m][metric] < (avgByModel[best]?.[metric] ?? Infinity) ? m : best, models[0])
  }

  // Fold-level RMSE trend for chart
  const foldTrendData = data.models[models[0]]?.folds.map((_, i) => {
    const row = { fold: `Fold ${i + 1}` }
    for (const m of models) row[m] = data.models[m].folds[i]?.rmse
    return row
  }) || []

  return (
    <>
      {/* KPI cards ranked */}
      <div className="kpi-grid">
        {rankedByRmse.map((m, i) => (
          <div key={m} className="kpi-card" style={{ borderTop: `3px solid ${COLORS[m]}` }}>
            <div className="kpi-label">
              {MEDALS[i]} {m.toUpperCase()}
              {selected.includes(m) &&
                <span className="badge-selected">IN ENSEMBLE</span>}
            </div>
            <div className="kpi-value" style={{ color: i === 0 ? '#22c55e' : COLORS[m] }}>
              {fmt(data.models[m].mean_rmse)}
            </div>
            <div className="kpi-sub">Mean CV RMSE</div>
          </div>
        ))}
      </div>

      {/* RMSE Bar chart */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2>Model RMSE Comparison — {state}</h2>
            <div className="subtitle">Mean RMSE across 5 walk-forward CV folds (lower = better)</div>
          </div>
          <div className="badge-row">
            {selected.map(m => (
              <span key={m} className="model-badge" style={{ background: COLORS[m] + '22', border: `1px solid ${COLORS[m]}`, color: COLORS[m] }}>
                {m.toUpperCase()} ✓ Selected
              </span>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rmseData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 13 }} />
            <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
              formatter={(v) => [fmt(v), 'Mean RMSE']}
            />
            <Bar dataKey="rmse" radius={[4, 4, 0, 0]} barSize={52}>
              {rmseData.map((entry, i) => <Cell key={i} fill={entry.fill} opacity={i === 0 ? 1 : 0.65} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Fold RMSE trend */}
      {foldTrendData.length > 1 && (
        <div className="card">
          <h2>RMSE per CV Fold</h2>
          <div className="subtitle">How each model performed fold-by-fold (expanding training window)</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={foldTrendData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="fold" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
                formatter={(v) => [fmt(v), 'RMSE']}
              />
              <Legend />
              {models.map(m => (
                <Line key={m} dataKey={m} stroke={COLORS[m]} strokeWidth={2} dot={{ r: 4 }}
                  name={m.toUpperCase()} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detailed table */}
      <div className="card">
        <h2>All Metrics — Averaged Across Folds</h2>
        <table className="metrics-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Rank</th>
              <th style={{ textAlign: 'left' }}>Model</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>MAPE (%)</th>
              <th>SMAPE (%)</th>
            </tr>
          </thead>
          <tbody>
            {rankedByRmse.map((m, i) => (
              <tr key={m} style={selected.includes(m) ? { background: 'rgba(59,130,246,0.06)' } : {}}>
                <td style={{ textAlign: 'left' }}>{MEDALS[i] || `#${i + 1}`}</td>
                <td style={{ textAlign: 'left' }}>
                  <span style={{ color: COLORS[m], fontWeight: 700 }}>{m.toUpperCase()}</span>
                  {selected.includes(m) &&
                    <span className="badge-selected" style={{ marginLeft: 8 }}>ENSEMBLE</span>}
                </td>
                {METRICS.map(metric => {
                  const v = avgByModel[m][metric]
                  const isBest = bestPerMetric[metric] === m
                  return (
                    <td key={metric} className={isBest ? 'best' : ''}>
                      {metric === 'mape' || metric === 'smape'
                        ? isNaN(v) ? '—' : v.toFixed(1) + '%'
                        : fmt(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
