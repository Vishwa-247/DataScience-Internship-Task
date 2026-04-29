import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend, Cell
} from 'recharts'
import { fetchMetrics } from '../api'

const fmt = (v) => {
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v?.toFixed(1) ?? ''
}

const COLORS = {
  arima: '#3b82f6',
  prophet: '#f59e0b',
  xgboost: '#ef4444',
  lstm: '#a855f7',
}

const METRICS = ['rmse', 'mae', 'mape', 'smape']

export default function MetricsTab({ state }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchMetrics(state)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [state])

  if (loading) return <div className="loading">Loading metrics…</div>
  if (error) return <div className="error-box">{error}</div>
  if (!data?.models) return null

  const models = Object.keys(data.models)

  // Bar chart data: one group per model, value = mean RMSE
  const rmseData = models.map(m => ({
    model: m.toUpperCase(),
    rmse: data.models[m].mean_rmse,
    fill: COLORS[m] || '#64748b',
  }))

  // Fold-level table: for each metric, average across folds
  const avgByModel = {}
  for (const m of models) {
    const folds = data.models[m].folds
    avgByModel[m] = {}
    for (const metric of METRICS) {
      const vals = folds.map(f => f[metric]).filter(v => v !== Infinity && !isNaN(v))
      avgByModel[m][metric] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN
    }
  }

  const bestPerMetric = {}
  for (const metric of METRICS) {
    let best = Infinity
    let bestModel = ''
    for (const m of models) {
      if (avgByModel[m][metric] < best) {
        best = avgByModel[m][metric]
        bestModel = m
      }
    }
    bestPerMetric[metric] = bestModel
  }

  return (
    <>
      <div className="card">
        <h2>Model Comparison — {state}</h2>
        <div className="subtitle">Mean RMSE across CV folds (lower is better)</div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={rmseData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 13 }} />
            <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
              formatter={(v) => [fmt(v), 'RMSE']}
            />
            <Bar dataKey="rmse" radius={[4, 4, 0, 0]} barSize={48}>
              {rmseData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Detailed Metrics</h2>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>MAPE (%)</th>
              <th>SMAPE (%)</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m}>
                <td style={{ color: COLORS[m], fontWeight: 600 }}>{m.toUpperCase()}</td>
                {METRICS.map(metric => {
                  const v = avgByModel[m][metric]
                  const isBest = bestPerMetric[metric] === m
                  return (
                    <td key={metric} className={isBest ? 'best' : ''}>
                      {metric === 'mape' || metric === 'smape'
                        ? v.toFixed(1) + '%'
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
