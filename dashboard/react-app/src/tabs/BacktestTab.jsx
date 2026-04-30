import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine
} from 'recharts'
import { fetchBacktest } from '../api'

const fmt = (v) => {
  if (v == null || isNaN(v)) return '—'
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v.toFixed(0)
}

const COLORS = {
  y_true: '#22c55e', arima: '#3b82f6', prophet: '#f59e0b', xgboost: '#ef4444', lstm: '#a855f7',
}

const MODEL_NAMES = ['arima', 'prophet', 'xgboost', 'lstm']

function rmse(rows, key) {
  const pairs = rows.filter(r => r[key] != null && r.y_true != null)
  if (!pairs.length) return null
  const sum = pairs.reduce((acc, r) => acc + (r[key] - r.y_true) ** 2, 0)
  return Math.sqrt(sum / pairs.length)
}

export default function BacktestTab({ state }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchBacktest(state)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [state])

  if (loading) return <div className="loading">Loading backtest…</div>
  if (error) return <div className="error-box">{error}</div>
  if (!data) return null

  const rows = data.rows
  const horizon = 8
  const modelRmse = MODEL_NAMES.map(m => ({ model: m, val: rmse(rows, m) }))
  const best = modelRmse.reduce((b, m) => (m.val != null && (b == null || m.val < b.val) ? m : b), null)

  // Fold boundary dates (every 8 rows)
  const foldBoundaries = rows
    .filter((_, i) => i > 0 && i % horizon === 0)
    .map(r => r.date)

  return (
    <>
      {/* Per-model RMSE cards */}
      <div className="kpi-grid">
        {modelRmse.map(({ model, val }) => (
          <div key={model} className="kpi-card" style={{ borderTop: `3px solid ${COLORS[model]}` }}>
            <div className="kpi-label">{model.toUpperCase()}</div>
            <div className="kpi-value" style={{ color: model === best?.model ? '#22c55e' : undefined }}>
              {val != null ? fmt(val) : '—'}
            </div>
            <div className="kpi-sub">
              {model === best?.model ? '🏆 Lowest RMSE' : 'Backtest RMSE'}
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h2>Walk-Forward Backtest — {state}</h2>
            <div className="subtitle">
              {rows.length} test weeks across {Math.ceil(rows.length / horizon)} CV folds — vertical lines mark fold boundaries
            </div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={rows} margin={{ top: 10, right: 20, bottom: 50, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-35} textAnchor="end" height={60} />
            <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
              labelStyle={{ color: '#e2e8f0' }}
              formatter={(v, name) => [fmt(v), name]}
            />
            <Legend verticalAlign="top" />
            {foldBoundaries.map(d => (
              <ReferenceLine key={d} x={d} stroke="#475569" strokeDasharray="6 3"
                label={{ value: 'Fold', position: 'top', fill: '#64748b', fontSize: 10 }} />
            ))}
            <Line dataKey="y_true" stroke={COLORS.y_true} strokeWidth={2.5} dot={false} name="Actual" />
            <Line dataKey="arima" stroke={COLORS.arima} strokeWidth={1.5} dot={false} strokeDasharray="5 2" name="ARIMA" />
            <Line dataKey="prophet" stroke={COLORS.prophet} strokeWidth={1.5} dot={false} strokeDasharray="5 2" name="Prophet" />
            <Line dataKey="xgboost" stroke={COLORS.xgboost} strokeWidth={1.5} dot={false} strokeDasharray="5 2" name="XGBoost" />
            <Line dataKey="lstm" stroke={COLORS.lstm} strokeWidth={1.5} dot={false} strokeDasharray="5 2" name="LSTM" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}
