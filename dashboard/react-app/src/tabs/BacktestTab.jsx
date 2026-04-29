import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts'
import { fetchBacktest } from '../api'

const fmt = (v) => {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v?.toFixed(0) ?? ''
}

const COLORS = {
  y_true: '#22c55e',
  arima: '#3b82f6',
  prophet: '#f59e0b',
  xgboost: '#ef4444',
  lstm: '#a855f7',
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

  return (
    <div className="card">
      <h2>Backtest — {state}</h2>
      <div className="subtitle">
        Actual vs predicted across {data.rows.length} test weeks (all CV folds)
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data.rows} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-30} textAnchor="end" height={50} />
          <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(v) => [fmt(v)]}
          />
          <Legend />
          <Line dataKey="y_true" stroke={COLORS.y_true} strokeWidth={2.5} dot={false} name="Actual" />
          <Line dataKey="arima" stroke={COLORS.arima} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="ARIMA" />
          <Line dataKey="prophet" stroke={COLORS.prophet} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="Prophet" />
          <Line dataKey="xgboost" stroke={COLORS.xgboost} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="XGBoost" />
          <Line dataKey="lstm" stroke={COLORS.lstm} strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="LSTM" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
