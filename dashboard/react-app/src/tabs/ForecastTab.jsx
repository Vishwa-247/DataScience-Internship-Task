import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts'
import { fetchPredict } from '../api'

const fmt = (v) => {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'
  return v.toFixed(0)
}

export default function ForecastTab({ state }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchPredict(state)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [state])

  if (loading) return <div className="loading">Loading forecast…</div>
  if (error) return <div className="error-box">{error}</div>
  if (!data) return null

  const chartData = data.forecast.map(p => ({
    date: p.date,
    yhat: p.yhat,
    ci: [p.yhat_lower, p.yhat_upper],
  }))

  return (
    <>
      <div className="card">
        <h2>8-Week Forecast — {state}</h2>
        <div className="subtitle">
          Ensemble: {data.selected_models.join(' + ')} (
          {Object.entries(data.ensemble_weights).map(([m, w]) =>
            `${m}: ${(w * 100).toFixed(0)}%`
          ).join(', ')})
        </div>

        <div className="chips">
          {data.forecast.slice(0, 4).map(p => (
            <div className="chip" key={p.date}>
              <span className="label">{p.date}</span>
              <span className="value">{fmt(p.yhat)}</span>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tickFormatter={fmt} tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
              labelStyle={{ color: '#e2e8f0' }}
              formatter={(v) => [typeof v === 'number' ? fmt(v) : v]}
            />
            <Legend />
            <Area
              dataKey="ci"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.15}
              name="95% CI"
            />
            <Line
              dataKey="yhat"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', r: 4 }}
              name="Ensemble"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}
