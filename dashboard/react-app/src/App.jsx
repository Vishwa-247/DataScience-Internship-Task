import { useState, useEffect } from 'react'
import { fetchHealth, fetchStates } from './api'
import ForecastTab from './tabs/ForecastTab'
import BacktestTab from './tabs/BacktestTab'
import MetricsTab from './tabs/MetricsTab'

const TABS = ['Forecast', 'Backtest', 'Metrics']

export default function App() {
  const [activeTab, setActiveTab] = useState('Forecast')
  const [states, setStates] = useState([])
  const [selectedState, setSelectedState] = useState('')
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchHealth()
      .then(h => {
        setHealth(h)
        return fetchStates()
      })
      .then(s => {
        setStates(s.trained || [])
        if (s.trained?.length) setSelectedState(s.trained[0])
      })
      .catch(e => setError(e.message))
  }, [])

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div>
          <h1>Sales Forecast</h1>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Beverages · 43 US States · 2019–2023
          </div>
        </div>

        <div className="sidebar-section">
          <div className="section-label">Select State</div>
          <select
            value={selectedState}
            onChange={e => setSelectedState(e.target.value)}
            disabled={!states.length}
          >
            {states.length === 0 && <option>No trained states</option>}
            {states.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        {health && (
          <div className="sidebar-section">
            <div className="section-label">Registry Info</div>
            <div className="info-row"><strong>{health.trained_states}</strong> states trained</div>
            <div className="info-row" style={{ fontSize: 10, wordBreak: 'break-all' }}>
              v{health.version?.slice(1, 9) || '—'}
            </div>
          </div>
        )}

        <div className={`status ${health ? '' : 'offline'}`} style={{ marginTop: 'auto' }}>
          <span className="dot" />
          {health ? 'API online' : 'API offline'}
        </div>
      </aside>

      {/* Main area */}
      <main className="main">
        {error && <div className="error-box">{error}</div>}

        <div className="tabs">
          {TABS.map(t => (
            <button
              key={t}
              className={`tab-btn ${activeTab === t ? 'active' : ''}`}
              onClick={() => setActiveTab(t)}
            >
              {t}
            </button>
          ))}
        </div>

        {selectedState && activeTab === 'Forecast' && (
          <ForecastTab state={selectedState} />
        )}
        {selectedState && activeTab === 'Backtest' && (
          <BacktestTab state={selectedState} />
        )}
        {selectedState && activeTab === 'Metrics' && (
          <MetricsTab state={selectedState} />
        )}
        {!selectedState && (
          <div className="loading">
            No trained state selected. Run training first via CLI or POST /train.
          </div>
        )}
      </main>
    </div>
  )
}
