import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, Cpu, CheckCircle2, Clock, BarChart3, Filter } from 'lucide-react';
import { api, type MetricsResponse, type ModelVersion } from '../api';

export const MetricsTab: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModels();
    loadMetrics('');
  }, []);

  const loadModels = async () => {
    try {
      const list = await api.getModels();
      setModels(list);
    } catch (e) {
      console.error(e);
    }
  };

  const loadMetrics = async (tag: string) => {
    setLoading(true);
    try {
      const data = await api.getMetrics(tag || undefined);
      setMetrics(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleModelFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const tag = e.target.value;
    setSelectedModel(tag);
    loadMetrics(tag);
  };

  if (loading && !metrics) {
    return <div style={{ padding: '4rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>Loading system telemetry & analytics...</div>;
  }

  if (!metrics) {
    return <div style={{ padding: '4rem 0', textAlign: 'center', color: 'var(--status-danger)' }}>Failed to load telemetry data.</div>;
  }

  return (
    <div style={{ padding: '1.5rem 0' }}>
      {/* Header & Filter */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.75rem', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem', color: '#f4f4f5' }}>System Telemetry & KPIs</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem' }}>Real-time XGBoost inference statistics and model evaluation metrics</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', background: 'var(--bg-input)', padding: '0.4rem 0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
          <Filter size={14} color="var(--text-secondary)" />
          <span style={{ fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Model Filter:</span>
          <select
            value={selectedModel}
            onChange={handleModelFilterChange}
            style={{
              background: 'transparent',
              color: '#f4f4f5',
              border: 'none',
              padding: '0.2rem 0.4rem',
              fontFamily: 'var(--font-main)',
              fontSize: '0.8125rem',
              fontWeight: 500,
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            <option value="" style={{ background: '#14161f' }}>All Models (Aggregate)</option>
            {models.map((m) => (
              <option key={m.id} value={m.version_tag} style={{ background: '#14161f' }}>
                {m.version_tag}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Total Inferences</span>
            <Activity size={16} color="var(--brand-primary)" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600, color: '#f4f4f5', fontFamily: 'var(--font-mono)' }}>{metrics.total_predictions.toLocaleString()}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Across {metrics.total_batches} batch jobs</span>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Flagged Fraud</span>
            <ShieldAlert size={16} color="var(--status-danger)" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--status-danger)', fontFamily: 'var(--font-mono)' }}>{metrics.flagged_fraud.toLocaleString()}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--status-danger)' }}>Rate: {(metrics.fraud_flag_rate * 100).toFixed(2)}%</span>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Approved Volume</span>
            <CheckCircle2 size={16} color="var(--status-success)" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--status-success)', fontFamily: 'var(--font-mono)' }}>{metrics.flagged_legitimate.toLocaleString()}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--status-success)' }}>Cleared automatically</span>
        </div>

        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Uptime</span>
            <Clock size={16} color="var(--text-secondary)" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600, color: '#f4f4f5', fontFamily: 'var(--font-mono)' }}>{(metrics.uptime_seconds / 60).toFixed(1)}m</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Threshold: {metrics.threshold}</span>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* ML Evaluation Metrics Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.875rem' }}>
            <BarChart3 size={18} color="var(--brand-primary)" />
            <div>
              <h3 style={{ fontSize: '1rem', margin: 0, color: '#f4f4f5' }}>Model Evaluation (Test Validation Split)</h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Holdout test set benchmark statistics</span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
            <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 500 }}>ROC AUC</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem', fontFamily: 'var(--font-mono)' }}>{(metrics.model_roc_auc * 100).toFixed(2)}%</div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 500 }}>F1 Score</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem', fontFamily: 'var(--font-mono)' }}>{(metrics.model_f1 * 100).toFixed(2)}%</div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 500 }}>Precision</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem', fontFamily: 'var(--font-mono)' }}>{(metrics.model_precision * 100).toFixed(2)}%</div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 500 }}>Recall</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem', fontFamily: 'var(--font-mono)' }}>{(metrics.model_recall * 100).toFixed(2)}%</div>
            </div>
          </div>
        </div>

        {/* Risk Level Distribution Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.875rem' }}>
            <Cpu size={18} color="var(--text-secondary)" />
            <div>
              <h3 style={{ fontSize: '1rem', margin: 0, color: '#f4f4f5' }}>Risk Classification Distribution</h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Volume distribution by severity queue</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
            {[
              { label: 'LOW', count: metrics.risk_distribution.LOW || 0, color: 'var(--status-success)' },
              { label: 'MEDIUM', count: metrics.risk_distribution.MEDIUM || 0, color: 'var(--status-warning)' },
              { label: 'CRITICAL', count: metrics.risk_distribution.CRITICAL || 0, color: 'var(--status-danger)' },
            ].map((item) => {
              const total = metrics.total_predictions || 1;
              const pct = ((item.count / total) * 100).toFixed(1);
              return (
                <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ width: '70px', fontWeight: 600, fontSize: '0.75rem', color: item.color, fontFamily: 'var(--font-mono)' }}>{item.label}</div>
                  <div style={{ flex: 1, height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: item.color, transition: 'width 0.3s ease' }} />
                  </div>
                  <div style={{ width: '90px', textAlign: 'right', fontSize: '0.8125rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    <strong style={{ color: '#f4f4f5' }}>{item.count}</strong> ({pct}%)
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: '1.5rem', padding: '0.875rem', background: 'var(--bg-input)', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            <div style={{ fontWeight: 500, color: '#f4f4f5', marginBottom: '0.375rem' }}>Active A/B Pool Models:</div>
            <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
              {metrics.active_model_versions.map((t) => (
                <span key={t} style={{ background: 'var(--bg-elevated)', color: '#f4f4f5', padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', border: '1px solid var(--border-color)', fontFamily: 'var(--font-mono)' }}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
