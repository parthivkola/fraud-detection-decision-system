import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, CheckCircle2, Clock, TrendingUp, Filter, RefreshCw } from 'lucide-react';
import { api, type MetricsResponse, type ModelVersion } from '../api';

/* ── Tiny Gauge Bar ───────────────────────────────────────────────────────── */
function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div className="progress-track" style={{ flex: 1 }}>
        <div className="progress-fill" style={{ width: `${(value * 100).toFixed(1)}%`, background: color }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: '#171717', minWidth: 42, textAlign: 'right' }}>
        {(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

/* ── Risk Row ─────────────────────────────────────────────────────────────── */
function RiskRow({ label, count, total, color, badgeClass }: {
  label: string; count: number; total: number; color: string; badgeClass: string;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span className={`badge ${badgeClass}`} style={{ minWidth: 70, justifyContent: 'center' }}>{label}</span>
      <div className="progress-track" style={{ flex: 1 }}>
        <div className="progress-fill" style={{ width: `${pct.toFixed(1)}%`, background: color }} />
      </div>
      <div style={{ minWidth: 90, textAlign: 'right', fontSize: 12, color: '#737373', fontFamily: 'var(--font-mono)' }}>
        <strong style={{ color: '#171717' }}>{count.toLocaleString()}</strong>&nbsp;({pct.toFixed(1)}%)
      </div>
    </div>
  );
}

/* ── Uptime formatter ─────────────────────────────────────────────────────── */
function formatUptime(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(1)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

/* ── Component ────────────────────────────────────────────────────────────── */
export const MetricsTab: React.FC = () => {
  const [metrics, setMetrics]           = useState<MetricsResponse | null>(null);
  const [models, setModels]             = useState<ModelVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading]           = useState(true);
  const [refreshing, setRefreshing]     = useState(false);

  useEffect(() => {
    loadModels();
    loadMetrics('');
  }, []);

  const loadModels = async () => {
    try { setModels(await api.getModels()); } catch { /* silent */ }
  };

  const loadMetrics = async (tag: string) => {
    setLoading(true);
    try { setMetrics(await api.getMetrics(tag || undefined)); }
    catch { /* silent */ }
    finally { setLoading(false); }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try { setMetrics(await api.getMetrics(selectedModel || undefined)); }
    catch { /* silent */ }
    finally { setRefreshing(false); }
  };

  const handleFilter = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const tag = e.target.value;
    setSelectedModel(tag);
    loadMetrics(tag);
  };

  /* ── Skeleton ── */
  if (loading && !metrics) {
    return (
      <div style={{ padding: '28px 0' }}>
        <div className="section-header">
          <div>
            <div className="skeleton" style={{ width: 200, height: 20, marginBottom: 6 }} />
            <div className="skeleton" style={{ width: 280, height: 14 }} />
          </div>
        </div>
        <div className="grid-4" style={{ marginBottom: 20 }}>
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card" style={{ padding: '20px' }}>
              <div className="skeleton" style={{ width: '60%', height: 12, marginBottom: 10 }} />
              <div className="skeleton" style={{ width: '40%', height: 28, marginBottom: 6 }} />
              <div className="skeleton" style={{ width: '80%', height: 12 }} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div style={{ padding: '28px 0', textAlign: 'center', color: '#dc2626' }}>
        Failed to load telemetry data.
      </div>
    );
  }

  return (
    <div style={{ padding: '28px 0' }}>

      {/* Section header */}
      <div className="section-header">
        <div>
          <h2 className="section-title">System Metrics</h2>
          <p className="section-sub">Live inference statistics and model evaluation results</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Model filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#fff', border: '1px solid #e5e5e5', borderRadius: 8, padding: '6px 10px' }}>
            <Filter size={12} color="#a3a3a3" />
            <select
              value={selectedModel}
              onChange={handleFilter}
              style={{ background: 'transparent', border: 'none', outline: 'none', fontSize: 12, color: '#171717', fontFamily: 'inherit', cursor: 'pointer' }}
            >
              <option value="">All models</option>
              {models.map(m => <option key={m.id} value={m.version_tag}>{m.version_tag}</option>)}
            </select>
          </div>
          {/* Refresh */}
          <button onClick={handleRefresh} className="btn btn-secondary btn-sm" disabled={refreshing}>
            <RefreshCw size={12} className={refreshing ? 'spin' : ''} style={refreshing ? { animation: 'spin 0.9s linear infinite' } : {}} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Primary KPIs */}
      <div className="grid-4" style={{ marginBottom: 20 }}>
        {[
          {
            label: 'Total Inferences', value: metrics.total_predictions.toLocaleString(),
            sub: `${metrics.total_batches} batch jobs`, Icon: Activity, iconColor: '#171717',
          },
          {
            label: 'Flagged Fraud', value: metrics.flagged_fraud.toLocaleString(),
            sub: `${(metrics.fraud_flag_rate * 100).toFixed(2)}% flag rate`,
            Icon: ShieldAlert, iconColor: '#dc2626', valueColor: '#dc2626', accent: '#dc2626',
          },
          {
            label: 'Approved', value: metrics.flagged_legitimate.toLocaleString(),
            sub: 'Auto-cleared',
            Icon: CheckCircle2, iconColor: '#16a34a', valueColor: '#16a34a', accent: '#16a34a',
          },
          {
            label: 'Uptime', value: formatUptime(metrics.uptime_seconds),
            sub: `Threshold: ${metrics.threshold}`,
            Icon: Clock, iconColor: '#737373',
          },
        ].map(kpi => (
          <div key={kpi.label} className="card" style={{ padding: '18px 20px', borderLeft: (kpi as any).accent ? `3px solid ${(kpi as any).accent}` : undefined }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <div className="stat-label">{kpi.label}</div>
              <kpi.Icon size={15} color={kpi.iconColor} />
            </div>
            <div className="stat-value" style={{ color: (kpi as any).valueColor ?? '#171717' }}>{kpi.value}</div>
            <div className="stat-sub">{kpi.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>

        {/* ML Evaluation Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 20 }}>
            <TrendingUp size={16} color="#171717" />
            <div>
              <h3 style={{ fontSize: 14, fontWeight: 600, color: '#171717' }}>Model Evaluation</h3>
              <p style={{ fontSize: 12, color: '#a3a3a3' }}>Holdout test set benchmark scores</p>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {[
              { label: 'ROC AUC',   value: metrics.model_roc_auc,   color: '#171717' },
              { label: 'F1 Score',  value: metrics.model_f1,        color: '#171717' },
              { label: 'Precision', value: metrics.model_precision,  color: '#16a34a' },
              { label: 'Recall',    value: metrics.model_recall,     color: '#d97706' },
            ].map(m => (
              <div key={m.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                  <span style={{ fontSize: 12, fontWeight: 500, color: '#525252' }}>{m.label}</span>
                </div>
                <ScoreBar value={m.value} color={m.color} />
              </div>
            ))}
          </div>

          {/* Accuracy callout */}
          <div style={{ marginTop: 18, padding: '12px 14px', background: '#f5f5f5', borderRadius: 8, border: '1px solid #e5e5e5', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 12, color: '#737373', fontWeight: 500 }}>Overall Accuracy</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: '#171717' }}>
              {(metrics.model_accuracy * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Risk Distribution + Active Models */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Risk distribution */}
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, color: '#171717', marginBottom: 4 }}>Risk Distribution</h3>
            <p style={{ fontSize: 12, color: '#a3a3a3', marginBottom: 18 }}>Transaction volume by severity tier</p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {[
                { label: 'LOW',      count: metrics.risk_distribution.LOW      ?? 0, color: '#16a34a', badgeClass: 'badge-low'      },
                { label: 'MEDIUM',   count: metrics.risk_distribution.MEDIUM   ?? 0, color: '#d97706', badgeClass: 'badge-medium'   },
                { label: 'HIGH',     count: metrics.risk_distribution.HIGH     ?? 0, color: '#c2410c', badgeClass: 'badge-high'     },
                { label: 'CRITICAL', count: metrics.risk_distribution.CRITICAL ?? 0, color: '#dc2626', badgeClass: 'badge-critical' },
              ].map(r => (
                <RiskRow key={r.label} {...r} total={metrics.total_predictions} />
              ))}
            </div>
          </div>

          {/* Active model versions */}
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, color: '#171717', marginBottom: 4 }}>Active A/B Pool</h3>
            <p style={{ fontSize: 12, color: '#a3a3a3', marginBottom: 14 }}>Currently serving model versions</p>
            {metrics.active_model_versions.length === 0 ? (
              <p style={{ fontSize: 13, color: '#a3a3a3' }}>No active models.</p>
            ) : (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {metrics.active_model_versions.map(tag => (
                  <span key={tag} className="tag" style={{ background: '#f0fdf4', color: '#16a34a', borderColor: '#bbf7d0' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#16a34a', display: 'inline-block' }} />
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
