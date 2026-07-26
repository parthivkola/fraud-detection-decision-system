import React, { useState, useEffect } from 'react';
import { Cpu, Sliders, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { api, type ModelVersion } from '../api';

export const ModelsTab: React.FC = () => {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [updatingId, setUpdatingId] = useState<number | null>(null);
  const [weights, setWeights] = useState<Record<number, number>>({});
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const list = await api.getModels();
      setModels(list);
      const w: Record<number, number> = {};
      list.forEach((m) => (w[m.id] = m.ab_weight));
      setWeights(w);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleActive = async (m: ModelVersion) => {
    setUpdatingId(m.id);
    setMessage(null);
    try {
      const updated = await api.setModelActive(m.id, !m.is_active);
      setModels(models.map((item) => (item.id === m.id ? updated : item)));
      setMessage({ type: 'success', text: `Model '${m.version_tag}' ${updated.is_active ? 'activated' : 'deactivated'}.` });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to update status' });
    } finally {
      setUpdatingId(null);
    }
  };

  const handleWeightSave = async (m: ModelVersion) => {
    const val = weights[m.id];
    if (val === undefined || val < 0 || val > 1) {
      setMessage({ type: 'error', text: 'Weight must be between 0.0 and 1.0' });
      return;
    }
    setUpdatingId(m.id);
    setMessage(null);
    try {
      const updated = await api.setModelWeight(m.id, val);
      setModels(models.map((item) => (item.id === m.id ? updated : item)));
      setMessage({ type: 'success', text: `Weight for '${m.version_tag}' updated to ${(val * 100).toFixed(0)}%.` });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to update weight' });
    } finally {
      setUpdatingId(null);
    }
  };

  if (loading && models.length === 0) {
    return <div style={{ padding: '4rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>Loading model registry...</div>;
  }

  return (
    <div style={{ padding: '1.5rem 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.75rem', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem', color: '#f4f4f5' }}>Model Registry & A/B Routing</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem' }}>Manage XGBoost model versions, toggle production activation, and adjust traffic weights</p>
        </div>
        <button onClick={loadModels} className="btn btn-secondary" style={{ padding: '0.4rem 0.875rem', fontSize: '0.8125rem' }}>
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {message && (
        <div style={{
          background: message.type === 'success' ? 'var(--status-success-bg)' : 'var(--status-danger-bg)',
          border: `1px solid ${message.type === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
          color: message.type === 'success' ? 'var(--status-success)' : 'var(--status-danger)',
          padding: '0.625rem 0.875rem',
          borderRadius: 'var(--radius-md)',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.8125rem'
        }}>
          {message.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          <span>{message.text}</span>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {models.map((m) => {
          const w = weights[m.id] ?? m.ab_weight;
          const isBusy = updatingId === m.id;
          return (
            <div key={m.id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderLeft: m.is_active ? '3px solid var(--status-success)' : '3px solid var(--border-color)' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: '8px',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: m.is_active ? 'var(--status-success)' : 'var(--text-muted)'
                  }}>
                    <Cpu size={20} />
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <h3 style={{ fontSize: '1.05rem', margin: 0, color: '#f4f4f5', fontFamily: 'var(--font-mono)' }}>{m.version_tag}</h3>
                      <span className={`badge ${m.is_active ? 'badge-low' : 'badge-high'}`}>
                        {m.is_active ? 'ACTIVE' : 'INACTIVE'}
                      </span>
                    </div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.8125rem', marginTop: '0.2rem' }}>
                      {m.description || 'No description provided.'}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <button
                    onClick={() => handleToggleActive(m)}
                    disabled={isBusy}
                    className={m.is_active ? 'btn btn-secondary' : 'btn btn-primary'}
                    style={{
                      padding: '0.4rem 0.875rem',
                      fontSize: '0.8125rem',
                      background: m.is_active ? 'var(--status-danger-bg)' : undefined,
                      color: m.is_active ? 'var(--status-danger)' : undefined,
                      borderColor: m.is_active ? 'rgba(239, 68, 68, 0.2)' : undefined
                    }}
                  >
                    {m.is_active ? (
                      <>
                        <XCircle size={14} />
                        <span>Deactivate</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle size={14} />
                        <span>Activate</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Artifact Paths & Settings */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', background: 'var(--bg-input)', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Weights Artifact</span>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: '#a1a1aa', marginTop: '0.1rem' }}>{m.file_path}</div>
                </div>

                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Scaler Pipeline</span>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: '#a1a1aa', marginTop: '0.1rem' }}>{m.scaler_path}</div>
                </div>

                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Metadata Schema</span>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: '#a1a1aa', marginTop: '0.1rem' }}>{m.metadata_path}</div>
                </div>
              </div>

              {/* A/B Testing Weight Controls */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', paddingTop: '0.25rem', borderTop: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem', flex: 1, minWidth: '260px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)' }}>
                    <Sliders size={15} />
                    <span style={{ fontSize: '0.8125rem', fontWeight: 500, whiteSpace: 'nowrap' }}>A/B Traffic Weight:</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={w}
                    onChange={(e) => setWeights({ ...weights, [m.id]: parseFloat(e.target.value) })}
                    disabled={isBusy || !m.is_active}
                    style={{ flex: 1, accentColor: 'var(--brand-primary)', cursor: m.is_active ? 'pointer' : 'not-allowed' }}
                  />
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.875rem', color: '#f4f4f5', width: '45px', textAlign: 'right' }}>
                    {(w * 100).toFixed(0)}%
                  </span>
                </div>

                <button
                  onClick={() => handleWeightSave(m)}
                  disabled={isBusy || w === m.ab_weight || !m.is_active}
                  className="btn btn-secondary"
                  style={{ padding: '0.35rem 0.875rem', fontSize: '0.75rem', background: w !== m.ab_weight ? 'var(--brand-subtle)' : undefined, color: w !== m.ab_weight ? 'var(--brand-primary)' : undefined, borderColor: w !== m.ab_weight ? 'var(--brand-primary)' : undefined }}
                >
                  Save Weight
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
