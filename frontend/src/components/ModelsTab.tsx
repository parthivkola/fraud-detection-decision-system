import React, { useState, useEffect } from 'react';
import { Cpu, Sliders, CheckCircle, XCircle, AlertCircle, RefreshCw, Calendar } from 'lucide-react';
import { api, type ModelVersion } from '../api';

function WeightBar({ value }: { value: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: '#f5f5f5', borderRadius: 99, overflow: 'hidden' }}>
        <div style={{ width: `${(value * 100).toFixed(0)}%`, height: '100%', background: '#171717', borderRadius: 99, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: '#525252', minWidth: 30 }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export const ModelsTab: React.FC = () => {
  const [models, setModels]       = useState<ModelVersion[]>([]);
  const [loading, setLoading]     = useState(true);
  const [updatingId, setUpdatingId] = useState<number | null>(null);
  const [weights, setWeights]     = useState<Record<number, number>>({});
  const [message, setMessage]     = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => { loadModels(); }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const list = await api.getModels();
      setModels(list);
      const w: Record<number, number> = {};
      list.forEach(m => (w[m.id] = m.ab_weight));
      setWeights(w);
    } catch { /* silent */ }
    finally { setLoading(false); }
  };

  const handleToggle = async (m: ModelVersion) => {
    setUpdatingId(m.id);
    setMessage(null);
    try {
      const updated = await api.setModelActive(m.id, !m.is_active);
      setModels(prev => prev.map(item => item.id === m.id ? updated : item));
      setMessage({ type: 'success', text: `"${m.version_tag}" ${updated.is_active ? 'activated' : 'deactivated'} successfully.` });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to update model status.' });
    } finally {
      setUpdatingId(null);
    }
  };

  const handleWeightSave = async (m: ModelVersion) => {
    const val = weights[m.id];
    if (val === undefined || val < 0 || val > 1) {
      setMessage({ type: 'error', text: 'Weight must be between 0.0 and 1.0.' });
      return;
    }
    setUpdatingId(m.id);
    setMessage(null);
    try {
      const updated = await api.setModelWeight(m.id, val);
      setModels(prev => prev.map(item => item.id === m.id ? updated : item));
      setMessage({ type: 'success', text: `Traffic weight for "${m.version_tag}" set to ${(val * 100).toFixed(0)}%.` });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to update weight.' });
    } finally {
      setUpdatingId(null);
    }
  };

  /* ── Skeleton ── */
  if (loading && models.length === 0) {
    return (
      <div style={{ padding: '28px 0' }}>
        <div className="section-header">
          <div>
            <div className="skeleton" style={{ width: 200, height: 20, marginBottom: 6 }} />
            <div className="skeleton" style={{ width: 340, height: 14 }} />
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card">
              <div className="skeleton" style={{ width: '30%', height: 18, marginBottom: 10 }} />
              <div className="skeleton" style={{ width: '60%', height: 13, marginBottom: 16 }} />
              <div className="skeleton" style={{ width: '100%', height: 4 }} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '28px 0' }}>

      {/* Header */}
      <div className="section-header">
        <div>
          <h2 className="section-title">Model Registry</h2>
          <p className="section-sub">Manage versions, toggle production activation, and tune A/B traffic routing</p>
        </div>
        <button onClick={loadModels} className="btn btn-secondary btn-sm">
          <RefreshCw size={12} /><span>Refresh</span>
        </button>
      </div>

      {/* Message banner */}
      {message && (
        <div className={`alert ${message.type === 'success' ? 'alert-success' : 'alert-error'} fade-in`}>
          {message.type === 'success' ? <CheckCircle size={13} /> : <AlertCircle size={13} />}
          <span>{message.text}</span>
          <button onClick={() => setMessage(null)} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex' }}>
            <XCircle size={13} />
          </button>
        </div>
      )}

      {models.length === 0 && (
        <div style={{ padding: '40px 0', textAlign: 'center', color: '#a3a3a3' }}>
          <Cpu size={28} style={{ marginBottom: 10, opacity: 0.4 }} />
          <p style={{ fontSize: 14 }}>No model versions registered.</p>
        </div>
      )}

      {/* Model cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {models.map(m => {
          const w = weights[m.id] ?? m.ab_weight;
          const isBusy = updatingId === m.id;
          const weightChanged = w !== m.ab_weight;
          const createdDate = new Date(m.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });

          return (
            <div
              key={m.id}
              className="card fade-in"
              style={{
                borderLeft: `3px solid ${m.is_active ? '#16a34a' : '#e5e5e5'}`,
                padding: '20px 24px',
                transition: 'all 0.2s ease',
              }}
            >
              {/* Top row */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  {/* Icon */}
                  <div style={{
                    width: 38, height: 38, borderRadius: 8,
                    background: m.is_active ? '#f0fdf4' : '#f5f5f5',
                    border: `1px solid ${m.is_active ? '#bbf7d0' : '#e5e5e5'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: m.is_active ? '#16a34a' : '#a3a3a3',
                    flexShrink: 0,
                  }}>
                    <Cpu size={18} />
                  </div>

                  {/* Name + status */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                      <h3 style={{ fontSize: 15, fontWeight: 700, color: '#171717', fontFamily: 'var(--font-mono)', margin: 0 }}>{m.version_tag}</h3>
                      <span className={`badge ${m.is_active ? 'badge-active' : 'badge-inactive'}`}>
                        <span style={{ width: 5, height: 5, borderRadius: '50%', background: m.is_active ? '#16a34a' : '#a3a3a3', display: 'inline-block' }} />
                        {m.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                    <p style={{ fontSize: 12, color: '#737373', margin: 0 }}>
                      {m.description || 'No description provided.'}
                    </p>
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#a3a3a3' }}>
                    <Calendar size={11} />
                    <span>{createdDate}</span>
                  </div>
                  <button
                    onClick={() => handleToggle(m)}
                    disabled={isBusy}
                    className={m.is_active ? 'btn btn-danger btn-sm' : 'btn btn-success btn-sm'}
                    style={{ minWidth: 96 }}
                  >
                    {isBusy ? (
                      <RefreshCw size={11} className="spin" style={{ animation: 'spin 0.9s linear infinite' }} />
                    ) : m.is_active ? (
                      <><XCircle size={11} /><span>Deactivate</span></>
                    ) : (
                      <><CheckCircle size={11} /><span>Activate</span></>
                    )}
                  </button>
                </div>
              </div>

              {/* Artifact paths */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8, marginBottom: 16, padding: '10px 12px', background: '#fafafa', borderRadius: 6, border: '1px solid #e5e5e5' }}>
                {[
                  { label: 'Model Weights', path: m.file_path },
                  { label: 'Scaler',         path: m.scaler_path },
                  { label: 'Metadata',       path: m.metadata_path },
                ].map(art => (
                  <div key={art.label}>
                    <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a3a3a3', marginBottom: 2 }}>{art.label}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#525252', wordBreak: 'break-all' }}>{art.path}</div>
                  </div>
                ))}
              </div>

              {/* A/B Weight slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', paddingTop: 12, borderTop: '1px solid #f5f5f5' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#737373', flexShrink: 0 }}>
                  <Sliders size={13} />
                  <span style={{ fontSize: 12, fontWeight: 500 }}>A/B Traffic Weight</span>
                </div>

                <div style={{ flex: 1, minWidth: 160, display: 'flex', alignItems: 'center', gap: 10 }}>
                  <WeightBar value={w} />
                </div>

                <input
                  type="range" min={0} max={1} step={0.05}
                  value={w}
                  onChange={e => setWeights({ ...weights, [m.id]: parseFloat(e.target.value) })}
                  disabled={isBusy || !m.is_active}
                  style={{ width: 120, accentColor: '#171717', cursor: m.is_active ? 'pointer' : 'not-allowed' }}
                />

                <button
                  onClick={() => handleWeightSave(m)}
                  disabled={isBusy || !weightChanged || !m.is_active}
                  className="btn btn-sm"
                  style={{
                    background: weightChanged && m.is_active ? '#171717' : '#f5f5f5',
                    color: weightChanged && m.is_active ? '#fff' : '#a3a3a3',
                    border: '1px solid',
                    borderColor: weightChanged && m.is_active ? '#171717' : '#e5e5e5',
                    transition: 'all 0.15s ease',
                    minWidth: 80,
                  }}
                >
                  {isBusy ? <RefreshCw size={11} style={{ animation: 'spin 0.9s linear infinite' }} /> : 'Save Weight'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
