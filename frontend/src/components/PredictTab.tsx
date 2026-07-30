import React, { useState, useEffect, useCallback } from 'react';
import {
  UploadCloud, FileText, Download, CheckCircle2, AlertTriangle,
  ShieldAlert, Cpu, RefreshCw, X, ChevronDown
} from 'lucide-react';
import { api, type ModelVersion, type PredictResponse } from '../api';

/* ── Helpers ──────────────────────────────────────────────────────────────── */
const riskColors: Record<string, { text: string; bar: string }> = {
  LOW:      { text: '#16a34a', bar: '#16a34a' },
  MEDIUM:   { text: '#d97706', bar: '#d97706' },
  HIGH:     { text: '#c2410c', bar: '#ea580c' },
  CRITICAL: { text: '#dc2626', bar: '#dc2626' },
};

const decisionColors: Record<string, string> = {
  approve: '#16a34a',
  review:  '#d97706',
  block:   '#dc2626',
};

function ProbBar({ value, isFraud }: { value: number; isFraud: boolean }) {
  const pct = Math.min(100, value * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, maxWidth: 64, height: 4, background: '#f5f5f5', borderRadius: 99 }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 99, background: isFraud ? '#dc2626' : '#16a34a', transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500, color: isFraud ? '#dc2626' : '#171717', minWidth: 44, textAlign: 'right' }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

/* ── Component ────────────────────────────────────────────────────────────── */
export const PredictTab: React.FC = () => {
  const [file, setFile]             = useState<File | null>(null);
  const [models, setModels]         = useState<ModelVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState<PredictResponse | null>(null);
  const [error, setError]           = useState<string | null>(null);
  const [dragging, setDragging]     = useState(false);

  useEffect(() => { loadModels(); }, []);

  const loadModels = async () => {
    try { setModels(await api.getModels()); } catch { /* silent */ }
  };

  const handleFileChange = (f: File | null) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a .csv file.');
      return;
    }
    setFile(f);
    setError(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileChange(f);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setError('Please select a CSV file first.'); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.predict(file, selectedModel || undefined);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSampleCsv = async () => {
    try {
      const res = await fetch(api.getSampleCsvUrl());
      const text = await res.text();
      const a = document.createElement('a');
      a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(text);
      a.download = 'sample_transactions.csv';
      a.click();
    } catch { /* silent */ }
  };

  const fraudCount = result?.predictions.filter(r => r.is_fraud).length ?? 0;
  const totalCount = result?.predictions.length ?? 0;
  const safeCount  = totalCount - fraudCount;

  return (
    <div style={{ padding: '28px 0' }}>
      <div className="grid-2" style={{ alignItems: 'start', marginBottom: 24 }}>

        {/* ── Upload Card ──────────────────────────────────────────────── */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
            <div>
              <h2 style={{ fontSize: 16, fontWeight: 700, color: '#171717', marginBottom: 2 }}>Batch Inference</h2>
              <p style={{ fontSize: 12, color: '#a3a3a3' }}>Upload a CSV to score transactions</p>
            </div>
            <button onClick={handleSampleCsv} className="btn btn-secondary btn-sm" style={{ flexShrink: 0 }}>
              <Download size={12} /><span>Sample CSV</span>
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            {/* Model selector */}
            <div className="input-group">
              <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Cpu size={12} /><span>Model routing</span>
              </label>
              <div style={{ position: 'relative' }}>
                <select
                  className="input-field"
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                >
                  <option value="">Weighted A/B pool (default)</option>
                  {models.map(m => (
                    <option key={m.id} value={m.version_tag}>
                      {m.version_tag}{m.is_active ? ` · Active · ${(m.ab_weight * 100).toFixed(0)}% weight` : ' · Inactive'}
                    </option>
                  ))}
                </select>
                <ChevronDown size={13} style={{ position: 'absolute', right: 10, top: 10, color: '#a3a3a3', pointerEvents: 'none' }} />
              </div>
            </div>

            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => document.getElementById('csv-upload')?.click()}
              style={{
                border: `1.5px dashed ${dragging ? '#171717' : file ? '#16a34a' : '#d4d4d4'}`,
                borderRadius: 10,
                padding: '28px 20px',
                textAlign: 'center',
                background: dragging ? '#fafafa' : file ? '#f0fdf4' : '#fafafa',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                marginBottom: 16,
                position: 'relative',
              }}
            >
              <input id="csv-upload" type="file" accept=".csv" onChange={e => handleFileChange(e.target.files?.[0] ?? null)} style={{ display: 'none' }} />

              <div style={{ display: 'inline-flex', padding: 10, borderRadius: 8, background: file ? '#dcfce7' : '#f5f5f5', border: `1px solid ${file ? '#bbf7d0' : '#e5e5e5'}`, marginBottom: 10, color: file ? '#16a34a' : '#a3a3a3' }}>
                {file ? <FileText size={20} /> : <UploadCloud size={20} />}
              </div>

              {file ? (
                <>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#171717', marginBottom: 2 }}>{file.name}</div>
                  <div style={{ fontSize: 12, color: '#737373' }}>{(file.size / 1024).toFixed(1)} KB · Click to change</div>
                </>
              ) : (
                <>
                  <div style={{ fontSize: 14, fontWeight: 500, color: '#525252', marginBottom: 2 }}>Drop CSV here or click to browse</div>
                  <div style={{ fontSize: 12, color: '#a3a3a3' }}>Features: V1–V28 + Amount column</div>
                </>
              )}

              {file && (
                <button
                  type="button"
                  onClick={e => { e.stopPropagation(); setFile(null); setResult(null); setError(null); }}
                  style={{ position: 'absolute', top: 8, right: 8, background: 'none', border: 'none', cursor: 'pointer', color: '#a3a3a3', padding: 2, display: 'flex', borderRadius: 4 }}
                >
                  <X size={14} />
                </button>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="alert alert-error">
                <AlertTriangle size={13} /><span>{error}</span>
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              className="btn btn-primary btn-lg"
              style={{ width: '100%', justifyContent: 'center' }}
              disabled={loading || !file}
            >
              {loading ? (
                <><RefreshCw size={14} className="spin" style={{ animation: 'spin 0.9s linear infinite' }} /><span>Scoring transactions…</span></>
              ) : (
                <><ShieldAlert size={14} /><span>Run Inference</span></>
              )}
            </button>
          </form>
        </div>

        {/* ── Decision Rules Card ───────────────────────────────────────── */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#171717', marginBottom: 2 }}>Decision Rules</h2>
            <p style={{ fontSize: 12, color: '#a3a3a3' }}>How each transaction is classified and routed</p>
          </div>

          {[
            {
              badge: 'LOW', badgeClass: 'badge-low', label: 'Auto Approve',
              desc: 'Fraud probability below decision threshold. Transaction cleared immediately without analyst review.',
              icon: <CheckCircle2 size={14} color="#16a34a" />,
            },
            {
              badge: 'MEDIUM', badgeClass: 'badge-medium', label: 'Analyst Review',
              desc: 'Anomalous signal detected. Routed to a fraud analyst queue before settlement.',
              icon: <AlertTriangle size={14} color="#d97706" />,
            },
            {
              badge: 'HIGH', badgeClass: 'badge-high', label: 'Elevated Review',
              desc: 'Strong fraud indicators present. Priority queue routing with manual verification required.',
              icon: <AlertTriangle size={14} color="#c2410c" />,
            },
            {
              badge: 'CRITICAL', badgeClass: 'badge-critical', label: 'Immediate Block',
              desc: 'High-confidence fraud pattern. Transaction blocked and account token frozen immediately.',
              icon: <ShieldAlert size={14} color="#dc2626" />,
            },
          ].map(item => (
            <div key={item.badge} style={{ display: 'flex', gap: 12, padding: '12px', borderRadius: 8, border: '1px solid #e5e5e5', background: '#fafafa' }}>
              <div style={{ paddingTop: 1, flexShrink: 0 }}>{item.icon}</div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                  <span className={`badge ${item.badgeClass}`}>{item.badge}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#171717' }}>{item.label}</span>
                </div>
                <p style={{ fontSize: 12, color: '#737373', lineHeight: 1.5 }}>{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Results ───────────────────────────────────────────────────────── */}
      {result && (
        <div className="fade-in">
          {/* KPI row */}
          <div className="grid-4" style={{ marginBottom: 20 }}>
            {[
              { label: 'Batch ID', value: `#${result.batch_id}`, sub: `Model: ${result.summary?.model_version || 'A/B Pool'}`, color: '#171717' },
              { label: 'Scored', value: totalCount, sub: 'Transactions', color: '#171717' },
              { label: 'Flagged', value: fraudCount, sub: `${totalCount > 0 ? ((fraudCount/totalCount)*100).toFixed(1) : '0.0'}% fraud rate`, color: '#dc2626', accent: '#dc2626' },
              { label: 'Approved', value: safeCount, sub: `Threshold: ${result.summary?.threshold_used?.toFixed(3) ?? '—'}`, color: '#16a34a', accent: '#16a34a' },
            ].map(kpi => (
              <div key={kpi.label} className="card" style={{ padding: '16px 20px', borderLeft: kpi.accent ? `3px solid ${kpi.accent}` : undefined }}>
                <div className="stat-label">{kpi.label}</div>
                <div className="stat-value" style={{ color: kpi.color, marginTop: 6 }}>{kpi.value}</div>
                <div className="stat-sub">{kpi.sub}</div>
              </div>
            ))}
          </div>

          {/* Results table */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '14px 20px', borderBottom: '1px solid #e5e5e5', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h3 style={{ fontSize: 14, fontWeight: 600, color: '#171717' }}>Scoring Results</h3>
                <p style={{ fontSize: 12, color: '#a3a3a3' }}>{result.predictions.length} transactions scored</p>
              </div>
              <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#16a34a' }}><CheckCircle2 size={12} /> {safeCount} approved</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#dc2626' }}><AlertTriangle size={12} /> {fraudCount} flagged</span>
              </div>
            </div>

            <div style={{ maxHeight: 460, overflowY: 'auto' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Fraud Probability</th>
                    <th>Risk Level</th>
                    <th>Decision</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map(row => (
                    <tr key={row.row_index}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 500, color: '#171717' }}>{row.row_index + 1}</td>
                      <td><ProbBar value={row.fraud_probability} isFraud={row.is_fraud} /></td>
                      <td>
                        <span className={`badge badge-${row.risk_level.toLowerCase()}`}>{row.risk_level}</span>
                      </td>
                      <td>
                        <span style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em', color: decisionColors[row.decision] ?? '#525252' }}>
                          {row.decision}
                        </span>
                      </td>
                      <td>
                        {row.is_fraud ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#dc2626', fontWeight: 500 }}>
                            <AlertTriangle size={12} />Flagged
                          </span>
                        ) : (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#16a34a', fontWeight: 500 }}>
                            <CheckCircle2 size={12} />Approved
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
