import React, { useState, useEffect } from 'react';
import { UploadCloud, FileSpreadsheet, Download, CheckCircle2, AlertTriangle, ShieldAlert, Cpu, RefreshCw } from 'lucide-react';
import { api, type ModelVersion, type PredictResponse } from '../api';

export const PredictTab: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const list = await api.getModels();
      setModels(list);
    } catch (e) {
      console.error(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file first');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.predict(file, selectedModel || undefined);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSample = async () => {
    try {
      const res = await fetch(api.getSampleCsvUrl());
      const text = await res.text();
      const blob = new Blob([text], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.setAttribute('download', 'sample_transactions.csv');
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 5000);
    } catch (e) {
      console.error('Failed to download sample CSV:', e);
    }
  };

  return (
    <div style={{ padding: '1.5rem 0' }}>
      <div className="grid-2" style={{ alignItems: 'start', marginBottom: '2rem' }}>
        {/* Upload Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '0.2rem', color: '#f4f4f5' }}>Batch Fraud Inference</h3>
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>Upload transaction CSVs for XGBoost scoring</p>
            </div>
            <button
              type="button"
              onClick={handleDownloadSample}
              className="btn btn-secondary"
              style={{ fontSize: '0.8125rem', padding: '0.4rem 0.875rem' }}
            >
              <Download size={14} />
              <span>Sample CSV</span>
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            {/* Model Selector */}
            <div className="input-group" style={{ marginBottom: '1.25rem' }}>
              <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Cpu size={14} color="var(--text-secondary)" />
                <span>Model Routing</span>
              </label>
              <select
                className="input-field"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ cursor: 'pointer' }}
              >
                <option value="">-- Weighted A/B Pool (Default) --</option>
                {models.map((m) => (
                  <option key={m.id} value={m.version_tag}>
                    {m.version_tag} {m.is_active ? `(Active | Weight: ${(m.ab_weight * 100).toFixed(0)}%)` : '(Inactive)'}
                  </option>
                ))}
              </select>
            </div>

            {/* Drag Drop Area */}
            <div
              style={{
                border: '1px dashed var(--border-hover)',
                borderRadius: 'var(--radius-md)',
                padding: '2rem 1rem',
                textAlign: 'center',
                background: 'var(--bg-input)',
                cursor: 'pointer',
                marginBottom: '1.25rem',
                transition: 'all 0.15s ease'
              }}
              onClick={() => document.getElementById('csv-upload')?.click()}
            >
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <div style={{ display: 'inline-flex', padding: '0.75rem', borderRadius: '8px', background: 'var(--bg-elevated)', marginBottom: '0.75rem', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}>
                {file ? <FileSpreadsheet size={24} /> : <UploadCloud size={24} />}
              </div>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.25rem', color: '#f4f4f5' }}>
                {file ? file.name : 'Select or drop CSV file'}
              </h4>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {file ? `${(file.size / 1024).toFixed(1)} KB selected` : 'Standard PCA features V1-V28 + Amount'}
              </p>
            </div>

            {error && (
              <div style={{
                background: 'var(--status-danger-bg)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
                color: 'var(--status-danger)',
                padding: '0.625rem 0.875rem',
                borderRadius: 'var(--radius-md)',
                marginBottom: '1rem',
                fontSize: '0.8125rem'
              }}>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.625rem', justifyContent: 'center' }}
              disabled={loading || !file}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Processing Batch...</span>
                </>
              ) : (
                <>
                  <ShieldAlert size={16} />
                  <span>Run Inference</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Instructions / Overview Card */}
        <div className="card">
          <h3 style={{ fontSize: '1.1rem', marginBottom: '0.75rem', color: '#f4f4f5' }}>Inference Rules & Routing</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: '1.5' }}>
            Transactions are evaluated using XGBoost gradient boosted decision trees. Features are scaled via trained joblib artifact pipelines and checked against decision thresholds.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', background: 'var(--bg-input)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div className="badge badge-low">LOW</div>
              <div>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#f4f4f5' }}>Automated Approval</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Probability below decision boundary. Settled immediately without analyst queue routing.</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', background: 'var(--bg-input)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div className="badge badge-medium">MEDIUM / HIGH</div>
              <div>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#f4f4f5' }}>Analyst Review</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Anomalous signature detected. Routed to fraud analysts for verification before clearing.</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', background: 'var(--bg-input)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div className="badge badge-critical">CRITICAL</div>
              <div>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#f4f4f5' }}>Immediate Block</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>High-confidence fraud pattern. Transaction blocked and token frozen.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div>
          {/* Summary KPIs */}
          <div className="grid-4" style={{ marginBottom: '1.25rem' }}>
            <div className="card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Batch ID</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem' }}>#{result.batch_id}</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--brand-primary)' }}>Model: {result.summary.model_version || 'A/B Pool'}</span>
            </div>

            <div className="card" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Transactions Scored</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#f4f4f5', marginTop: '0.2rem' }}>{result.summary.total_transactions}</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Latency: &lt;50ms</span>
            </div>

            <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--status-danger)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Flagged Fraud</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--status-danger)', marginTop: '0.2rem' }}>{result.summary.flagged_fraud}</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--status-danger)' }}>Rate: {(result.summary.fraud_rate * 100).toFixed(2)}%</span>
            </div>

            <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--status-success)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Approved</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--status-success)', marginTop: '0.2rem' }}>{result.summary.flagged_legitimate}</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--status-success)' }}>Threshold: {result.summary.threshold_used}</span>
            </div>
          </div>

          {/* Table */}
          <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
            <div style={{ padding: '1rem 1.25rem', background: 'var(--bg-input)', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <h3 style={{ fontSize: '0.95rem', margin: 0, color: '#f4f4f5' }}>Scoring Results</h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Showing all {result.predictions.length} rows</span>
            </div>

            <div className="table-container" style={{ border: 'none', maxHeight: '450px', overflowY: 'auto' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Fraud Probability</th>
                    <th>Severity Level</th>
                    <th>Decision</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((row) => (
                    <tr key={row.row_index}>
                      <td style={{ fontWeight: 500, color: '#f4f4f5', fontFamily: 'var(--font-mono)' }}>{row.row_index + 1}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                          <div style={{ width: '60px', height: '4px', background: 'var(--bg-input)', borderRadius: '2px', overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(100, row.fraud_probability * 100)}%`,
                              height: '100%',
                              background: row.is_fraud ? 'var(--status-danger)' : 'var(--status-success)'
                            }} />
                          </div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: row.is_fraud ? 'var(--status-danger)' : '#f4f4f5' }}>
                            {(row.fraud_probability * 100).toFixed(2)}%
                          </span>
                        </div>
                      </td>
                      <td>
                        <span className={`badge badge-${row.risk_level.toLowerCase()}`}>
                          {row.risk_level}
                        </span>
                      </td>
                      <td style={{ textTransform: 'uppercase', fontWeight: 600, fontSize: '0.75rem', color: row.decision === 'block' ? 'var(--status-danger)' : row.decision === 'review' ? 'var(--status-warning)' : 'var(--status-success)' }}>
                        {row.decision}
                      </td>
                      <td>
                        {row.is_fraud ? (
                          <span style={{ color: 'var(--status-danger)', display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8125rem' }}>
                            <AlertTriangle size={14} /> Flagged
                          </span>
                        ) : (
                          <span style={{ color: 'var(--status-success)', display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8125rem' }}>
                            <CheckCircle2 size={14} /> Approved
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
