/**
 * FraudGuard Dashboard — Pure JS frontend for the Fraud Detection API.
 *
 * No build step, no framework, no dependencies.
 * Talks to the FastAPI backend at API_BASE.
 */

const API_BASE = "";

// ── State ────────────────────────────────────────────────────────
let token = localStorage.getItem("fg_token") || null;
let currentUser = null;

// ── DOM refs ─────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const authScreen   = $("#auth-screen");
const dashScreen   = $("#dashboard-screen");
const loginForm    = $("#login-form");
const registerForm = $("#register-form");
const authError    = $("#auth-error");

// ── Helpers ──────────────────────────────────────────────────────

async function api(path, opts = {}) {
    const headers = { ...opts.headers };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    if (opts.json) {
        headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(opts.json);
        delete opts.json;
    }
    const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
        const msg = data?.detail || `Error ${res.status}`;
        throw new Error(msg);
    }
    return data;
}

function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 5000);
}

function formatDate(iso) {
    return new Date(iso).toLocaleString("en-IN", {
        day: "2-digit", month: "short", year: "numeric",
        hour: "2-digit", minute: "2-digit",
    });
}

// ── Auth ─────────────────────────────────────────────────────────

// Tab switching
$$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        loginForm.classList.toggle("hidden", btn.dataset.tab !== "login");
        registerForm.classList.toggle("hidden", btn.dataset.tab !== "register");
        authError.classList.add("hidden");
    });
});

loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
        const data = await api("/api/v1/auth/login", {
            method: "POST",
            json: {
                username: $("#login-username").value.trim(),
                email: "placeholder@login.com",
                password: $("#login-password").value,
            },
        });
        token = data.access_token;
        localStorage.setItem("fg_token", token);
        await enterDashboard();
    } catch (err) {
        showError(authError, err.message);
    }
});

registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
        await api("/api/v1/auth/register", {
            method: "POST",
            json: {
                username: $("#reg-username").value.trim(),
                email: $("#reg-email").value.trim(),
                password: $("#reg-password").value,
            },
        });
        // Auto-login after register
        const data = await api("/api/v1/auth/login", {
            method: "POST",
            json: {
                username: $("#reg-username").value.trim(),
                email: $("#reg-email").value.trim(),
                password: $("#reg-password").value,
            },
        });
        token = data.access_token;
        localStorage.setItem("fg_token", token);
        await enterDashboard();
    } catch (err) {
        showError(authError, err.message);
    }
});

async function enterDashboard() {
    try {
        currentUser = await api("/api/v1/auth/me");
    } catch {
        logout();
        return;
    }

    $("#nav-username").textContent = currentUser.username;
    const badge = $("#nav-role");
    badge.textContent = currentUser.role;
    badge.className = `role-badge ${currentUser.role}`;

    // Show/hide admin-only tabs
    $$(".admin-only").forEach((el) => {
        el.style.display = currentUser.role === "admin" ? "" : "none";
    });

    authScreen.classList.remove("active");
    dashScreen.classList.add("active");

    switchSection("predict");
}

function logout() {
    token = null;
    currentUser = null;
    localStorage.removeItem("fg_token");
    dashScreen.classList.remove("active");
    authScreen.classList.add("active");
    loginForm.reset();
    registerForm.reset();
}

$("#logout-btn").addEventListener("click", logout);

// ── Navigation ───────────────────────────────────────────────────

$$(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchSection(btn.dataset.section));
});

function switchSection(name) {
    $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.section === name));
    $$(".section").forEach((s) => s.classList.toggle("active", s.id === `section-${name}`));

    if (name === "history") loadHistory();
    if (name === "models")  loadModels();
    if (name === "metrics") loadMetrics();
}

// ── Predict ──────────────────────────────────────────────────────

const uploadZone = $("#upload-zone");
const csvInput   = $("#csv-input");

uploadZone.addEventListener("click", () => csvInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
csvInput.addEventListener("change", () => { if (csvInput.files.length) handleFile(csvInput.files[0]); });

async function handleFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Please upload a .csv file");
        return;
    }

    $("#predict-results").classList.add("hidden");
    $("#predict-loading").classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const data = await api("/api/v1/fraud/predict", {
            method: "POST",
            body: formData,
        });
        renderPredictions(data);
    } catch (err) {
        alert(`Prediction failed: ${err.message}`);
    } finally {
        $("#predict-loading").classList.add("hidden");
        csvInput.value = "";
    }
}

function renderPredictions(data) {
    $("#stat-total").textContent = data.summary.total_transactions;
    $("#stat-flagged").textContent = data.summary.flagged_fraud;
    $("#stat-threshold").textContent = data.threshold_used.toFixed(2);
    $("#stat-model").textContent = data.model_version || "default";

    // Risk chart
    const riskOrder = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
    const dist = data.summary.risk_distribution;
    const maxCount = Math.max(...riskOrder.map((r) => dist[r] || 0), 1);
    const chartEl = $("#risk-chart");
    chartEl.innerHTML = riskOrder.map((level) => {
        const count = dist[level] || 0;
        const pct = (count / maxCount) * 100;
        return `
            <div class="risk-bar-col">
                <span class="risk-bar-count">${count}</span>
                <div class="risk-bar ${level}" style="height: ${Math.max(pct, 3)}%"></div>
                <span class="risk-bar-label">${level}</span>
            </div>`;
    }).join("");

    // Table
    const tbody = $("#predictions-table tbody");
    tbody.innerHTML = data.predictions.map((p) => `
        <tr>
            <td>${p.row_index}</td>
            <td>${(p.fraud_probability * 100).toFixed(2)}%</td>
            <td class="${p.is_fraud ? "fraud-yes" : "fraud-no"}">${p.is_fraud ? "Yes" : "No"}</td>
            <td><span class="badge badge-${p.risk_level.toLowerCase()}">${p.risk_level}</span></td>
            <td><span class="badge badge-${p.decision}">${p.decision}</span></td>
        </tr>`).join("");

    $("#predict-results").classList.remove("hidden");
}

// ── History ──────────────────────────────────────────────────────

async function loadHistory() {
    try {
        const batches = await api("/api/v1/fraud/history?limit=50");
        const listEl = $("#history-list");
        const emptyEl = $("#history-empty");

        if (!batches.length) {
            listEl.innerHTML = "";
            emptyEl.classList.remove("hidden");
            return;
        }
        emptyEl.classList.add("hidden");

        listEl.innerHTML = batches.map((b) => `
            <div class="history-card glass">
                <div class="history-info">
                    <h4>Batch #${b.id}</h4>
                    <p>${formatDate(b.created_at)}</p>
                </div>
                <div class="history-stats">
                    <div>
                        <div class="history-stat-label">Transactions</div>
                        <div class="history-stat-value">${b.total_transactions}</div>
                    </div>
                    <div>
                        <div class="history-stat-label">Flagged</div>
                        <div class="history-stat-value" style="color: var(--danger)">${b.flagged_fraud}</div>
                    </div>
                    <div>
                        <div class="history-stat-label">Threshold</div>
                        <div class="history-stat-value">${b.threshold_used.toFixed(2)}</div>
                    </div>
                </div>
            </div>`).join("");
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

// ── Models ───────────────────────────────────────────────────────

async function loadModels() {
    try {
        const models = await api("/api/v1/models/");
        const listEl = $("#models-list");
        const emptyEl = $("#models-empty");

        if (!models.length) {
            listEl.innerHTML = "";
            emptyEl.classList.remove("hidden");
            return;
        }
        emptyEl.classList.add("hidden");

        listEl.innerHTML = models.map((m) => `
            <div class="model-card glass">
                <div class="model-info">
                    <h4>${m.version_tag}</h4>
                    <p>${m.description || "No description"} · ${formatDate(m.created_at)}</p>
                </div>
                <div class="model-meta">
                    <span class="weight-pill">Weight: ${m.ab_weight}</span>
                    <span class="active-badge ${m.is_active ? "active" : "inactive"}">
                        ${m.is_active ? "Active" : "Inactive"}
                    </span>
                </div>
            </div>`).join("");
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

// ── Metrics ──────────────────────────────────────────────────────

async function loadMetrics() {
    const contentEl = $("#metrics-content");
    const deniedEl = $("#metrics-denied");

    if (currentUser?.role !== "admin" && currentUser?.role !== "analyst") {
        contentEl.innerHTML = "";
        deniedEl.classList.remove("hidden");
        return;
    }
    deniedEl.classList.add("hidden");

    try {
        const m = await api("/api/v1/metrics");
        
        let html = `
            <div class="stat-card glass">
                <div class="stat-label">Precision</div>
                <div class="stat-value">${(m.model_precision * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">${(m.model_accuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Recall</div>
                <div class="stat-value">${(m.model_recall * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">${(m.model_f1 * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">ROC-AUC</div>
                <div class="stat-value">${(m.model_roc_auc * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Threshold</div>
                <div class="stat-value">${m.threshold}</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Active Models</div>
                <div class="stat-value">${m.active_model_versions.length ? m.active_model_versions.join(", ") : "default"}</div>
            </div>`;

        if (currentUser.role === "admin") {
            html += `
            <div class="stat-card glass">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value">${m.total_predictions.toLocaleString()}</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Batches Run</div>
                <div class="stat-value">${m.total_batches}</div>
            </div>
            <div class="stat-card glass stat-fraud">
                <div class="stat-label">Flagged Fraud</div>
                <div class="stat-value">${m.flagged_fraud}</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Flagged Legit</div>
                <div class="stat-value">${m.flagged_legitimate}</div>
            </div>
            <div class="stat-card glass">
                <div class="stat-label">Uptime</div>
                <div class="stat-value">${formatUptime(m.uptime_seconds)}</div>
            </div>`;
        }

        contentEl.innerHTML = html;
    } catch (err) {
        contentEl.innerHTML = `<div class="empty-state"><p>Failed to load metrics: ${err.message}</p></div>`;
    }
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

// ── Sample CSV download ──────────────────────────────────────────

function downloadSampleCSV() {
    // Hit the backend endpoint that returns real dataset rows
    const a = document.createElement("a");
    a.href = `${API_BASE}/api/v1/sample-csv`;
    a.download = "sample_transactions.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Inject sample download link AFTER (outside) the upload zone
const sampleLink = document.createElement("a");
sampleLink.href = "#";
sampleLink.className = "sample-link";
sampleLink.textContent = "⬇ Download sample CSV to test";
sampleLink.addEventListener("click", (e) => { e.preventDefault(); downloadSampleCSV(); });
uploadZone.parentNode.insertBefore(sampleLink, uploadZone.nextSibling);

// ── Auto-login if token exists ───────────────────────────────────
if (token) {
    enterDashboard();
}
