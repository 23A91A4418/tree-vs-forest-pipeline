// Tab switching logic
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(
        btn => btn.getAttribute('onclick').includes(tabId)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const activeTab = document.getElementById(tabId);
    if (activeTab) activeTab.classList.add('active');
}

// Fetch model info and populate Leaderboard on page load
async function fetchModelInfo() {
    try {
        const response = await fetch('/model/info');
        if (!response.ok) return;

        const data = await response.json();

        // Update header & winner badge
        document.getElementById('championName').innerText = data.champion_model;
        document.getElementById('winnerNameDisplay').innerText = data.champion_model;
        document.getElementById('primaryMetricName').innerText = data.primary_metric.toUpperCase();

        const dt = data.dt_metrics;
        const rf = data.rf_metrics;

        // Metric boxes
        document.getElementById('dtF1').innerText = (dt.f1_score || 0).toFixed(4);
        document.getElementById('rfF1').innerText = (rf.f1_score || 0).toFixed(4);

        // JSON params
        document.getElementById('dtParamsJson').innerText = JSON.stringify(dt.params || {}, null, 2);
        document.getElementById('rfParamsJson').innerText = JSON.stringify(rf.params || {}, null, 2);

        // Leaderboard table
        const tbody = document.getElementById('leaderboardTableBody');
        tbody.innerHTML = '';

        const metricsToCompare = [
            { key: 'f1_score', label: 'F1 Score', format: v => v.toFixed(4) },
            { key: 'accuracy', label: 'Accuracy', format: v => `${(v * 100).toFixed(2)}%` },
            { key: 'precision', label: 'Precision', format: v => v.toFixed(4) },
            { key: 'recall', label: 'Recall', format: v => v.toFixed(4) },
            { key: 'roc_auc', label: 'ROC-AUC Score', format: v => v.toFixed(4) },
            { key: 'avg_latency_ms', label: 'Avg Latency (ms)', format: v => `${v.toFixed(3)} ms` },
        ];

        metricsToCompare.forEach(m => {
            const dtVal = dt[m.key] || 0;
            const rfVal = rf[m.key] || 0;
            const delta = (rfVal - dtVal).toFixed(4);

            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${m.label}</strong></td>
                <td>${m.format(dtVal)}</td>
                <td>${m.format(rfVal)}</td>
                <td><span style="color: ${delta >= 0 ? '#10b981' : '#ef4444'}">${delta > 0 ? '+' : ''}${delta}</span></td>
            `;
            tbody.appendChild(row);
        });

    } catch (err) {
        console.error("Error fetching model info:", err);
    }
}

// Fill sample data into form
function loadSampleData() {
    document.getElementById('age').value = Math.floor(Math.random() * (70 - 22) + 22);
    document.getElementById('annual_income').value = Math.floor(Math.random() * (120000 - 30000) + 30000);
    document.getElementById('credit_score').value = Math.floor(Math.random() * (800 - 450) + 450);
    document.getElementById('tenure_months').value = Math.floor(Math.random() * (60 - 2) + 2);
    document.getElementById('account_balance').value = Math.floor(Math.random() * (80000 - 0) + 0);
    document.getElementById('monthly_usage_hours').value = (Math.random() * (100 - 5) + 5).toFixed(1);
    document.getElementById('support_tickets').value = Math.floor(Math.random() * (10 - 0) + 0);

    const contracts = ['Month-to-Month', 'One-Year', 'Two-Year'];
    document.getElementById('contract_type').value = contracts[Math.floor(Math.random() * contracts.length)];

    const payments = ['Electronic Check', 'Credit Card', 'Bank Transfer'];
    document.getElementById('payment_method').value = payments[Math.floor(Math.random() * payments.length)];

    const devices = ['Mobile', 'Desktop', 'Tablet'];
    document.getElementById('device_category').value = devices[Math.floor(Math.random() * devices.length)];
}

// Predict Submit handler
async function handlePredictSubmit(event) {
    event.preventDefault();

    const payload = {
        age: parseInt(document.getElementById('age').value),
        annual_income: parseFloat(document.getElementById('annual_income').value),
        credit_score: parseInt(document.getElementById('credit_score').value),
        tenure_months: parseInt(document.getElementById('tenure_months').value),
        account_balance: parseFloat(document.getElementById('account_balance').value),
        monthly_usage_hours: parseFloat(document.getElementById('monthly_usage_hours').value),
        support_tickets: parseInt(document.getElementById('support_tickets').value),
        contract_type: document.getElementById('contract_type').value,
        payment_method: document.getElementById('payment_method').value,
        device_category: document.getElementById('device_category').value,
    };

    try {
        const response = await fetch('/predict/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            alert("Prediction failed. Please check form inputs.");
            return;
        }

        const res = await response.json();

        document.getElementById('predictionPlaceholder').classList.add('hidden');
        document.getElementById('predictionResults').classList.remove('hidden');

        // Agreement banner
        const aggBanner = document.getElementById('agreementBanner');
        if (res.agreement) {
            aggBanner.className = 'agreement-banner success';
            aggBanner.innerText = '🤝 Decision Tree & Random Forest AGREE on prediction!';
        } else {
            aggBanner.className = 'agreement-banner warning';
            aggBanner.innerText = '⚠️ Models DISAGREE! Random Forest ensemble refined boundary.';
        }

        // Decision Tree Result
        const dt = res.decision_tree;
        document.getElementById('dtPredBadge').className = `prediction-badge ${dt.is_churn === 1 ? 'churn' : 'retain'}`;
        document.getElementById('dtPredBadge').innerText = dt.prediction_label;
        const dtPct = (dt.churn_probability * 100).toFixed(1);
        document.getElementById('dtProbVal').innerText = `${dtPct}%`;
        document.getElementById('dtProbFill').style.width = `${dtPct}%`;

        // Random Forest Result
        const rf = res.random_forest;
        document.getElementById('rfPredBadge').className = `prediction-badge ${rf.is_churn === 1 ? 'churn' : 'retain'}`;
        document.getElementById('rfPredBadge').innerText = rf.prediction_label;
        const rfPct = (rf.churn_probability * 100).toFixed(1);
        document.getElementById('rfProbVal').innerText = `${rfPct}%`;
        document.getElementById('rfProbFill').style.width = `${rfPct}%`;

        // Champion Decision
        const champ = res.champion;
        document.getElementById('champDecisionText').innerHTML = `
            Winner Model: <strong>${champ.model_name}</strong> &bull; Verdict: 
            <span style="color: ${champ.is_churn === 1 ? '#f87171' : '#34d399'}">${champ.prediction_label} (${(champ.churn_probability * 100).toFixed(1)}%)</span>
        `;

    } catch (err) {
        console.error("Prediction error:", err);
    }
}

// Sample batch run
async function runSampleBatch() {
    const sampleRecords = Array.from({ length: 10 }).map(() => ({
        age: Math.floor(Math.random() * (70 - 22) + 22),
        annual_income: Math.floor(Math.random() * (120000 - 30000) + 30000),
        credit_score: Math.floor(Math.random() * (800 - 450) + 450),
        tenure_months: Math.floor(Math.random() * (60 - 2) + 2),
        account_balance: Math.floor(Math.random() * (80000 - 0) + 0),
        monthly_usage_hours: parseFloat((Math.random() * (100 - 5) + 5).toFixed(1)),
        support_tickets: Math.floor(Math.random() * (10 - 0) + 0),
        contract_type: ['Month-to-Month', 'One-Year', 'Two-Year'][Math.floor(Math.random() * 3)],
        payment_method: ['Electronic Check', 'Credit Card', 'Bank Transfer'][Math.floor(Math.random() * 3)],
        device_category: ['Mobile', 'Desktop', 'Tablet'][Math.floor(Math.random() * 3)],
    }));

    try {
        const response = await fetch('/predict/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ records: sampleRecords }),
        });

        const res = await response.json();

        document.getElementById('batchStats').classList.remove('hidden');
        document.getElementById('batchTotal').innerText = res.total_records;
        document.getElementById('batchChurn').innerText = res.churn_count;
        document.getElementById('batchRetain').innerText = res.retention_count;
        document.getElementById('batchRate').innerText = `${((res.churn_count / res.total_records) * 100).toFixed(1)}%`;

        const tbody = document.getElementById('batchTableBody');
        tbody.innerHTML = '';

        res.predictions.forEach((pred, idx) => {
            const rec = sampleRecords[idx];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>#${idx + 1}</td>
                <td>${rec.age} y/o</td>
                <td>$${rec.annual_income.toLocaleString()}</td>
                <td>${rec.tenure_months} m</td>
                <td>${rec.support_tickets}</td>
                <td>${rec.contract_type}</td>
                <td><strong>${(pred.churn_probability * 100).toFixed(1)}%</strong></td>
                <td><span style="color: ${pred.is_churn === 1 ? '#ef4444' : '#10b981'}; font-weight: bold;">${pred.prediction_label}</span></td>
            `;
            tbody.appendChild(row);
        });

    } catch (err) {
        console.error("Batch error:", err);
    }
}

// Trigger bake-off button
async function triggerBakeoff() {
    const btn = document.getElementById('runBakeoffBtn');
    btn.disabled = true;
    btn.innerText = '⏳ Training ML Models...';

    try {
        const response = await fetch('/bakeoff/run', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert(`🎉 ML Bake-Off Complete! Winning Model: ${data.summary.champion_model}`);
            fetchModelInfo();
        } else {
            alert("Bake-off execution failed.");
        }
    } catch (err) {
        alert("Error running bake-off.");
    } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Run ML Bake-Off';
    }
}

// Init on load
document.addEventListener('DOMContentLoaded', () => {
    fetchModelInfo();
});
