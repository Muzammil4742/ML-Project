// ---------- index.js (fixed version) ----------
let uploadedFilePath = "";

// Helper: show alert and return
function showAlert(msg) { alert(msg); }

// ------------------ Upload ------------------
document.getElementById("uploadBtn").addEventListener("click", async () => {
    const file = document.getElementById("csvFile").files[0];
    if (!file) return showAlert("Select a file first");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) return showAlert("Upload error: " + data.error);
    uploadedFilePath = data.path;
    showAlert("File uploaded: " + uploadedFilePath);

    // After upload, show data types (if backend route exists)
});

document.getElementById("dtypeBtn").addEventListener("click", showDataTypes);


// ------------------ Clean ------------------
document.getElementById("cleanBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");
    const res = await fetch("/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: uploadedFilePath })
    });
    const data = await res.json();
    if (data.error) return showAlert("Clean error: " + data.error);
    uploadedFilePath = data.path;
    showAlert("Dataset cleaned: " + uploadedFilePath);

    // refresh dtypes
    await showDataTypes();
});

// ------------------ Normalize ------------------
document.getElementById("normalBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");
    const res = await fetch("/normalize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: uploadedFilePath })
    });
    const data = await res.json();
    if (data.error) return showAlert("Normalize error: " + data.error);
    uploadedFilePath = data.path;
    showAlert("Dataset normalized: " + uploadedFilePath);

    // refresh dtypes
    await showDataTypes();
});

// ------------------ Show Data Types ------------------
async function showDataTypes() {
    const res = await fetch("/dtypes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: uploadedFilePath })
    });
    const data = await res.json();

    const dtDiv = document.getElementById("dtypeResult");
    let html = "<h3>Data Types:</h3><table border='1'><tr><th>Column</th><th>Type</th></tr>";
    for (let col in data.data_types) {
        html += `<tr><td>${col}</td><td>${data.data_types[col]}</td></tr>`;
    }
    html += "</table>";
    dtDiv.innerHTML = html;
}

// wire dtype button
document.getElementById("dtypeBtn").addEventListener("click", showDataTypes);

// ------------------ Correlation ------------------
document.getElementById("corrBtn").addEventListener("click", async () => {
    const col1 = document.getElementById("corrCol1").value.trim();
    const col2 = document.getElementById("corrCol2").value.trim();
    const out = document.getElementById("corrResult");
    out.innerHTML = "";

    if (!uploadedFilePath) return showAlert("Upload a file first");
    if (!col1 || !col2) return showAlert("Provide two column names");

    try {
        const res = await fetch("/correlation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: uploadedFilePath, col1, col2 })
        });
        const data = await res.json();
        if (res.status !== 200) {
            out.innerHTML = `<pre>Error: ${data.error || JSON.stringify(data)}</pre>`;
            return;
        }
        // safe conversion and formatting
        const corrVal = Number(data.correlation);
        if (Number.isNaN(corrVal)) {
            out.innerHTML = `<pre>Correlation: invalid result</pre>`;
            return;
        }
        const corrStr = corrVal.toFixed(3);
        const absv = Math.abs(corrVal);
        const quality = absv > 0.7 ? "Strong" : absv > 0.4 ? "Moderate" : "Weak";
        out.innerHTML = `<pre>Correlation (${col1} vs ${col2}): ${corrStr} — ${quality}</pre>`;
    } catch (e) {
        out.innerHTML = `<pre>Request error: ${e.message}</pre>`;
    }
});

// ------------------ Graph ------------------
document.getElementById("graphBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");
    const x = document.getElementById("xCol").value.trim();
    const y = document.getElementById("yCol").value.trim();
    const graph_type = document.getElementById("graphType").value;
    const out = document.getElementById("graphResult");
    out.innerHTML = "";

    if (!graph_type) return showAlert("Select a graph type");
    if (graph_type !== "heatmap" && (!x || !y)) return showAlert("Provide X and Y columns");

    try {
        const res = await fetch("/graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: uploadedFilePath, x, y, graph_type })
        });
        const data = await res.json();
        if (!res.ok) {
            out.innerHTML = `<pre>Graph error: ${data.error || JSON.stringify(data)}</pre>`;
            return;
        }
        // graph_path is a server filepath; use it as src (Flask serves static/graphs route)
        // if response gives full path, convert to relative '/static/graphs/filename.png'
        let imgPath = data.graph_path;
        // if backend returned absolute path, keep only filename
        const fname = imgPath.split("/").pop();
        imgPath = `/static/graphs/${fname}`;

        out.innerHTML = `<img src="${imgPath}" alt="Graph" style="width:100%;max-height:420px;object-fit:contain;border-radius:8px;">`;
    } catch (e) {
        out.innerHTML = `<pre>Request error: ${e.message}</pre>`;
    }
});

// ------------------ Train Model ------------------

document.getElementById("trainBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");

    const featuresRaw = document.getElementById("featuresInput").value.trim();
    const target = document.getElementById("targetInput").value.trim();
    const model_type = document.getElementById("modelType").value;
    // For initial training we send eval_type empty (or you can set)
    const eval_type = ""; // no evaluation on plain Train button

    if (!featuresRaw || !target || !model_type) return showAlert("Provide features, target, and model type");

    const features = featuresRaw.split(",").map(s => s.trim()).filter(s => s);

    const out = document.getElementById("results");
    out.innerHTML = "<pre>Training... please wait</pre>";

    try {
        const res = await fetch("/train_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: uploadedFilePath, features, target, model_type, eval_type })
        });
        const data = await res.json();
        if (!res.ok) {
            out.innerHTML = `<pre>Train error: ${JSON.stringify(data)}</pre>`;
            return;
        }

        // train_model returns correlation_output (per your app.py)
        // out.innerHTML = `<h4>Feature correlations</h4><pre>${JSON.stringify(data, null, 2)}</pre>`;

        // Populate predictInputs so the user can enter values for each feature
        const predictInputs = document.getElementById("predictInputs");
        predictInputs.innerHTML = "";
        features.forEach((f, idx) => {
            const wrapper = document.createElement("div");
            wrapper.style.marginBottom = "8px";
            wrapper.innerHTML = `<label>${f}:</label><input type="text" data-feature="${f}" class="predict-field" placeholder="value for ${f}" style="width:100%;padding:8px;margin-top:4px;background:#2f303d;border-radius:8px;border:1px solid rgba(0,234,255,0.1);color:#fff;">`;
            predictInputs.appendChild(wrapper);
        });

    } catch (e) {
        out.innerHTML = `<pre>Request error: ${e.message}</pre>`;
    }
});

// ------------------ Predict ------------------
document.getElementById("predictBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");

    // collect features used (from results in predictInputs)
    const fields = Array.from(document.querySelectorAll("#predictInputs .predict-field"));
    if (fields.length === 0) return showAlert("Train model first to enable prediction (or add /predict route on server).");

    const values = [];
    const features = [];
    for (const el of fields) {
        const val = el.value.trim();
        if (val === "") return showAlert("Provide all feature values");
        values.push(Number(val));
        features.push(el.getAttribute("data-feature"));
    }

    // Call /predict if route exists
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: uploadedFilePath, features, target: document.getElementById("targetInput").value.trim(), model_type: document.getElementById("modelType").value, predict_values: values })
        });
        const data = await res.json();
        const out = document.getElementById("predictResult");
        if (!res.ok) {
            out.innerHTML = `<pre>Predict error: ${JSON.stringify(data)}</pre>`;
            return;
        }
        let evalHtml = "";
        if (data.evaluation) {
            evalHtml = `<pre>R²: ${data.evaluation.r2} | MSE: ${data.evaluation.mse}</pre>`;
        }
        out.innerHTML = `<pre>Prediction: ${data.prediction}</pre>${evalHtml}`;
    } catch (e) {
        // likely /predict route missing
        document.getElementById("predictResult").innerHTML = `<pre>Prediction failed: ${e.message}. Make sure /predict route exists in app.py</pre>`;
    }
});

// ------------------ Evaluate Model ------------------
document.getElementById("evalBtn").addEventListener("click", async () => {
    if (!uploadedFilePath) return showAlert("Upload a file first");

    const featuresRaw = document.getElementById("featuresInput").value.trim();
    const target = document.getElementById("targetInput").value.trim();
    const model_type = document.getElementById("modelType").value;
    const evalType = document.getElementById("evalType").value;

    if (!featuresRaw || !target || !model_type || !evalType) return showAlert("Provide features, target, model and evaluation type");

    const features = featuresRaw.split(",").map(s => s.trim()).filter(s => s);
    const body = { path: uploadedFilePath, features, target, model_type, eval_type: evalType };

    // add eval options
    if (evalType === "split") {
        const testSize = parseFloat(document.getElementById("testSize") ? document.getElementById("testSize").value : 0.2);
        body.test_size = testSize;
    } else if (evalType === "kfold") {
        const k = parseInt(document.getElementById("kfoldK") ? document.getElementById("kfoldK").value : 5, 10);
        body.k = k;
    }

    const out = document.getElementById("evalResult");
    out.innerHTML = "<pre>Evaluating...</pre>";

    try {
        const res = await fetch("/train_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) {
            out.innerHTML = `<pre>Evaluate error: ${JSON.stringify(data)}</pre>`;
            return;
        }
        out.innerHTML = `<h4>Evaluation Results</h4><pre>${JSON.stringify(data, null, 2)}</pre>`;
    } catch (e) {
        out.innerHTML = `<pre>Request error: ${e.message}</pre>`;
    }
});

// ------------------ evalType dynamic options (attach once) ------------------
document.getElementById("evalType").addEventListener("change", () => {
    const evalType = document.getElementById("evalType").value;
    const evalOptionsDiv = document.getElementById("evalOptions");
    evalOptionsDiv.innerHTML = "";
    if (evalType === "split") {
        evalOptionsDiv.innerHTML = `<label>Test Size (0-1):</label>
            <input type="number" id="testSize" min="0.1" max="0.9" step="0.05" value="0.2">`;
    } else if (evalType === "kfold") {
        evalOptionsDiv.innerHTML = `<label>Number of folds (k):</label>
            <input type="number" id="kfoldK" min="2" max="20" step="1" value="5">`;
    }
});
