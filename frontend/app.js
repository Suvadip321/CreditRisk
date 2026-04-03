const form = document.getElementById("riskForm");
const resetBtn = document.getElementById("resetBtn");
const submitBtn = document.getElementById("submitBtn");
const resultBox = document.getElementById("resultBox");
const statusBox = document.getElementById("status");
const apiBaseUrlInput = document.getElementById("apiBaseUrl");
const changeUrlBtn = document.getElementById("changeUrlBtn");
const probabilityValue = document.getElementById("probabilityValue");
const responseDetails = document.querySelector(".response-details");
const DEPLOYED_API_BASE_URL = "https://creditrisk-api-09ps.onrender.com";
const LOCAL_API_BASE_URL = "http://127.0.0.1:8000";

const numericFields = new Set([
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "revol_bal",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "mort_acc",
    "delinq_2yrs",
    "pub_rec",
    "pub_rec_bankruptcies",
    "mths_since_last_delinq",
]);

function setStatus(type, message) {
    statusBox.className = `status ${type}`;
    statusBox.textContent = message;
}

function updateResultSummary(probability) {
    probabilityValue.textContent = Number.isFinite(probability) ? `${(probability * 100).toFixed(1)}%` : "-";
}

function getDefaultApiBaseUrl() {
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "") {
        return LOCAL_API_BASE_URL;
    }
    return DEPLOYED_API_BASE_URL;
}

function toggleUrlEditMode() {
    const isReadonly = apiBaseUrlInput.hasAttribute("readonly");

    if (isReadonly) {
        apiBaseUrlInput.removeAttribute("readonly");
        apiBaseUrlInput.focus();
        apiBaseUrlInput.select();
        changeUrlBtn.textContent = "Apply";
        return;
    }

    const baseUrl = apiBaseUrlInput.value.trim().replace(/\/+$/, "");
    if (!baseUrl) {
        setStatus("error", "Enter a valid API base URL.");
        apiBaseUrlInput.focus();
        return;
    }

    apiBaseUrlInput.value = baseUrl;
    apiBaseUrlInput.setAttribute("readonly", "readonly");
    changeUrlBtn.textContent = "Change";
    setStatus("idle", "API connection updated.");
}

function collectPayload() {
    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        const trimmed = String(value).trim();

        if (!trimmed) {
            continue;
        }

        if (numericFields.has(key)) {
            payload[key] = Number(trimmed);
        } else {
            payload[key] = trimmed;
        }
    }

    return payload;
}

async function predictRisk(event) {
    event.preventDefault();

    const baseUrl = apiBaseUrlInput.value.trim().replace(/\/+$/, "");
    if (!baseUrl) {
        setStatus("error", "Enter a valid API base URL.");
        return;
    }

    const payload = collectPayload();

    setStatus("idle", "Request in progress...");
    resultBox.textContent = "Running model inference...";
    responseDetails.open = false;
    updateResultSummary(NaN);
    submitBtn.disabled = true;

    try {
        const response = await fetch(`${baseUrl}/predict_risk`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            setStatus("error", `Request failed (${response.status})`);
            resultBox.textContent = JSON.stringify(data, null, 2);
            responseDetails.open = true;
            updateResultSummary(NaN);
            return;
        }

        const decision = String(data.decision || "").toUpperCase();
        const probability = Number(data.probability_of_default);
        if (decision === "APPROVE") {
            setStatus("approve", "Decision: APPROVE");
        } else if (decision === "REJECT") {
            setStatus("reject", "Decision: REJECT");
        } else {
            setStatus("idle", "Prediction received.");
        }

        updateResultSummary(probability);
        resultBox.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        setStatus("error", "Could not connect to API.");
        resultBox.textContent = String(error);
        responseDetails.open = true;
        updateResultSummary(NaN);
    } finally {
        submitBtn.disabled = false;
    }
}

function resetForm() {
    form.reset();
    setStatus("idle", "Waiting for prediction...");
    resultBox.textContent = "";
    responseDetails.open = false;
    updateResultSummary(NaN);
}

apiBaseUrlInput.value = getDefaultApiBaseUrl();
form.addEventListener("submit", predictRisk);
resetBtn.addEventListener("click", resetForm);
changeUrlBtn.addEventListener("click", toggleUrlEditMode);
