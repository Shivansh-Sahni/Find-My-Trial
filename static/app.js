const bootstrap = JSON.parse(document.getElementById("bootstrap-data").textContent);

const shell = document.querySelector(".page-shell");
const form = document.getElementById("match-form");
const patientText = document.getElementById("patient-text");
const patientFile = document.getElementById("patient-file");
const fileName = document.getElementById("file-name");
const loadSampleButton = document.getElementById("load-sample");
const statusText = document.getElementById("status-text");
const submitButton = document.getElementById("submit-button");
const patientSummary = document.getElementById("patient-summary");
const diagnosisSignals = document.getElementById("diagnosis-signals");
const biomarkerSignals = document.getElementById("biomarker-signals");
const therapySignals = document.getElementById("therapy-signals");
const contextSignals = document.getElementById("context-signals");
const resultsTitle = document.getElementById("results-title");
const resultsMeta = document.getElementById("results-meta");
const resultsGrid = document.getElementById("results-grid");
const emptyState = document.getElementById("empty-state");

patientFile.addEventListener("change", () => {
  fileName.textContent = patientFile.files?.[0]?.name || "No file attached";
});

loadSampleButton.addEventListener("click", async () => {
  setStatus("Loading demo patient chart...");
  try {
    const response = await fetch("/api/sample-patient");
    const payload = await response.json();
    patientText.value = payload.text || "";
    setStatus("Demo patient loaded. Run the matcher when you're ready.");
  } catch (error) {
    setStatus("Could not load the sample chart.");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  setLoading(true);
  setStatus("Parsing chart, extracting signals, and ranking trials...");

  try {
    const response = await fetch("/api/match", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Match request failed.");
    }
    renderPayload(payload);
    setStatus("Match run complete.");
  } catch (error) {
    renderError(error.message);
    setStatus(error.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  shell.classList.toggle("is-loading", isLoading);
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Matching..." : "Run Trial Match";
}

function setStatus(message) {
  statusText.textContent = message;
}

function renderPayload(payload) {
  renderPatient(payload.patient);
  renderResults(payload.meta, payload.results);
}

function renderPatient(patient) {
  const summary = patient.summary || "No structured patient synopsis was generated.";
  patientSummary.classList.remove("empty-card");
  patientSummary.innerHTML = `
    <div class="summary-lead">${escapeHtml(summary)}</div>
    <div class="chip-cloud summary-meta">
      ${renderChip(patient.age ? `${patient.age} years` : "Age not parsed", "is-soft")}
      ${renderChip(patient.sex || "Sex not parsed", "is-soft")}
      ${renderChip(patient.stage || (patient.metastatic ? "Metastatic context" : "Stage not parsed"), "is-accent")}
      ${renderChip(patient.performance_status || "Performance status not parsed", "is-teal")}
    </div>
  `;

  diagnosisSignals.innerHTML = renderChipCloud(patient.diagnoses, "No diagnoses extracted.");
  biomarkerSignals.innerHTML = renderChipCloud(patient.biomarkers, "No biomarkers extracted.");
  therapySignals.innerHTML = renderChipCloud(patient.therapies, "No therapy history extracted.");

  const context = [
    patient.metastatic ? "Metastatic / advanced disease context" : null,
    ...(patient.metastatic_sites || []).map((site) => `${site} involvement`),
    ...(patient.comorbidities || []),
    ...(patient.location_hints || []),
  ].filter(Boolean);

  contextSignals.innerHTML = renderChipCloud(context, "No extra clinical context extracted.");
}

function renderResults(meta, results) {
  if (!results.length) {
    resultsGrid.innerHTML = "";
    emptyState.style.display = "block";
    resultsTitle.textContent = "Top Trial Matches";
    resultsMeta.textContent = meta.fallback_hint || "Try broader chart text or loosen one of the filters.";
    return;
  }

  emptyState.style.display = "none";
  const semanticLine = meta.semantic_used ? "semantic reranking active" : "lexical and structured ranking only";
  resultsTitle.textContent = `Top ${results.length} Trial Matches`;
  resultsMeta.textContent = `${meta.candidate_count} candidates cleared the current filters; ${semanticLine}.`;

  resultsGrid.innerHTML = results
    .map((result, index) => renderTrialCard(result, index))
    .join("");
}

function renderTrialCard(result, index) {
  const fitClass =
    result.fit_label === "High Conviction"
      ? "is-high"
      : result.fit_label === "Promising"
        ? "is-mid"
        : "is-low";

  const reasons = result.reasons.length
    ? result.reasons.map((reason) => `<div class="reason-item">${escapeHtml(reason)}</div>`).join("")
    : `<div class="reason-item">This match surfaced primarily from overall textual and clinical similarity.</div>`;

  const cautions = result.cautions.length
    ? `<div class="caution-list">${result.cautions
        .map((caution) => `<div class="caution-item">${escapeHtml(caution)}</div>`)
        .join("")}</div>`
    : "";

  const breakdownRows = Object.entries(result.breakdown)
    .filter(([, value]) => value > 0)
    .map(
      ([label, value]) => `
        <div class="breakdown-row">
          <span>${escapeHtml(label)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.min(value, 100)}%"></div></div>
          <span>${value.toFixed(0)}</span>
        </div>
      `,
    )
    .join("");

  return `
    <article class="trial-card" style="animation-delay:${index * 55}ms">
      <div class="trial-header">
        <div class="trial-header-main">
          <div class="trial-rank">
            <span>#${result.rank}</span>
            <span class="score-pill">${result.score.toFixed(1)} match</span>
            <span class="fit-pill ${fitClass}">${escapeHtml(result.fit_label)}</span>
          </div>
          <h3 class="trial-title">
            <a href="${escapeHtml(result.url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(result.title)}</a>
          </h3>
        </div>
        <div class="trial-meta">
          ${renderMetaPill(result.status)}
          ${renderMetaPill(result.phase)}
          ${renderMetaPill(result.study_type)}
          ${renderMetaPill(result.nct_number)}
        </div>
      </div>

      <div class="trial-summary">${escapeHtml(result.brief_summary || "No summary available for this trial row.")}</div>

      <div class="section-block">
        <div class="section-title">Matched Signals</div>
        <div class="chip-cloud">
          ${renderArrayAsChips(result.matched_conditions, "is-teal")}
          ${renderArrayAsChips(result.matched_biomarkers, "is-accent")}
          ${renderArrayAsChips(result.matched_therapies, "is-soft")}
        </div>
      </div>

      <div class="section-block">
        <div class="section-title">Why It Ranked</div>
        <div class="reason-list">${reasons}</div>
      </div>

      ${cautions ? `
      <div class="section-block">
        <div class="section-title">Cautions</div>
        ${cautions}
      </div>` : ""}

      <div class="section-block">
        <div class="section-title">Score Breakdown</div>
        <div class="breakdown">${breakdownRows}</div>
      </div>

      <div class="trial-footer">
        <div class="chip-cloud">
          ${renderArrayAsChips(result.conditions, "is-soft")}
          ${renderArrayAsChips(result.interventions, "is-soft")}
          ${renderArrayAsChips(result.locations, "is-soft")}
        </div>

        <div class="eligibility-list">
          ${renderMetaPill(`Sponsor: ${result.sponsor || "Unknown"}`)}
          ${renderMetaPill(`Sex: ${result.eligibility.sex}`)}
          ${renderMetaPill(`Age: ${result.eligibility.age}`)}
        </div>
      </div>
    </article>
  `;
}

function renderChipCloud(items, emptyMessage) {
  if (!items || !items.length) {
    return emptyMessage;
  }
  return items.map((item, index) => renderChip(item, index % 2 === 0 ? "is-soft" : "")).join("");
}

function renderArrayAsChips(items, variant) {
  if (!items || !items.length) {
    return "";
  }
  return items.slice(0, 4).map((item) => renderChip(item, variant)).join("");
}

function renderChip(text, variant = "") {
  return `<span class="chip ${variant}">${escapeHtml(text)}</span>`;
}

function renderMetaPill(text) {
  return `<span class="meta-pill">${escapeHtml(text)}</span>`;
}

function renderError(message) {
  emptyState.style.display = "block";
  resultsGrid.innerHTML = "";
  resultsTitle.textContent = "Match request failed.";
  resultsMeta.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
