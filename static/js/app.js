/**
 * Dataset Validator — Frontend Application
 * Uses XMLHttpRequest for upload (real progress bar, no timeout on large files).
 * fetch() used for all other API calls.
 */
"use strict";

const state = {
  sessionId: null, filename: null, issues: [],
  summary: {}, shape: null, activeFilter: "ALL", fixedIds: new Set(),
};

const screens = {
  upload:  document.getElementById("upload-screen"),
  loading: document.getElementById("loading-screen"),
  report:  document.getElementById("report-screen"),
};

const el = {
  dropZone:     document.getElementById("drop-zone"),
  fileInput:    document.getElementById("file-input"),
  browseBtn:    document.getElementById("browse-btn"),
  loaderText:   document.getElementById("loader-text"),
  loaderSteps:  document.getElementById("loader-steps"),
  progressBar:  document.getElementById("progress-bar"),
  progressWrap: document.getElementById("progress-wrap"),
  fileNameDisp: document.getElementById("file-name-display"),
  fileMeta:     document.getElementById("file-meta"),
  summaryGrid:  document.getElementById("summary-grid"),
  sectionNav:   document.getElementById("section-nav"),
  issuesList:   document.getElementById("issues-list"),
  filterBtns:   document.querySelectorAll(".filter-btn"),
  downloadBtn:  document.getElementById("download-btn"),
  previewBtn:   document.getElementById("preview-btn"),
  backBtn:      document.getElementById("back-btn"),
  previewModal: document.getElementById("preview-modal"),
  modalClose:   document.getElementById("modal-close"),
  previewWrap:  document.getElementById("preview-table-wrap"),
  toast:        document.getElementById("toast"),
};

// ── Screen ────────────────────────────────────────────────────
function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove("active"));
  screens[name].classList.add("active");
}

// ── Toast ─────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "success") {
  el.toast.textContent = msg;
  el.toast.className = `toast ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 4000);
}

// ── Loader ────────────────────────────────────────────────────
const VALIDATION_STEPS = [
  "Checking structure…","Inspecting data types…","Counting missing values…",
  "Hunting for duplicates…","Validating numerics…","Cleaning categories…",
  "Parsing dates…","Verifying keys…","Checking business logic…","Assembling report…",
];
let loaderInterval = null, loaderIdx = 0;

function startValidationLoader() {
  el.progressWrap.classList.add("hidden");
  el.loaderSteps.innerHTML = "";
  loaderIdx = 0;
  el.loaderText.textContent = VALIDATION_STEPS[0];
  loaderInterval = setInterval(() => {
    loaderIdx++;
    if (loaderIdx < VALIDATION_STEPS.length) {
      el.loaderText.textContent = VALIDATION_STEPS[loaderIdx];
      const step = document.createElement("div");
      step.className = "loader-step done";
      step.textContent = "✓  " + VALIDATION_STEPS[loaderIdx - 1];
      el.loaderSteps.appendChild(step);
    }
  }, 700);
}

function stopLoader() { clearInterval(loaderInterval); }

// ── Upload via XHR ────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showToast("Please upload a .csv file", "error"); return;
  }
  state.filename = file.name;
  showScreen("loading");
  el.progressWrap.classList.remove("hidden");
  el.loaderSteps.innerHTML = "";
  el.loaderText.textContent = "Uploading file…";
  setProgress(0);

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener("progress", e => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      setProgress(pct);
      el.loaderText.textContent =
        `Uploading… ${pct}% (${fmtBytes(e.loaded)} / ${fmtBytes(e.total)})`;
    }
  });

  xhr.upload.addEventListener("load", () => {
    setProgress(100);
    el.loaderText.textContent = "File received. Running validation…";
    startValidationLoader();
  });

  xhr.addEventListener("load", () => {
    stopLoader();
    if (xhr.status !== 200) {
      showToast(`Server error ${xhr.status}. Check terminal.`, "error");
      showScreen("upload"); return;
    }
    let data;
    try { data = JSON.parse(xhr.responseText); }
    catch {
      showToast("Unexpected server response. Check terminal.", "error");
      console.error(xhr.responseText.slice(0, 500));
      showScreen("upload"); return;
    }
    if (data.error) { showToast(data.error, "error"); showScreen("upload"); return; }
    state.sessionId = data.session_id;
    state.issues    = data.issues || [];
    state.summary   = data.summary || {};
    state.shape     = data.shape;
    state.fixedIds  = new Set();
    renderReport(data);
    showScreen("report");
  });

  xhr.addEventListener("error", () => {
    stopLoader(); showToast("Network error. Is the server running?", "error"); showScreen("upload");
  });

  xhr.open("POST", "/api/upload");
  xhr.send(form);
}

function setProgress(pct) { if (el.progressBar) el.progressBar.style.width = pct + "%"; }
function fmtBytes(b) {
  return b < 1024*1024 ? (b/1024).toFixed(1)+" KB" : (b/(1024*1024)).toFixed(1)+" MB";
}

// ── Drop zone ─────────────────────────────────────────────────
el.dropZone.addEventListener("click", e => { if (e.target !== el.browseBtn) el.fileInput.click(); });
el.browseBtn.addEventListener("click", e => { e.stopPropagation(); el.fileInput.click(); });
el.fileInput.addEventListener("change", e => { handleFile(e.target.files[0]); e.target.value = ""; });
el.dropZone.addEventListener("dragover", e => { e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone.addEventListener("dragleave", () => el.dropZone.classList.remove("drag-over"));
el.dropZone.addEventListener("drop", e => {
  e.preventDefault(); el.dropZone.classList.remove("drag-over"); handleFile(e.dataTransfer.files[0]);
});

// ── Render ────────────────────────────────────────────────────
function renderReport(data) {
  el.fileNameDisp.textContent = data.filename;
  el.fileMeta.textContent = data.shape
    ? `${data.shape.rows.toLocaleString()} rows × ${data.shape.cols} columns`
    : "Could not load file";
  renderSummary(data.summary);
  renderNav(data.issues);
  renderIssues(data.issues, "ALL");
  el.filterBtns.forEach(b => b.classList.toggle("active", b.dataset.filter === "ALL"));
  state.activeFilter = "ALL";
}

function renderSummary(s) {
  el.summaryGrid.innerHTML = [
    {key:"CRITICAL",label:"Critical",cls:"sc-critical"},
    {key:"WARNING", label:"Warnings",cls:"sc-warning"},
    {key:"INFO",    label:"Info",    cls:"sc-info"},
    {key:"OK",      label:"Passed",  cls:"sc-ok"},
  ].map(c => `<div class="summary-card ${c.cls}">
    <span class="s-count">${s[c.key]||0}</span>
    <span class="s-label">${c.label}</span></div>`).join("");
}

function renderNav(issues) {
  const ORDER = {CRITICAL:0,WARNING:1,INFO:2,OK:3};
  const sections = [...new Set(issues.map(i => i.section))];
  el.sectionNav.innerHTML = sections.map(sec => {
    const worst = issues.filter(i=>i.section===sec)
      .reduce((b,i) => ORDER[i.severity]<ORDER[b]?i.severity:b,"OK");
    const dot = {CRITICAL:"critical",WARNING:"warning",INFO:"info",OK:"ok"}[worst];
    const slug = slugify(sec);
    return `<a class="nav-item" href="#section-${slug}" data-slug="${slug}">
      <span class="nav-dot ${dot}"></span>${escHtml(sec)}</a>`;
  }).join("");
}

function renderIssues(issues, filter) {
  const sections = [...new Set(issues.map(i => i.section))];
  el.issuesList.innerHTML = sections.map(sec => {
    const visible = issues.filter(i => i.section===sec && (filter==="ALL"||i.severity===filter));
    if (!visible.length) return "";
    const slug = slugify(sec);
    return `<div class="section-group" id="section-${slug}">
      <div class="section-heading">${escHtml(sec)}</div>
      ${visible.map(buildIssueCard).join("")}</div>`;
  }).join("");

  el.issuesList.querySelectorAll(".fix-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      applyFix(btn,
        btn.dataset.fixId,
        JSON.parse(btn.dataset.params||"{}"),
        btn.closest(".issue-card").dataset.issueId);
    });
  });
}

function buildIssueCard(issue) {
  const sev   = issue.severity.toLowerCase();
  const fixed = state.fixedIds.has(issue.id) ? "fixed" : "";

  const details = issue.details
    ? `<div class="details-block">${escHtml(issue.details)}</div>` : "";

  const suggestion = issue.fix_text
    ? `<div class="fix-suggestion"><strong>💡 Suggestion:</strong> ${escHtml(issue.fix_text)}</div>` : "";

  const fixBtns = (issue.fixes && issue.fixes.length)
    ? `<div class="fix-prompt">
         <span class="fix-prompt-label">Apply an automated fix?</span>
         <div class="fix-actions">
           ${issue.fixes.map(f=>`
             <button class="fix-btn"
               data-fix-id="${escHtml(f.fix_id)}"
               data-params="${escHtml(JSON.stringify(f.params))}">
               <span class="spinner"></span>⚡ ${escHtml(f.label)}
             </button>`).join("")}
           <button class="skip-btn">Skip</button>
         </div>
       </div>` : "";

  const body = details + suggestion + fixBtns;
  return `<div class="issue-card sev-${sev} ${fixed}" data-issue-id="${issue.id}">
    <div class="card-top">
      <span class="badge badge-${sev}">${issue.severity}</span>
      <span class="card-message">${escHtml(issue.message)}</span>
    </div>
    ${body ? `<div class="card-body">${body}</div>` : ""}
  </div>`;
}

// ── Fix ───────────────────────────────────────────────────────
function applyFix(btn, fixId, params, issueId) {
  const card = btn.closest(".issue-card");
  card.querySelectorAll(".fix-btn").forEach(b => b.disabled=true);
  btn.classList.add("loading");
  fetch("/api/fix",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({session_id:state.sessionId,fix_id:fixId,params})})
    .then(r=>r.json())
    .then(data=>{
      if(data.error){
        showToast("Fix failed: "+data.error,"error");
        card.querySelectorAll(".fix-btn").forEach(b=>b.disabled=false);
        btn.classList.remove("loading"); return;
      }
      showToast("✓ "+data.message,"success");
      state.fixedIds.add(issueId);
      state.issues=data.issues; state.summary=data.summary; state.shape=data.shape;
      renderSummary(state.summary);
      el.fileMeta.textContent=`${data.shape.rows.toLocaleString()} rows × ${data.shape.cols} columns`;
      renderNav(state.issues);
      renderIssues(state.issues,state.activeFilter);
    })
    .catch(err=>{
      showToast("Network error: "+err.message,"error");
      card.querySelectorAll(".fix-btn").forEach(b=>b.disabled=false);
      btn.classList.remove("loading");
    });
}

// ── Skip ──────────────────────────────────────────────────────
document.addEventListener("click",e=>{
  if(e.target.classList.contains("skip-btn")){
    const p=e.target.closest(".fix-prompt");
    if(p){p.style.opacity="0.35";p.style.pointerEvents="none";}
  }
});

// ── Filters ───────────────────────────────────────────────────
el.filterBtns.forEach(btn=>btn.addEventListener("click",()=>{
  state.activeFilter=btn.dataset.filter;
  el.filterBtns.forEach(b=>b.classList.toggle("active",b===btn));
  renderIssues(state.issues,state.activeFilter);
}));

// ── Download ──────────────────────────────────────────────────
el.downloadBtn.addEventListener("click",()=>{
  if(state.sessionId) window.location.href=`/api/download/${state.sessionId}`;
});

// ── Preview ───────────────────────────────────────────────────
el.previewBtn.addEventListener("click",()=>{
  el.previewModal.classList.remove("hidden");
  el.previewWrap.innerHTML="<p class='loading-msg'>Loading…</p>";
  fetch(`/api/preview/${state.sessionId}`).then(r=>r.json()).then(data=>{
    if(data.error){el.previewWrap.innerHTML=`<p class='loading-msg'>${data.error}</p>`;return;}
    const headers=data.columns.map(c=>`<th>${escHtml(String(c))}</th>`).join("");
    const rows=data.rows.map(row=>`<tr>${row.map(cell=>
      cell===null||cell===undefined
        ?`<td class="null-cell">null</td>`
        :`<td>${escHtml(String(cell))}</td>`).join("")}</tr>`).join("");
    el.previewWrap.innerHTML=`
      <p style="padding:10px 14px;font-size:12px;color:var(--text-muted);font-family:var(--font-mono);">
        Showing first ${data.rows.length} of ${data.total_rows.toLocaleString()} rows</p>
      <table class="data-table">
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody></table>`;
  }).catch(err=>{el.previewWrap.innerHTML=`<p class='loading-msg'>Error: ${err.message}</p>`;});
});
el.modalClose.addEventListener("click",()=>el.previewModal.classList.add("hidden"));
el.previewModal.addEventListener("click",e=>{if(e.target===el.previewModal)el.previewModal.classList.add("hidden");});

// ── Back ──────────────────────────────────────────────────────
el.backBtn.addEventListener("click",()=>{
  state.sessionId=null;state.issues=[];state.fixedIds=new Set();showScreen("upload");
});

// ── Nav scroll highlight ──────────────────────────────────────
const reportMain=document.getElementById("report-main");
if(reportMain){
  reportMain.addEventListener("scroll",()=>{
    let current=null;
    el.issuesList.querySelectorAll(".section-group").forEach(g=>{
      if(g.getBoundingClientRect().top<160) current=g.id;
    });
    el.sectionNav.querySelectorAll(".nav-item").forEach(item=>{
      item.classList.toggle("active",item.dataset.slug&&current==="section-"+item.dataset.slug);
    });
  });
}

// ── Utils ─────────────────────────────────────────────────────
function escHtml(str){
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function slugify(str){
  return str.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,"");
}
