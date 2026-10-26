/* =====================================================================
   Estado Global
   ===================================================================== */
const $ = id => document.getElementById(id);
let SCHEMA = { model: null, inputs: [], result_columns: [] };

const state = { rows: [], results: [] };
let _checkTokens = {};

const escHtml = v => String(v == null ? "" : v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const escAttr = v => escHtml(v).replace(/"/g,"&quot;");

/* Evaluador unificado de cumplimiento (Ratios + Comprobaciones Geométricas) */
function evaluateCumple(chk) {
  if (!chk) return false;
  let ok = chk.cumple !== undefined ? Boolean(chk.cumple) : true;

  if (chk.ratio_pesimo != null) {
    const rp = Number(chk.ratio_pesimo);
    if (!isNaN(rp) && rp > 1.0) ok = false;
  }

  if (Array.isArray(chk.ratios)) {
    for (const r of chk.ratios) {
      if (r.ratio != null && Number(r.ratio) > 1.0) {
        ok = false;
        break;
      }
    }
  }

  if (Array.isArray(chk.complementarios)) {
    for (const c of chk.complementarios) {
      const resStr = String(c.resultado || "").toLowerCase();
      if (resStr.includes("no cumple") || resStr.includes("fail") || resStr.includes("error") || c.cumple === false) {
        ok = false;
        break;
      }
    }
  }

  return ok;
}

/* Helper seguro para obtener variaciones de una conexión */
function getConnVariations(connIdx) {
  const conn = state.results[connIdx];
  if (!conn) return [];
  if (Array.isArray(conn.variations) && conn.variations.length > 0) {
    return conn.variations;
  }
  const r = conn.data || conn;
  const lbl = r.Escenario || r.escenario || r.descripcion || "Diseño Principal";
  const chk = conn.check || null;
  const isOk = evaluateCumple(chk);
  return [{
    label: lbl,
    tipo: "E1",
    data: r,
    adjusted: { ...r },
    raw: conn.raw || r,
    check: chk,
    adjustedCheck: chk ? JSON.parse(JSON.stringify(chk)) : null,
    status: conn.status || (isOk ? "CUMPLE" : "NO CUMPLE"),
    accepted: false
  }];
}

/* =====================================================================
   Formulario dinámico
   ===================================================================== */
function renderForm(){
  const form = $("form");
  form.innerHTML = "";
  if (!SCHEMA.inputs.length){
    form.innerHTML = `<div class="empty">Sin esquema: no hay modelo cargado.</div>`;
    return;
  }
  SCHEMA.inputs.forEach(c => {
    const box = document.createElement("div");
    box.className = "f"; box.dataset.key = c.key;
    const id = "in_" + c.key;
    const desc = c.description ? ` <small>· ${escHtml(c.description)}</small>` : "";
    const lab = `<label for="${id}">${escHtml(c.key)}${desc}</label>`;
    let ctl;
    if (c.type === "categorical"){
      const opts = c.options || [];
      if (opts.length > 25){
        const dl = "dl_" + c.key;
        ctl = `<input id="${id}" list="${dl}" autocomplete="off" placeholder="Escribe para buscar">` +
              `<datalist id="${dl}">${opts.map(o => `<option value="${escAttr(o)}">`).join("")}</datalist>`;
      } else {
        ctl = `<select id="${id}"><option value="">— Selecciona —</option>` +
              opts.map(o => `<option value="${escAttr(o)}">${escHtml(o)}</option>`).join("") + `</select>`;
      }
    } else if (c.type === "bool"){
      ctl = `<select id="${id}"><option value="">—</option><option value="true">true</option><option value="false">false</option></select>`;
    } else if (c.type === "number"){
      ctl = `<input id="${id}" type="number" step="any" autocomplete="off">`;
    } else {
      ctl = `<input id="${id}" type="text" autocomplete="off">`;
    }
    box.innerHTML = lab + ctl + `<div class="msg"></div>`;
    form.appendChild(box);
  });
}

function readForm(){
  let ok = true; const row = {};
  SCHEMA.inputs.forEach(c => {
    const box = $("form").querySelector(`[data-key="${c.key}"]`);
    const el = $("in_" + c.key);
    const val = (el && el.value != null ? el.value : "").trim();
    let msg = "";
    if (val !== "" && c.type === "categorical" && Array.isArray(c.options) && c.options.length){
      const m = c.options.find(o => String(o).toLowerCase() === val.toLowerCase());
      if (m === undefined) msg = "No está en el catálogo"; else row[c.key] = m;
    } else if (val !== ""){
      row[c.key] = (c.type === "number") ? Number(val) : val;
    }
    box.classList.toggle("err", !!msg);
    box.querySelector(".msg").textContent = msg;
    if (msg) ok = false;
  });
  return ok ? row : null;
}

/* =====================================================================
   Carga de Excel / CSV
   ===================================================================== */
$("btnUploadExcel").onclick = () => $("excelFileInput").click();

$("excelFileInput").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (evt) => {
    try {
      const data = new Uint8Array(evt.target.result);
      const workbook = XLSX.read(data, { type: "array" });
      const firstSheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];
      const json = XLSX.utils.sheet_to_json(worksheet, { defval: "" });

      if (!json.length) { setStatus("bad", "El archivo Excel está vacío."); return; }

      if (json[0].Escenario || json[0]["Unicode origen"] || json[0].escenario) {
        state.results = parseScenariosResponse({ rows: json }, []);
        renderResults();
        renderAcceptedTable();
        setStatus("ok", `Se han cargado ${state.results.length} conexiones con sus variaciones desde "${file.name}".`);
        return;
      }

      if (!SCHEMA.inputs.length) { setStatus("bad", "No hay esquema de modelo cargado."); return; }

      let count = 0;
      json.forEach(rawRow => {
        const rowNorm = {};
        Object.keys(rawRow).forEach(k => { rowNorm[k.trim().toLowerCase()] = rawRow[k]; });
        const newRow = {};
        SCHEMA.inputs.forEach(c => {
          const keyLower = c.key.toLowerCase();
          if (rowNorm[keyLower] !== undefined && rowNorm[keyLower] !== "") {
            let val = rowNorm[keyLower];
            if (c.type === "number" && typeof val !== "number") {
              const num = Number(String(val).replace(",", "."));
              if (!isNaN(num)) val = num;
            }
            newRow[c.key] = val;
          }
        });
        if (Object.keys(newRow).length > 0) { state.rows.push(newRow); count++; }
      });
      renderTable();
      setStatus("ok", `Se han cargado ${count} filas desde "${file.name}".`);
    } catch (err) {
      setStatus("bad", "Error procesando el archivo", err.message);
    }
    $("excelFileInput").value = "";
  };
  reader.readAsArrayBuffer(file);
};

/* =====================================================================
   Tablas de Trabajo
   ===================================================================== */
function table(rows, cols, withDelete){
  if (!rows.length) return `<div class="empty">Todavía no hay combinaciones. Rellena el formulario o carga un Excel.</div>`;
  const head = cols.map(c => `<th>${escHtml(colName(c))}</th>`).join("") + (withDelete ? "<th></th>" : "");
  const body = rows.map((r, i) =>
    "<tr>" + cols.map(c => `<td>${escHtml(cellVal(r, colName(c)))}</td>`).join("") +
    (withDelete ? `<td class="x"><button data-del="${i}" title="Quitar fila">✕</button></td>` : "") + "</tr>"
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderTable(){
  const cols = SCHEMA.inputs.map(c => c.key);
  $("tbl").innerHTML = table(state.rows, cols, true);
  $("count").textContent = state.rows.length;
  $("send").disabled = $("clear").disabled = !state.rows.length;
  $("tbl").querySelectorAll("[data-del]").forEach(b => b.onclick = () => { state.rows.splice(+b.dataset.del, 1); renderTable(); });
}

/* =====================================================================
   Renderizado de la Pantalla Inicial (Sección 1: Resumen por Conexión)
   ===================================================================== */
function renderResults(){
  const res = state.results;
  $("resSec").style.display = res.length ? "" : "none";
  if (!res.length){ renderAcceptedTable(); return; }

  const head = `
    <th>Conexión (Unicode)</th>
    <th title="Visualizar Plano 2D">Plano 2D</th>
    <th>Selección de Variantes Aceptadas</th>
  `;

  const body = res.map((conn, connIdx) => {
    const vars = getConnVariations(connIdx);
    const activeVar = vars[conn.activeVarIndex || 0] || vars[0];
    const unicodeVal = activeVar?.data?.["Unicode origen"] || activeVar?.data?.Unicode || (`Conexión_${connIdx + 1}`);

    const cardsHtml = `<div class="var-cards-wrap">` +
      vars.map((v, vI) => {
        if (v.unavailable) {
          return `
          <div class="var-card-item unavailable" title="${escAttr(v.motivo)}">
            <span class="var-name">${escHtml(v.label)}</span>
            <span class="badge mod">${escHtml(v.status)}</span>
            <small class="var-note">${escHtml(v.motivo)}</small>
          </div>`;
        }
        const isCumple = evaluateCumple(v.check);
        const st = v.status || (isCumple ? "CUMPLE" : "NO CUMPLE");
        const badgeCls = st === "CUMPLE" ? "ok" : st === "NO CUMPLE" ? "bad" : "mod";
        const isChecked = !!v.accepted;

        return `
          <div class="var-card-item ${isChecked ? "accepted" : ""}">
            <label>
              <input type="checkbox" class="var-accept-chk" data-conn="${connIdx}" data-var="${vI}" ${isChecked ? "checked" : ""}>
              <span>${escHtml(v.label)}</span>
            </label>
            <span class="badge ${badgeCls}">${escHtml(st)}</span>
          </div>
        `;
      }).join("") + `</div>`;

    return `<tr>` +
      `<td><b style="color:var(--tr-navy); font-size:14px;">${escHtml(unicodeVal)}</b></td>` +
      `<td><button class="btn ghost vbtn" data-view-conn="${connIdx}" data-view-var="${conn.activeVarIndex || 0}" style="padding:4px 10px; font-size:13px;">👁 Ver Plano 2D</button></td>` +
      `<td>${cardsHtml}</td>` +
    `</tr>`;
  }).join("");

  $("res").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  $("res").querySelectorAll("input.var-accept-chk").forEach(chk => {
    chk.addEventListener("change", () => {
      const cI = +chk.dataset.conn;
      const vI = +chk.dataset.var;
      const vars = getConnVariations(cI);
      if (vars[vI]) {
        vars[vI].accepted = chk.checked;
      }
      renderResults();
      renderAcceptedTable();
      if (VIEW && VIEW.connIdx === cI) buildViewerPanel();
    });
  });

  $("res").querySelectorAll("button.vbtn").forEach(b => {
    b.onclick = () => openViewer(+b.dataset.viewConn, +b.dataset.viewVar);
  });

  renderAcceptedTable();
}

/* =====================================================================
   Renderizado de la Pantalla Inicial (Sección 2: Variantes Aceptadas)
   ===================================================================== */
const colName = c => (typeof c === "string") ? c : String((c && (c.key || c.name || c.label)) || "");

/* Valor de celda legible: busca la columna sin distinguir mayúsculas, redondea decimales */
function cellVal(r, c){
  let v = r[c];
  if (v === undefined) {
    const k = Object.keys(r).find(x => x.toLowerCase() === String(c).toLowerCase());
    if (k !== undefined) v = r[k];
  }
  if (v === null || v === undefined || v === "") return "";
  if (typeof v === "number") return Number.isInteger(v) ? v : Math.round(v * 1000) / 1000;
  if (typeof v === "boolean") return v ? "Sí" : "No";
  if (typeof v === "object") return JSON.stringify(v);
  return v;
}

function renderAcceptedTable(){
  const acceptedList = [];

  state.results.forEach((conn, connIdx) => {
    const vars = getConnVariations(connIdx);
    vars.forEach((v, vIdx) => {
      if (v.accepted) {
        const isCumple = evaluateCumple(v.check);
        acceptedList.push({
          connIdx,
          varIdx: vIdx,
          unicode: v.data["Unicode origen"] || v.data.Unicode || (`Conexión_${connIdx + 1}`),
          label: v.label,
          data: v.data,
          check: v.check,
          status: v.status || (isCumple ? "CUMPLE" : "NO CUMPLE")
        });
      }
    });
  });

  const sec = $("acceptedSec");
  $("acceptedCount").textContent = acceptedList.length;
  $("sendAccepted").disabled = !acceptedList.length;

  if (!state.results.length) {
    sec.style.display = "none";
    return;
  }
  sec.style.display = "";

  const container = $("acceptedTbl");
  if (!acceptedList.length) {
    container.innerHTML = `<div class="empty">No has marcado ninguna variante como aceptada todavía. Haz clic en los casilleros para agregar diseños a la lista final.</div>`;
    return;
  }

  const sampleData = acceptedList[0].data;
  const cols = SCHEMA.result_columns.length ? SCHEMA.result_columns : Object.keys(sampleData).filter(k => !["raw", "adjusted", "adjustedCheck", "check"].includes(k));

  const head = `<th>Visualizar</th><th>Variante Aceptada</th>` +
    cols.map(c => `<th>${escHtml(colName(c))}</th>`).join("") +
    `<th>Comprobación AISC</th><th>Quitar</th>`;

  const body = acceptedList.map((item) => {
    const r = item.data;
    const st = item.status;
    const stCls = st === "CUMPLE" ? "ok" : st === "NO CUMPLE" ? "bad" : "mod";

    return `<tr>` +
      `<td><button class="vbtn" data-view-conn="${item.connIdx}" data-view-var="${item.varIdx}" title="Ver en 2D">👁</button></td>` +
      `<td><b style="color:var(--tr-navy);">${escHtml(item.label)}</b></td>` +
      cols.map(c => `<td>${escHtml(cellVal(r, colName(c)))}</td>`).join("") +
      `<td class="st ${stCls}">${escHtml(st)}</td>` +
      `<td class="x"><button class="discard-btn" data-conn="${item.connIdx}" data-var="${item.varIdx}" title="Quitar de aceptadas">✕</button></td>` +
    `</tr>`;
  }).join("");

  container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  container.querySelectorAll("button.discard-btn").forEach(b => {
    b.onclick = () => {
      const cI = +b.dataset.conn;
      const vI = +b.dataset.var;
      const vars = getConnVariations(cI);
      if (vars[vI]) vars[vI].accepted = false;
      renderResults();
      renderAcceptedTable();
      if (VIEW && VIEW.connIdx === cI) buildViewerPanel();
    };
  });

  container.querySelectorAll("button.vbtn").forEach(b => {
    b.onclick = () => openViewer(+b.dataset.viewConn, +b.dataset.viewVar);
  });
}

/* =====================================================================
   Exportación Excel
   ===================================================================== */
$("exportCheckExcel").onclick = () => {
  const acceptedList = [];
  state.results.forEach((conn, connIdx) => {
    const vars = getConnVariations(connIdx);
    vars.forEach((v, vIdx) => {
      if (v.accepted) acceptedList.push({ connIdx, vIdx, v });
    });
  });

  if (!acceptedList.length) { setStatus("bad", "No hay variantes aceptadas para exportar."); return; }

  const exportRows = [];
  acceptedList.forEach(({ connIdx, vIdx, v }) => {
    const chk = v.check || {};
    const isCumple = evaluateCumple(chk);
    const rowObj = {
      "Conexión #": connIdx + 1,
      "Variación": v.label || (`Variación ${vIdx + 1}`),
      "UNICODE": v.data.Unicode || v.data.UNICODE || (connIdx + 1),
      "Estado Global": v.status || (isCumple ? "CUMPLE" : "NO CUMPLE"),
      "Ratio Pésimo": chk.ratio_pesimo ?? "",
      "Chequeo Crítico": chk.descripcion_ratio_pesimo || ""
    };
    Object.keys(v.data).forEach(k => {
      if (!["ratios", "complementarios", "cumple", "ratio_pesimo", "descripcion_ratio_pesimo", "raw", "adjusted", "adjustedCheck", "check"].includes(k)) {
        rowObj[k] = v.data[k];
      }
    });
    if (Array.isArray(chk.ratios)) {
      chk.ratios.forEach(ratioItem => { rowObj[`Ratio: ${ratioItem.descripcion}`] = ratioItem.ratio ?? "N/A"; });
    }
    if (Array.isArray(chk.complementarios)) {
      chk.complementarios.forEach(compItem => { rowObj[`Geometría: ${compItem.descripcion}`] = compItem.resultado; });
    }
    exportRows.push(rowObj);
  });

  const worksheet = XLSX.utils.json_to_sheet(exportRows);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Comprobaciones AISC360");
  XLSX.writeFile(workbook, `Comprobaciones_VariantesAceptadas_${new Date().toISOString().slice(0, 10)}.xlsx`);
  setStatus("ok", "Exportación de comprobaciones de variantes aceptadas a Excel completada.");
};

$("exportAcceptedExcel").onclick = () => {
  const acceptedList = [];
  state.results.forEach((conn, connIdx) => {
    const vars = getConnVariations(connIdx);
    vars.forEach((v, vIdx) => {
      if (v.accepted) acceptedList.push({ connIdx, vIdx, v });
    });
  });

  if (!acceptedList.length) { setStatus("bad", "No hay variantes aceptadas para exportar."); return; }

  const exportRows = acceptedList.map(({ connIdx, vIdx, v }) => {
    const cleanObj = { "Conexión #": connIdx + 1, "Variación": v.label || (`Variación ${vIdx + 1}`) };
    Object.keys(v.data).forEach(k => {
      if (!["raw", "adjusted", "adjustedCheck", "check"].includes(k)) cleanObj[k] = v.data[k];
    });
    return cleanObj;
  });

  const worksheet = XLSX.utils.json_to_sheet(exportRows);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Variantes Aceptadas");
  XLSX.writeFile(workbook, `Variantes_Aceptadas_${new Date().toISOString().slice(0, 10)}.xlsx`);
  setStatus("ok", "Variantes aceptadas exportadas a Excel.");
};

function setStatus(kind, text, detail){
  const s = $("status");
  s.className = kind; s.style.display = text ? "block" : "none";
  s.innerHTML = `<b>${escHtml(text)}</b>` + (detail ? `<pre></pre>` : "");
  if (detail) s.querySelector("pre").textContent = detail;
}

/* =====================================================================
   Visualizador 2D con Categorización de Parámetros Completa
   ===================================================================== */
const PROFILE_CACHE = {};
let VIEW = null;

const P_KEYS = ["p3","p4","p5","p6","p7","p8","p9","p10","p11","p12"];

const VIEW_GROUPS_DEF = [
  {
    title: "1. Elementos y Materiales",
    fields: [
      { label: "Perfil Viga (BEAM)", aliases: ["BEAM", "beam", "Viga"], type: "text" },
      { label: "Perfil Pilar (COLUMN)", aliases: ["COLUMN", "column", "Pilar"], type: "text" },
      { label: "Diámetro Tornillo", aliases: ["BoltName_ep", "boltname_ep", "BoltName"], type: "text" },
      { label: "Clase de Tornillo", aliases: ["BoltGrade", "boltgrade", "Grade"], type: "text" },
    ]
  },
  {
    title: "2. Chapa Frontal y Cantonera",
    fields: [
      { label: "Bp · Ancho de chapa", aliases: ["b_ep", "Bp", "bp"], type: "mm" },
      { label: "tp · Espesor de chapa", aliases: ["ThicknessName_t_ep", "t_ep", "tp"], type: "mm" },
      { label: "Lh · Longitud cantonera/cartela", aliases: ["Lh", "lh"], type: "mm" },
    ]
  },
  {
    title: "3. Distribución de Tornillería",
    fields: [
      { label: "g · Gramil horizontal", aliases: ["p2_ep", "g", "gramil"], type: "mm" },
      { label: "e · Borde sup. a 1ª fila", aliases: ["dev", "e", "borde"], type: "mm" },
      { label: "pfo · 1ª fila a ala sup. viga", aliases: ["pfo"], type: "mm" },
      { label: "pfi · Ala sup. a 2ª fila", aliases: ["pfi"], type: "mm" },
      { label: "p2 · Separación 2ª a 3ª fila", aliases: ["pb", "p2"], type: "mm" },
      ...P_KEYS.map(k => ({ label: `${k} · Separación de fila`, aliases: [k], type: "mm" }))
    ]
  },
  {
    title: "4. Rigidizadores y Refuerzos",
    fields: [
      { label: "tst · Rigid. tracción pilar", aliases: ["ThicknessName_t_tenstiff", "t_tenstiff", "tst"], type: "mm" },
      { label: "ts · Rigid. compresión pilar", aliases: ["ThicknessName_t_compstiff", "t_compstiff", "ts"], type: "mm" },
      { label: "tws · Placa de alma pilar", aliases: ["ThicknessName_t_webstiff", "t_webstiff", "tws"], type: "mm" },
      { label: "Bws · Ancho placa alma", aliases: ["Bws", "bws"], type: "mm" },
      { label: "Lws · Alto placa alma", aliases: ["Lws", "lws"], type: "mm" },
      { label: "tbs · Rigid. alma viga", aliases: ["ThicknessName_t_BeamHcompstiff", "t_BeamHcompstiff", "tbs"], type: "mm" },
    ]
  },
  {
    title: "5. Solicitaciones y Cargas",
    fields: [
      { label: "Mz · Solicitación momento", aliases: ["ratio_moment", "MzEd_Porcen"], type: "pct" },
      { label: "Fy · Solicitación cortante", aliases: ["ratio_shear", "FyEd_Porcen"], type: "pct" },
      { label: "Fx · Solicitación axial", aliases: ["ratio_axial", "FxEd_Porcen"], type: "pct" },
    ]
  }
];

const SEC_FIELDS_GROUP = {
  title: "6. Cotas de Secciones (Catálogo)",
  fields: [
    { k: "beam_d", label: "d_viga · Canto viga", type: "mm" },
    { k: "beam_bf", label: "bf_viga · Ancho ala viga", type: "mm" },
    { k: "beam_tw", label: "tw_viga · Espesor alma viga", type: "mm" },
    { k: "beam_tf", label: "tf_viga · Espesor ala viga", type: "mm" },
    { k: "col_d", label: "d_pilar · Canto pilar", type: "mm" },
    { k: "col_bf", label: "bf_pilar · Ancho ala pilar", type: "mm" },
    { k: "col_tw", label: "tw_pilar · Espesor alma pilar", type: "mm" },
    { k: "col_tf", label: "tf_pilar · Espesor ala pilar", type: "mm" },
  ]
};

function getFieldInfo(row, aliases) {
  for (const a of aliases) {
    for (const k of Object.keys(row)) {
      if (k.toLowerCase() === a.toLowerCase()) {
        return { key: k, value: row[k] };
      }
    }
  }
  return { key: aliases[0], value: "" };
}

function vNum(v){
  const s = String(v == null ? "" : v).toLowerCase().replace("mm", "").replace(",", ".").trim();
  const n = parseFloat(s);
  return isNaN(n) ? 0 : n;
}
function vMmText(v){ return Math.round(vNum(v)) + "mm"; }
function fmt(n){ return String(Math.round(n * 10) / 10); }

async function fetchProfiles(names){
  const need = [...new Set(names.filter(Boolean).map(n => String(n).trim()))].filter(n => !(n in PROFILE_CACHE));
  if (need.length){
    try{
      const r = await fetch("/profiles?names=" + encodeURIComponent(need.join(",")));
      if (r.ok){ const d = await r.json(); Object.assign(PROFILE_CACHE, d.profiles || {}); }
    }catch(e){}
  }
  const out = {};
  names.filter(Boolean).forEach(n => { out[n] = PROFILE_CACHE[String(n).trim()] || null; });
  return out;
}

async function openViewer(connIdx, initialVarIdx = 0){
  const vars = getConnVariations(connIdx);
  if (!vars.length) return;

  VIEW = { connIdx, activeVarIndex: initialVarIdx };
  
  const activeVar = vars[initialVarIdx] || vars[0];
  const connTitle = state.results[connIdx]?.inputRow?.Unicode || 
                    activeVar?.data?.["Unicode origen"] || 
                    activeVar?.data?.Unicode || 
                    ("Conexión " + (connIdx + 1));
                    
  $("viewerTitle").textContent = "Visualización 2D · " + connTitle;

  await loadViewerVariation(initialVarIdx);

  $("viewer").style.display = "flex";
  document.body.classList.add("modal-open");
  updateSaveButton();
}

async function loadViewerVariation(varIdx){
  if (!VIEW) return;
  clearTimeout(VIEW.checkTimer);
  VIEW.activeVarIndex = varIdx;
  const vars = getConnVariations(VIEW.connIdx);
  const activeVar = vars[varIdx] || vars[0];
  if (!activeVar) return;

  const row = activeVar.data;
  const prof = await fetchProfiles([row.BEAM, row.COLUMN]);
  const beam = prof[row.BEAM] || null, col = prof[row.COLUMN] || null;

  VIEW.row = { ...row };
  VIEW.rowBaseline = { ...row };
  VIEW.check = activeVar.check ? JSON.parse(JSON.stringify(activeVar.check)) : null;
  VIEW.checking = false; VIEW.pending = false; VIEW.checkError = ""; VIEW.needRebuild = false;
  VIEW.beam = beam; VIEW.col = col; VIEW.warn = "";

  setVals(row);
  VIEW.baselineVals = JSON.parse(JSON.stringify(VIEW.vals));
  VIEW.profKey = VIEW.row[VIEW.keyOf["Perfil Viga (BEAM)"]] + "|" + VIEW.row[VIEW.keyOf["Perfil Pilar (COLUMN)"]];

  VIEW.sec = {
    beam_d: beam ? beam.d_mm : "", beam_bf: beam ? beam.bf_mm : "",
    beam_tw: beam ? beam.tw_mm : "", beam_tf: beam ? beam.tf_mm : "",
    col_d: col ? col.d_mm : "", col_bf: col ? col.bf_mm : "",
    col_tw: col ? col.tw_mm : "", col_tf: col ? col.tf_mm : "",
  };

  const missed = [];
  if (!String(row.BEAM || "").trim()) missed.push("viga (columna BEAM vacía)");
  else if (!beam) missed.push("viga '" + row.BEAM + "'");
  if (!String(row.COLUMN || "").trim()) missed.push("pilar (columna COLUMN vacía)");
  else if (!col) missed.push("pilar '" + row.COLUMN + "'");

  VIEW.warn = missed.length
    ? "No encontrado en el catálogo: " + missed.join(", ") + ". Introduce las cotas de sección a mano."
    : "";

  buildViewerPanel();
  drawAll();
  checkViewerConnection();
}

window.switchViewerTab = function(varIdx){
  if (!VIEW || VIEW.activeVarIndex === varIdx) return;
  const tv = getConnVariations(VIEW.connIdx)[varIdx];
  if (!tv || tv.unavailable) return;
  if (state.results[VIEW.connIdx]) state.results[VIEW.connIdx].activeVarIndex = varIdx;
  loadViewerVariation(varIdx);
};

function varListHtml(){
  const vars = getConnVariations(VIEW.connIdx);
  if (!vars.length) return "";
  let html = `<div style="font-weight:700; font-size:13px; color:var(--tr-navy); margin-bottom:8px;">Variaciones del Diseño</div>`;
  html += `<div style="display:flex; flex-direction:column; gap:8px; margin-bottom:14px;">` +
    vars.map((v, idx) => {
      if (v.unavailable) {
        return `
        <div style="background:#F2F5F9; border:1px dashed var(--tr-line); border-radius:5px; padding:6px 10px; color:var(--tr-muted);" title="${escAttr(v.motivo)}">
          <div style="display:flex; align-items:center; gap:8px; justify-content:space-between;">
            <span style="font-weight:600; font-size:12.5px;">${escHtml(v.label)}</span>
            <span class="badge mod">${escHtml(v.status)}</span>
          </div>
          <div style="font-size:11.5px; margin-top:4px; line-height:1.35;">${escHtml(v.motivo)}</div>
        </div>`;
      }
      const isActive = idx === VIEW.activeVarIndex;
      const chk = isActive ? VIEW.check : v.check;
      let st, badgeCls;
      if (isActive && VIEW.checking) { st = "COMPROBANDO..."; badgeCls = "mod"; }
      else if (!chk || (isActive && VIEW.pending)) { st = "SIN COMPROBAR"; badgeCls = "mod"; }
      else { const ok = evaluateCumple(chk); st = ok ? "CUMPLE" : "NO CUMPLE"; badgeCls = ok ? "ok" : "bad"; }
      const btnBg = isActive ? "#002B5C" : "#F2F5F9";
      const btnColor = isActive ? "#FFFFFF" : "var(--tr-navy)";
      const border = isActive ? "2px solid #002B5C" : "1px solid var(--tr-line)";
      return `
        <div style="display:flex; align-items:center; gap:8px; background:${btnBg}; border:${border}; border-radius:5px; padding:6px 10px;">
          <input type="checkbox" id="modal_accept_${idx}" class="modal-var-chk" data-var="${idx}" ${v.accepted ? "checked" : ""} style="width:16px; height:16px; cursor:pointer;" title="Aceptar variante">
          <button type="button" onclick="switchViewerTab(${idx})" style="
            flex:1; border:none; background:transparent; color:${btnColor}; cursor:pointer; font-weight:600; font-size:12.5px; text-align:left; padding:2px 0;
          ">
            <span>${escHtml(v.label)}</span>
          </button>
          <span class="badge ${badgeCls}">${escHtml(st)}</span>
        </div>
      `;
    }).join("") + `</div>`;
  return html;
}

function bindVarChecks(){
  const p = $("viewerPanel");
  if (!p || !VIEW) return;
  p.querySelectorAll("input.modal-var-chk").forEach(chk => {
    chk.addEventListener("change", (e) => {
      e.stopPropagation();
      const vars = getConnVariations(VIEW.connIdx);
      const vI = +chk.dataset.var;
      if (vars[vI]) {
        vars[vI].accepted = chk.checked;
        renderResults();
        renderAcceptedTable();
      }
    });
  });
}

function renderVarList(){
  const el = $("viewerVarList");
  if (!el || !VIEW) return;
  el.innerHTML = varListHtml();
  bindVarChecks();
}

function buildViewerPanel(){
  const p = $("viewerPanel");
  if (!p || !VIEW) return;
  const scroll = p.scrollTop;
  let html = `<div id="viewerVarList">${varListHtml()}</div>`;

  if (VIEW.warn) html += `<div class="view-warn">${escHtml(VIEW.warn)}</div>`;
  html += `<div id="viewerCheckSection">${checkSectionHtml()}</div>`;

  VIEW_GROUPS_DEF.forEach(grp => {
    html += `<div class="view-group"><div class="view-group-title">${escHtml(grp.title)}</div><div class="view-group-body">`;
    grp.fields.forEach(f => {
      const key = VIEW.keyOf[f.label];
      const t = VIEW.types[key] === "pct01" ? "pct" : f.type;
      html += renderFieldRow(f.label, key, VIEW.vals[key] ?? "", t);
    });
    html += `</div></div>`;
  });

  /* Cotas de Sección */
  html += `<div class="view-group"><div class="view-group-title">${escHtml(SEC_FIELDS_GROUP.title)}</div><div class="view-group-body">`;
  SEC_FIELDS_GROUP.fields.forEach(f => {
    html += renderFieldRow(f.label, f.k, VIEW.sec[f.k] ?? "", "mm", false, "data-skey");
  });
  html += `</div></div>`;

  /* Cotación Calculada */
  html += `
    <div class="view-group">
      <div class="view-group-title">Cotación Calculada</div>
      <div class="view-group-body">
        <div class="view-row ro">
          <label>p1 = pfo + tf_viga + pfi</label>
          <div class="mm-wrap"><input id="vP1" readonly><span class="mm-suf">mm</span></div>
        </div>
        <div class="view-row ro">
          <label>Lp = 2·e + Σp (alto chapa)</label>
          <div class="mm-wrap"><input id="vLp" readonly><span class="mm-suf">mm</span></div>
        </div>
      </div>
    </div>
  `;

  /* Otros Parámetros (columnas no agrupadas): también son editables y se comprueban */
  if (VIEW.extra.length > 0) {
    html += `<div class="view-group"><div class="view-group-title">7. Otros Parámetros de la Conexión</div><div class="view-group-body">`;
    VIEW.extra.forEach(k => {
      const t = VIEW.types[k] === "mm" ? "mm" : (VIEW.types[k] === "text" ? "text" : "num");
      html += renderFieldRow(k, k, VIEW.vals[k] ?? "", t);
    });
    html += `</div></div>`;
  }

  p.innerHTML = html;
  p.scrollTop = scroll;
  bindVarChecks();

  p.querySelectorAll("[data-vkey]").forEach(inp => {
    inp.addEventListener("input", () => onField(inp));
    inp.addEventListener("change", () => { onFieldBlur(inp); checkViewerConnection(); });
  });
}

function renderFieldRow(label, key, val, type, editable = true, bindAttr = "data-vkey"){
  const sufHtml = type === "mm" ? `<span class="mm-suf">mm</span>` :
                  type === "pct" ? `<span class="mm-suf">%</span>` : "";
  const displayVal = type === "mm" ? mmInput(val) : String(val == null ? "" : val);

  return `
    <div class="view-row ${editable ? "" : "ro"}">
      <label>${escHtml(label)}</label>
      <div class="mm-wrap">
        <input ${bindAttr}="${escAttr(key)}" value="${escAttr(displayVal)}" ${editable ? "" : "readonly"}>
        ${sufHtml}
      </div>
    </div>
  `;
}

function checkSectionHtml(){
  const vars = getConnVariations(VIEW.connIdx);
  const activeVar = vars[VIEW.activeVarIndex] || vars[0];
  const checkData = VIEW.check;
  const checking = !!VIEW.checking;
  const hdr = `<div style="font-weight:700; font-size:13px; color:var(--tr-navy); margin-bottom:6px;">Comprobación AISC 360-10</div>`;
  if (VIEW.checkError) return hdr + `<div class="view-warn">${escHtml(VIEW.checkError)}</div>`;
  if (VIEW.pending && !checking) return hdr + `<div style="margin-bottom:12px;"><span class="badge mod">CAMBIOS SIN COMPROBAR</span></div>`;

  if (checking) {
    return `
      <div style="font-weight:700; font-size:13px; color:var(--tr-navy); margin-bottom:6px;">Comprobación AISC 360-10</div>
      <div style="margin-bottom:12px;">
        <span class="badge mod">COMPROBANDO...</span>
      </div>
    `;
  }

  if (checkData){
    const isCumple = evaluateCumple(checkData);
    checkData.cumple = isCumple;
    const st = isCumple ? "CUMPLE" : "NO CUMPLE";
    const badgeCls = isCumple ? "ok" : "bad";

    let out = `
      <div style="font-weight:700; font-size:13px; color:var(--tr-navy); margin-bottom:6px;">Comprobación AISC 360-10</div>
      <div style="margin-bottom:10px;">
        <span class="badge ${badgeCls}">${escHtml(st)}</span>
      </div>
      <div style="font-size:12.5px; margin-bottom:8px; color:var(--tr-navy);">
        <div><b>Ratio pésimo:</b> <span style="font-weight:bold;">${checkData.ratio_pesimo ?? 'N/A'}</span></div>
        <div style="margin-top:2px;"><b>Chequeo crítico:</b> <span>${escHtml(checkData.descripcion_ratio_pesimo || 'N/A')}</span></div>
      </div>
    `;

    if (Array.isArray(checkData.complementarios) && checkData.complementarios.length > 0) {
      out += `
        <details style="margin-bottom:10px; background:#F7FAFD; padding:6px 8px; border:1px solid #D5DDE8; border-radius:4px;">
          <summary style="font-weight:600; font-size:12px; cursor:pointer; color:var(--tr-navy);">Comprobaciones Geométricas (${checkData.complementarios.length})</summary>
          <ul style="margin:6px 0 0 0; padding-left:16px; font-size:11.5px; line-height:1.45;">
            ${checkData.complementarios.map(c => {
              const resStr = String(c.resultado || "");
              const isBad = resStr.toLowerCase().includes("no cumple") || c.cumple === false;
              return `<li><b>${escHtml(c.descripcion)}:</b> <span style="${isBad ? 'color:var(--tr-red);font-weight:bold;' : ''}">${escHtml(resStr)}</span></li>`;
            }).join("")}
          </ul>
        </details>
      `;
    }

    if (Array.isArray(checkData.ratios) && checkData.ratios.length > 0) {
      out += `
        <details style="margin-bottom:10px; background:#F7FAFD; padding:6px 8px; border:1px solid #D5DDE8; border-radius:4px;">
          <summary style="font-weight:600; font-size:12px; cursor:pointer; color:var(--tr-navy);">Ratios de Resistencia (${checkData.ratios.length})</summary>
          <ul style="margin:6px 0 0 0; padding-left:16px; font-size:11.5px; line-height:1.45;">
            ${checkData.ratios.map(f => {
              const val = f.ratio !== null && f.ratio !== undefined ? Number(f.ratio).toFixed(3) : 'N/A';
              const isBad = f.ratio > 1.0;
              return `<li><b>${escHtml(f.descripcion)}:</b> <span style="${isBad ? 'color:var(--tr-red);font-weight:bold;' : ''}">${val}</span></li>`;
            }).join("")}
          </ul>
        </details>
      `;
    }
    return out;
  }

  return `<div style="font-weight:700; font-size:13px; color:var(--tr-navy); margin-bottom:6px;">Comprobación AISC 360-10</div><div style="margin-bottom:12px; background:#EAF3FB; padding:10px; border-left:3px solid var(--tr-blue); border-radius:3px; font-size:12px;">Esta variación no se ha comprobado aún.</div>`;
}

function renderCheckSection(){
  const el = $("viewerCheckSection");
  if (el) el.innerHTML = checkSectionHtml();
}

/* ---------- Valores editables del visualizador ----------
   VIEW.vals   : texto que ve el usuario en cada campo
   VIEW.types  : tipo de cada campo (text | mm | pct | pct01 | num | numstr)
   VIEW.keyOf  : etiqueta del campo -> clave real en la fila CODAME
   VIEW.extra  : claves no agrupadas (sección "Otros parámetros")
   Lo editado en CUALQUIER campo se vuelca a la fila antes de comprobar y al guardar. */
const VIEW_IGNORED = new Set(["unicode", "unicode origen", "unicode_origen", "escenario", "raw", "adjusted",
  "check", "adjustedcheck", "status", "accepted", "n_cantonera"]);

function setVals(src){
  const vals = {}, types = {}, keyOf = {}, mapped = new Set();
  VIEW_GROUPS_DEF.forEach(grp => grp.fields.forEach(f => {
    const info = getFieldInfo(src, f.aliases);
    let v = info.value == null ? "" : info.value, t = f.type;
    if (t === "pct" && /^ratio_/i.test(info.key)) {          // ratios 0-1 -> se muestran en %
      t = "pct01";
      if (String(v).trim() !== "") v = String(Math.round(vNum(v) * 10000) / 100);
    }
    keyOf[f.label] = info.key;
    mapped.add(info.key.toLowerCase());
    vals[info.key] = v; types[info.key] = t;
  }));
  const extra = [];
  Object.keys(src).forEach(k => {
    const v = src[k], kl = k.toLowerCase();
    if (mapped.has(kl) || VIEW_IGNORED.has(kl)) return;
    if (typeof v === "boolean" || (v !== null && typeof v === "object")) return;
    let t = "text";
    if (typeof v === "number") t = "num";
    else if (/^\s*-?\d+([.,]\d+)?\s*mm\s*$/i.test(String(v))) t = "mm";
    else if (String(v).trim() !== "" && !isNaN(Number(String(v).replace(",", ".")))) t = "numstr";
    vals[k] = v == null ? "" : v; types[k] = t; extra.push(k);
  });
  VIEW.vals = vals; VIEW.types = types; VIEW.keyOf = keyOf; VIEW.extra = extra;
}

function sameVal(t, a, b){
  return t === "text" ? String(a ?? "") === String(b ?? "") : vNum(a) === vNum(b);
}

/* Vuelca sobre la fila los valores editados (los que difieren del original), conservando
   el formato de cada columna ("20mm", número, texto...). */
function applyValsToRow(row, vals, types, baseRow, baseVals){
  Object.keys(vals).forEach(k => {
    const t = types[k] || "text";
    if (baseVals && k in baseVals && sameVal(t, vals[k], baseVals[k])) return;
    const s = String(vals[k] == null ? "" : vals[k]).trim();
    if (!(k in row) && s === "") return;                      // no inventar columnas vacías
    const orig = (baseRow && k in baseRow) ? baseRow[k] : row[k];
    if (t === "mm") row[k] = s === "" ? "" : (typeof orig === "number" ? Math.round(vNum(s)) : vMmText(s));
    else if (t === "pct01") row[k] = s === "" ? "" : Math.round(vNum(s) * 100) / 10000;
    else if (t === "text" || t === "numstr") row[k] = s;
    else { const n = Number(s.replace(",", ".")); row[k] = s === "" ? "" : (isNaN(n) ? s : n); }
  });
}

function isDirty(){
  if (!VIEW || !VIEW.vals || !VIEW.baselineVals) return false;
  return Object.keys(VIEW.vals).some(k => !sameVal(VIEW.types[k], VIEW.vals[k], VIEW.baselineVals[k]));
}

/* Lp y p1 dependen de las cotas editadas: si existen como columnas se mantienen coherentes */
function syncDerived(){
  if (!isDirty()) return;
  const m = viewModel();
  [["lp", m.Lp], ["p1", m.p1]].forEach(([name, n]) => {
    const k = Object.keys(VIEW.vals).find(x => x.toLowerCase() === name);
    if (k && vNum(VIEW.vals[k]) !== Math.round(n * 10) / 10) { VIEW.vals[k] = fmt(n); VIEW.needRebuild = true; }
  });
}

/* Comprueba la fila actual del visualizador contra /check. Devuelve el resultado (o null). */
async function checkViewerConnection(){
  if (!VIEW) return null;
  const v = VIEW;
  clearTimeout(v.checkTimer);
  v.pending = false;

  syncDerived();
  applyValsToRow(v.row, v.vals, v.types, v.rowBaseline, v.baselineVals);

  const beamName = v.row[v.keyOf["Perfil Viga (BEAM)"]], colName_ = v.row[v.keyOf["Perfil Pilar (COLUMN)"]];
  const pk = beamName + "|" + colName_;
  if (pk !== v.profKey) {
    const prof = await fetchProfiles([beamName, colName_]);
    if (VIEW !== v) return null;
    v.profKey = pk;
    v.beam = prof[beamName] || null; v.col = prof[colName_] || null;
    v.sec = {
      beam_d: v.beam ? v.beam.d_mm : "", beam_bf: v.beam ? v.beam.bf_mm : "",
      beam_tw: v.beam ? v.beam.tw_mm : "", beam_tf: v.beam ? v.beam.tf_mm : "",
      col_d: v.col ? v.col.d_mm : "", col_bf: v.col ? v.col.bf_mm : "",
      col_tw: v.col ? v.col.tw_mm : "", col_tf: v.col ? v.col.tf_mm : "",
    };
    v.needRebuild = true;
  }

  const token = (v.checkToken || 0) + 1;
  v.checkToken = token;
  v.checking = true;
  v.checkError = "";
  renderCheckSection();
  renderVarList();

  let result = null;
  try {
    const r = await fetch("/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: [v.row] }),
    });
    const data = await r.json().catch(() => ({}));
    if (VIEW !== v || v.checkToken !== token) return null;
    if (r.ok && Array.isArray(data.rows) && data.rows.length > 0) {
      const chk = data.rows[0];
      chk.cumple = evaluateCumple(chk);
      v.check = chk;
      result = chk;
    } else {
      v.check = null;                                          // nunca dejar visible un resultado antiguo
      const det = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
      v.checkError = `La comprobación falló (HTTP ${r.status}): ${String(det).slice(0, 300)}`;
    }
  } catch (e) {
    if (VIEW !== v || v.checkToken !== token) return null;
    v.check = null;
    v.checkError = "No se pudo contactar con el servidor: " + e.message;
  } finally {
    if (VIEW === v && v.checkToken === token) {
      v.checking = false;
      const ae = document.activeElement;
      const typing = ae && ae.tagName === "INPUT" && ae.closest && ae.closest("#viewerPanel");
      if (typing && !v.needRebuild) { renderCheckSection(); renderVarList(); }   // no robar el foco al escribir
      else buildViewerPanel();
      v.needRebuild = false;
      drawAll();
      updateSaveButton();
    }
  }
  return result;
}

function mmInput(v){
  const s = String(v == null ? "" : v).trim();
  if (s === "") return "";
  return String(Math.round(vNum(s)));
}

function sanitizeNum(v){
  let s = String(v == null ? "" : v).replace(/[^\d.,]/g, "").replace(/,/g, ".");
  const i = s.indexOf(".");
  if (i >= 0) s = s.slice(0, i + 1) + s.slice(i + 1).replace(/\./g, "");
  return s;
}

function markPending(){
  VIEW.pending = true;
  VIEW.checkError = "";
  renderCheckSection();
  renderVarList();
}

/* Mientras se escribe: se actualiza el dibujo y se programa una comprobación (campos numéricos) */
function onField(inp){
  if (!VIEW) return;
  const key = inp.dataset.vkey, t = VIEW.types[key] || "text";
  if (t === "text") {
    VIEW.vals[key] = inp.value;                 // los nombres de perfil/tornillo NO se filtran
  } else {
    const s = sanitizeNum(inp.value);
    VIEW.vals[key] = s;
    if (inp.value !== s) inp.value = s;
  }
  markPending();
  updateSaveButton();
  drawAll();
  if (t !== "text") {
    clearTimeout(VIEW.checkTimer);
    VIEW.checkTimer = setTimeout(() => checkViewerConnection(), 700);
  }
}

function onFieldBlur(inp){
  if (!VIEW) return;
  const key = inp.dataset.vkey, t = VIEW.types[key] || "text";
  if (t === "text") {
    const s = inp.value.trim();
    VIEW.vals[key] = s;
    if (inp.value !== s) inp.value = s;
  } else {
    const s = sanitizeNum(inp.value);
    let n = (s === "" || isNaN(parseFloat(s))) ? 0 : parseFloat(s);
    if (n < 0) n = 0;
    VIEW.vals[key] = String(n);
    if (inp.value !== String(n)) inp.value = String(n);
  }
  updateSaveButton();
  drawAll();
}

function updateSaveButton(){
  if (!VIEW) return;
  const btn = $("viewerSave");
  if (btn) btn.disabled = !isDirty();
}

function loadAdjustedPrediction(){
  if (!VIEW) return;
  const vars = getConnVariations(VIEW.connIdx);
  const activeVar = vars[VIEW.activeVarIndex];
  const adjusted = activeVar?.adjusted || activeVar?.data;
  if (!adjusted){ alert("No hay predicción ajustada disponible."); return; }

  VIEW.row = { ...VIEW.rowBaseline, ...adjusted };
  setVals(VIEW.row);
  VIEW.check = activeVar.adjustedCheck ? JSON.parse(JSON.stringify(activeVar.adjustedCheck)) : null;
  VIEW.pending = false; VIEW.checkError = "";

  buildViewerPanel();
  drawAll();
  updateSaveButton();
  checkViewerConnection();
}

function loadRawPrediction(){
  if (!VIEW) return;
  const vars = getConnVariations(VIEW.connIdx);
  const activeVar = vars[VIEW.activeVarIndex];
  const raw = activeVar?.raw;
  if (!raw){ alert("No hay predicción sin ajuste disponible para esta variación."); return; }

  VIEW.row = { ...VIEW.rowBaseline, ...raw };
  setVals(VIEW.row);
  VIEW.check = null;
  VIEW.pending = false; VIEW.checkError = "";

  buildViewerPanel();
  drawAll();
  updateSaveButton();
  checkViewerConnection();
}

/* Guardar: SIEMPRE vuelve a comprobar con los valores finales y espera la respuesta
   antes de cerrar el visor (antes se cerraba y el resultado nuevo se descartaba). */
async function saveViewer(){
  if (!VIEW) return;
  const v = VIEW;
  const btn = $("viewerSave");
  if (btn) btn.disabled = true;
  await checkViewerConnection();
  if (VIEW !== v) return;

  const vars = getConnVariations(v.connIdx);
  const activeVar = vars[v.activeVarIndex];
  if (!activeVar) return;

  Object.keys(v.row).forEach(k => { activeVar.data[k] = v.row[k]; });

  if (v.check) {
    activeVar.check = JSON.parse(JSON.stringify(v.check));
    activeVar.status = evaluateCumple(v.check) ? "CUMPLE" : "NO CUMPLE";
  } else {
    activeVar.check = null;
    activeVar.status = "MODIFICADO";
    setStatus("bad", "Se guardaron los cambios, pero la comprobación AISC falló.", v.checkError || "");
  }

  closeViewer();
  renderResults();
  renderAcceptedTable();
}

function closeViewer(){
  $("viewer").style.display = "none";
  document.body.classList.remove("modal-open");
  VIEW = null;
}

$("viewerBack").onclick = closeViewer;
$("viewerReset").onclick = loadAdjustedPrediction;
$("viewerRawReset").onclick = loadRawPrediction;
$("viewerSave").onclick = saveViewer;

/* ---------- Dibujo SVG ---------- */
function svgRect(x,y,w,h,fill,stroke,sw){ return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill||"none"}" stroke="${stroke||"black"}" stroke-width="${sw||1}"/>`; }
function svgCircle(cx,cy,r,fill){ return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="black" stroke-width="1"/>`; }
function svgLine(x1,y1,x2,y2,stroke,dash,sw){ return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke||"black"}" stroke-width="${sw||1}" stroke-dasharray="${dash||""}"/>`; }
function svgPoly(points,fill){ return `<polygon points="${points}" fill="${fill}" stroke="black" stroke-width="1"/>`; }
function svgText(x,y,t,anchor,color,size){ return `<text x="${x}" y="${y}" font-size="${size||11}" font-family="Segoe UI,Arial" fill="${color||"#1B2733"}" text-anchor="${anchor||"start"}">${escHtml(t)}</text>`; }
function dimV(x,y1,y2,label,side){
  side = side || "left";
  const tx = side === "left" ? x - 4 : x + 4, anc = side === "left" ? "end" : "start";
  return svgLine(x,y1,x,y2,"#D52B1E") + svgLine(x-3,y1,x+3,y1,"#D52B1E") + svgLine(x-3,y2,x+3,y2,"#D52B1E") + svgText(tx,(y1+y2)/2+4,label,anc,"#D52B1E");
}
function dimH(y,x1,x2,label){
  return svgLine(x1,y,x2,y,"#D52B1E") + svgLine(x1,y-3,x1,y+3,"#D52B1E") + svgLine(x2,y-3,x2,y+3,"#D52B1E") + svgText((x1+x2)/2,y-5,label,"middle","#D52B1E");
}

function viewModel(){
  const V = VIEW.vals, S = VIEW.sec;
  const tf = vNum(S.beam_tf), bf = vNum(S.beam_bf), tw = vNum(S.beam_tw), bd = vNum(S.beam_d);
  const ctf = vNum(S.col_tf), cbf = vNum(S.col_bf), ctw = vNum(S.col_tw), cd = vNum(S.col_d);
  const e = vNum(V.dev || V.e), g = vNum(V.p2_ep || V.g), Bp = vNum(V.b_ep || V.Bp), tp = vNum(V.ThicknessName_t_ep || V.tp), Lh = vNum(V.Lh);
  const pfo = vNum(V.pfo), pfi = vNum(V.pfi), pb = vNum(V.pb || V.p2);
  const p1 = pfo + tf + pfi;
  const steps = [p1, pb, ...P_KEYS.map(k => vNum(V[k]))].filter(x => x > 0);
  const sumP = steps.reduce((a,b) => a+b, 0);
  const Lp = 2*e + sumP;
  const rows = [e]; let y = e;
  steps.forEach(s => { y += s; rows.push(y); });
  const beamTop = e + pfo, beamBot = beamTop + bd;
  const boltDia = parseInt(String(VIEW.row.BoltName_ep || VIEW.row.BoltName || "").replace(/\D/g,""), 10) || 24;
  return {
    tf,bf,tw,bd,ctf,cbf,ctw,cd,e,g,Bp,tp,Lh,pfo,pfi,pb,p1,steps,sumP,Lp,rows,beamTop,beamBot,boltDia,
    Hhaunch: Math.max(0, Lp - beamBot),
    tws: vNum(V.ThicknessName_t_webstiff || V.t_webstiff), ts: vNum(V.ThicknessName_t_compstiff || V.t_compstiff),
    tbs: vNum(V.ThicknessName_t_BeamHcompstiff || V.t_BeamHcompstiff), tst: vNum(V.ThicknessName_t_tenstiff || V.t_tenstiff),
    bws: vNum(V.Bws), lws: vNum(V.Lws),
  };
}

function drawAll(){
  if (!VIEW) return;
  const m = viewModel();
  if ($("vP1")) $("vP1").value = fmt(m.p1);
  if ($("vLp")) $("vLp").value = fmt(m.Lp);
  drawFront(m); drawSide(m); drawTop(m);
}

function drawFront(m){
  const W = 520, H = 620, cx = 230, top = 40;
  const maxH = Math.max(m.Lp, m.beamBot, 1);
  const scale = Math.min(0.9, (H - 90)/maxH, (W - 200)/Math.max(m.Bp, 1));
  const Y = v => top + v*scale, X = v => cx + v*scale;
  let s = "";
  s += svgRect(cx - (m.Bp/2)*scale, top, m.Bp*scale, m.Lp*scale, "#cfd8e3");
  s += svgRect(cx - (m.bf/2)*scale, Y(m.beamTop), m.bf*scale, m.tf*scale, "none", "#33475b", 1.5);
  s += svgRect(cx - (m.tw/2)*scale, Y(m.beamTop + m.tf), m.tw*scale, Math.max(0,(m.bd - 2*m.tf))*scale, "none", "#33475b", 1.5);
  s += svgRect(cx - (m.bf/2)*scale, Y(m.beamBot - m.tf), m.bf*scale, m.tf*scale, "none", "#33475b", 1.5);
  if (m.Lh > 0){
    const hz = m.Lh;
    s += svgRect(cx - (m.tw/2)*scale, Y(m.beamBot), m.tw*scale, Math.max(0, hz - m.tf)*scale, "none", "#1e88e5", 1.5);
    s += svgRect(cx - (m.bf/2)*scale, Y(m.beamBot + hz - m.tf), m.bf*scale, m.tf*scale, "none", "#1e88e5", 1.5);
  }
  const r = Math.max(3, (m.boltDia/2)*scale*0.8);
  m.rows.forEach(yy => {
    s += svgCircle(X(-m.g/2), Y(yy), r, "#ff4500");
    s += svgCircle(X(m.g/2), Y(yy), r, "#ff4500");
  });
  const xL = cx - (m.Bp/2)*scale - 18;
  const xR = cx + (m.Bp/2)*scale + 22;
  s += dimV(xL, Y(0), Y(m.rows[0]), "e=" + fmt(m.e));
  s += dimV(xL, Y(m.rows[0]), Y(m.beamTop), "pfo=" + fmt(m.pfo));
  if (m.rows[1] != null && m.rows[1] > m.beamTop + m.tf)
    s += dimV(xL, Y(m.beamTop + m.tf), Y(m.rows[1]), "pfi=" + fmt(m.pfi));
  const pit = ["p2=" + fmt(m.pb), ...P_KEYS.map(k => k + "=" + fmt(vNum(VIEW.vals[k])))];
  for (let i = 1; i < m.rows.length - 1; i++)
    s += dimV(xL, Y(m.rows[i]), Y(m.rows[i+1]), pit[i-1] || "");
  s += dimV(xR, Y(m.beamTop), Y(m.beamBot), "d_viga=" + fmt(m.bd), "right");
  if (m.Lh > 0)
    s += dimV(xR, Y(m.beamBot), Y(m.beamBot + m.Lh), "Lh=" + fmt(m.Lh), "right");
  s += dimV(xR + 72, Y(0), Y(m.Lp), "Lp=" + fmt(m.Lp), "right");
  s += dimH(top - 14, X(-m.Bp/2), X(m.Bp/2), "Bp=" + fmt(m.Bp));
  s += dimH(Y(m.Lp) + 18, X(-m.g/2), X(m.g/2), "g=" + fmt(m.g));
  $("vFront").innerHTML = s;
}

function drawSide(m){
  const W = 580, H = 620, top = 40;
  const t = Math.tan(Math.PI / 3);
  const hxMM = m.Lh * t;
  let scale = Math.min(0.85, (H - 110)/Math.max(m.Lp, 1));
  const denom = m.cd + m.tp + Math.max(hxMM, 0);
  if (denom > 0) scale = Math.min(scale, (W - 160)/denom);
  scale = Math.max(scale, 0.12);
  const cRight = m.cd*scale + 90;
  const Y = v => top + v*scale;
  let s = "";
  const colTop = top - 20, colBot = top + m.Lp*scale + 20;
  s += svgRect(cRight - m.cd*scale, colTop, m.cd*scale, colBot - colTop, "#e6e6e6", "gray");
  s += svgRect(cRight - m.ctf*scale, colTop, m.ctf*scale, colBot - colTop, "#b0c4de");
  s += svgRect(cRight - m.cd*scale, colTop, m.ctf*scale, colBot - colTop, "#b0c4de");
  if (m.tws > 0 && m.bws > 0 && m.lws > 0){
    const wx = cRight - (m.cd/2)*scale - (m.bws/2)*scale;
    const wy = Y(m.beamTop + m.bd/2) - (m.lws/2)*scale;
    s += svgRect(wx, wy, m.bws*scale, m.lws*scale, "#4169e1", "#1e3a8a", 1.2);
    const bwsY = Math.min(wy, Y(m.beamTop)) - 18;
    s += dimH(bwsY, wx, wx + m.bws*scale, "Bws=" + fmt(m.bws));
    s += dimV(wx - 8, wy, wy + m.lws*scale, "Lws=" + fmt(m.lws), "left");
  }
  s += svgRect(cRight, top, m.tp*scale, m.Lp*scale, "#708090");
  const vigaX = cRight + m.tp*scale;
  const beamPx = W - vigaX;
  s += svgRect(vigaX, Y(m.beamTop), beamPx, m.tf*scale, "#b0c4de");
  s += svgRect(vigaX, Y(m.beamTop + m.tf), beamPx, Math.max(0,(m.bd - 2*m.tf))*scale, "#b0c4de");
  s += svgRect(vigaX, Y(m.beamBot - m.tf), beamPx, m.tf*scale, "#b0c4de");
  if (m.Lh > 0){
    const hx = vigaX + hxMM*scale;
    s += svgPoly(`${vigaX},${Y(m.beamBot)} ${hx},${Y(m.beamBot)} ${vigaX},${Y(m.beamBot + m.Lh)}`, "#a2b5cd");
  }
  if (m.tst > 0) s += svgRect(cRight - m.cd*scale + m.ctf*scale, Y(m.beamTop), Math.max(0,(m.cd - 2*m.ctf))*scale, m.tst*scale, "#00008b");
  if (m.ts > 0) s += svgRect(cRight - m.cd*scale + m.ctf*scale, Y(m.beamBot - m.ts), Math.max(0,(m.cd - 2*m.ctf))*scale, m.ts*scale, "#00008b");
  if (m.tbs > 0) s += svgRect(vigaX + beamPx - 30, Y(m.beamTop + m.tf), m.tbs*scale, Math.max(0,(m.bd - 2*m.tf))*scale, "#00008b");
  m.rows.forEach(yy => { s += svgLine(cRight - m.ctf*scale - 10, Y(yy), vigaX, Y(yy), "#ff4500", "4,2"); });
  const xL = cRight - m.cd*scale - 18;
  s += dimV(xL, Y(0), Y(m.rows[0]), "e=" + fmt(m.e));
  s += dimV(xL, Y(m.rows[0]), Y(m.beamTop), "pfo=" + fmt(m.pfo));
  if (m.Lh > 0)
    s += dimV(vigaX + hxMM*scale, Y(m.beamBot), Y(m.beamBot + m.Lh), "Lh=" + fmt(m.Lh), "right");
  s += dimH(top - 14, cRight, vigaX, "tp=" + fmt(m.tp));
  $("vSide").innerHTML = s;
}

function drawTop(m){
  const W = 500, cy = 180;
  const x0 = 50, beamPx = 190;
  const avail = W - x0 - beamPx - 70;
  const scale = Math.min(0.8, avail / Math.max(m.cd + m.tp, 1));
  const cx = x0 + m.cd*scale;
  const yTop = v => cy - v*scale;
  const steel = "#b0c4de", plate = "#708090", stiff = "#00008b", webStiff = "#4169e1";
  let s = "";
  s += svgRect(cx - m.cd*scale, cy - (m.cbf/2)*scale, m.ctf*scale, m.cbf*scale, steel, "#33475b", 1);
  s += svgRect(cx - m.ctf*scale, cy - (m.cbf/2)*scale, m.ctf*scale, m.cbf*scale, steel);
  s += svgRect(cx - m.cd*scale + m.ctf*scale, yTop(m.ctw/2), Math.max(0,(m.cd - 2*m.ctf))*scale, m.ctw*scale, steel);
  if (m.tws > 0) s += svgRect(cx - m.cd*scale + m.ctf*scale, yTop(m.ctw/2) - m.tws*scale, Math.max(0,(m.cd - 2*m.ctf))*scale, m.tws*scale, webStiff);
  s += svgRect(cx, cy - (m.Bp/2)*scale, m.tp*scale, m.Bp*scale, plate);
  s += svgRect(cx + m.tp*scale, cy - (m.bf/2)*scale, beamPx, m.bf*scale, steel);
  s += svgLine(cx + m.tp*scale, cy, cx + m.tp*scale + beamPx, cy, "black", "5,5");
  if (m.tbs > 0) s += svgRect(cx + m.tp*scale + beamPx - 30, cy - (m.bf/2)*scale, m.tbs*scale, m.bf*scale, stiff);
  s += dimH(cy + m.Bp*scale/2 + 16, cx, cx + m.tp*scale, "tp=" + fmt(m.tp));
  if (m.tbs > 0){
    const tx = cx + m.tp*scale + beamPx - 30 + m.tbs*scale;
    s += svgLine(tx, cy - (m.bf/2)*scale, tx + 18, cy - (m.bf/2)*scale - 12, "#D52B1E");
    s += svgText(tx + 20, cy - (m.bf/2)*scale - 14, "tbs=" + fmt(m.tbs), "start", "#D52B1E");
  }
  if (m.tws > 0){
    const wx = cx - m.cd*scale + (m.cd - 2*m.ctf)*scale*0.5;
    const wy = yTop(m.ctw/2) - m.tws*scale;
    s += svgLine(wx, wy, wx, wy - 18, "#D52B1E");
    s += svgText(wx + 4, wy - 20, "tws=" + fmt(m.tws), "start", "#D52B1E");
  }
  $("vTop").innerHTML = s;
}

/* =====================================================================
   Procesador de respuestas de POST /scenarios
   ===================================================================== */
function parseScenariosResponse(data, inputRow) {
  function buildCheck(r, fallback) {
    if (fallback) return fallback;
    const isCumple = evaluateCumple(r);
    return {
      cumple: isCumple,
      ratio_pesimo: r.ratio_pesimo ?? null,
      descripcion_ratio_pesimo: r.descripcion_ratio_pesimo || r.chequeo_critico || "",
      ratios: Array.isArray(r.ratios) ? r.ratios : [],
      complementarios: Array.isArray(r.complementarios) ? r.complementarios : []
    };
  }

  function getLabel(r, defaultIdx) {
    const raw = r.Escenario || r.escenario || r.descripcion || r.nombre || r.tipo || r.variacion || r.label;
    if (raw) return String(raw).trim();
    const defaults = ["E1 - Diseño de entrada", "E2 - Máximo sin cantonera", "E3 - Máximo con cantonera"];
    return defaults[defaultIdx] || `Variación ${defaultIdx + 1}`;
  }

  function getTipoCode(label, idx) {
    if (!label) return `E${idx + 1}`;
    if (label.includes(" - ")) return label.split(" - ")[0].trim();
    if (label.includes(":")) return label.split(":")[0].trim();
    return label.length <= 6 ? label : `E${idx + 1}`;
  }

  if (Array.isArray(data.rows) && data.rows.length > 0) {
    const hasMultipleConns = data.rows.some(r => r["Unicode origen"] || r.Unicode_origen);
    
    if (hasMultipleConns) {
      const groupedMap = new Map();
      data.rows.forEach((r, rowIdx) => {
        let key = r["Unicode origen"] || r.Unicode_origen || r.UNICODE_origen;
        if (!key && r.Unicode) key = r.Unicode.includes("_E") ? r.Unicode.split("_E")[0] : r.Unicode;
        if (!key) key = "Conexión_1";
        if (!groupedMap.has(key)) groupedMap.set(key, []);
        groupedMap.get(key).push({ row: r, rawIdx: rowIdx });
      });

      const connections = [];
      groupedMap.forEach((items, key) => {
        const vars = items.map((item, idx) => {
          const r = item.row;
          const fallbackCheck = (Array.isArray(data.checks) && data.checks[item.rawIdx]) ? data.checks[item.rawIdx] : null;
          const fallbackRaw = (Array.isArray(data.rows_raw) && data.rows_raw[item.rawIdx]) ? data.rows_raw[item.rawIdx] : { ...r };
          const lbl = getLabel(r, idx);
          const chk = buildCheck(r, fallbackCheck);
          const isCumple = evaluateCumple(chk);
          return {
            label: lbl,
            tipo: getTipoCode(lbl, idx),
            data: { ...r },
            adjusted: { ...r },
            raw: { ...fallbackRaw },
            check: chk,
            adjustedCheck: JSON.parse(JSON.stringify(chk)),
            status: isCumple ? "CUMPLE" : "NO CUMPLE",
            accepted: false
          };
        });
        connections.push({ inputRow: inputRow || {}, activeVarIndex: 0, variations: vars, marked: false });
      });
      return connections;
    }

    let vars = data.rows.map((v, idx) => {
      const lbl = getLabel(v, idx);
      const chk = buildCheck(v, data.checks ? data.checks[idx] : null);
      const raw = data.rows_raw ? data.rows_raw[idx] : { ...v };
      const isCumple = evaluateCumple(chk);
      return {
        label: lbl,
        tipo: getTipoCode(lbl, idx),
        data: { ...v },
        adjusted: { ...v },
        raw: { ...raw },
        check: chk,
        adjustedCheck: JSON.parse(JSON.stringify(chk)),
        status: isCumple ? "CUMPLE" : "NO CUMPLE",
        accepted: false
      };
    });

    /* Escenarios que el servidor no ha podido generar (p. ej. E3 sin ninguna cantonera):
       aparecen igualmente, como tarjeta no seleccionable con el motivo. */
    if (Array.isArray(data.escenarios) && data.escenarios.some(e => !e.encontrado)) {
      const merged = []; let ri = 0;
      data.escenarios.forEach(e => {
        if (e.encontrado) { if (vars[ri]) merged.push(vars[ri++]); }
        else {
          const lbl = `${e.id} - ${e.escenario}`;
          merged.push({
            label: lbl, tipo: e.id, data: {}, adjusted: {}, raw: {}, check: null, adjustedCheck: null,
            unavailable: true, sinCantonera: !!e.sin_opcion_cantonera,
            motivo: e.motivo || "Escenario no disponible.",
            status: e.sin_opcion_cantonera ? "SIN CANTONERA" : "NO ENCONTRADO", accepted: false
          });
        }
      });
      vars = merged;
    }

    return [{
      inputRow: inputRow || {},
      activeVarIndex: 0,
      variations: vars,
      marked: false
    }];
  }

  return [];
}

/* =====================================================================
   Acciones principales (Llamadas a POST /scenarios)
   ===================================================================== */
$("add").onclick = () => {
  const row = readForm(); if (!row) return;
  state.rows.push(row); renderTable(); renderForm();
  const first = SCHEMA.inputs[0];
  if (first) $("in_" + first.key).focus();
};
$("reset").onclick = renderForm;
$("clear").onclick = () => {
  state.rows = []; state.results = []; _checkTokens = {};
  renderTable(); renderResults(); setStatus("", "");
};

$("send").onclick = async () => {
  const btn = $("send");
  btn.disabled = true; 
  setStatus("", "Pre-diseñando " + state.rows.length + " combinaciones (calculando 3 variaciones por conexión)…");

  try {
    const model = $("model").value;

    const promises = state.rows.map(row => 
      fetch("/scenarios", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model,
          row: row,
          incremento: 0.05,
          max_pasos: 40
        })
      }).then(async r => {
        const json = await r.json().catch(() => ({}));
        return { ok: r.ok, status: r.status, data: json, inputRow: row };
      }).catch(err => ({ ok: false, error: err.message, inputRow: row }))
    );

    const responses = await Promise.all(promises);
    const parsedConns = [];
    let errorsCount = 0;

    responses.forEach(res => {
      if (res.ok && res.data) {
        const conns = parseScenariosResponse(res.data, res.inputRow);
        if (conns.length > 0) {
          parsedConns.push(...conns);
        }
      } else {
        errorsCount++;
      }
    });

    state.results = parsedConns;
    _checkTokens = {};
    renderResults();

    const totalVars = state.results.reduce((acc, c) => acc + getConnVariations(state.results.indexOf(c)).filter(v => !v.unavailable).length, 0);
    
    if (errorsCount > 0) {
      setStatus("bad", `Finalizado con ${errorsCount} errores.`, `${parsedConns.length} conexiones procesadas correctamente (${totalVars} variaciones).`);
    } else {
      setStatus("ok", `Predicción completada: ${parsedConns.length} conexiones (${totalVars} variaciones obtenidas). Pulsa 👁 para ver en 2D.`);
    }

  } catch (e){
    setStatus("bad", "Error de conexión", e.message);
  }
  btn.disabled = !state.rows.length;
};

/* =====================================================================
   Envío de Variantes Aceptadas al Dataset de Reentreno
   ===================================================================== */
$("sendAccepted").onclick = async () => {
  const rowsToSend = [];
  state.results.forEach((conn, connIdx) => {
    const vars = getConnVariations(connIdx);
    vars.forEach((v) => {
      if (v.accepted) rowsToSend.push(v.data);
    });
  });

  if (!rowsToSend.length){ setStatus("bad", "No hay variantes aceptadas seleccionadas."); return; }
  if (!confirm("¿Enviar " + rowsToSend.length + " variantes aceptadas al dataset de reentreno?")) return;

  const btn = $("sendAccepted"); btn.disabled = true;
  setStatus("", "Enviando " + rowsToSend.length + " variantes aceptadas…");
  try {
    const r = await fetch("/dataset/append", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: rowsToSend }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok){ setStatus("bad", `Error (${r.status})`, JSON.stringify(data)); }
    else {
      setStatus("ok", `Guardadas: ${data.added} nuevas, ${data.updated} actualizadas. Pendientes reentreno: ${data.retrain?.pending ?? 0}/${data.retrain?.threshold ?? 0}` + (data.retrain_started ? " · reentreno iniciado" : ""), data.rejected?.length ? "Rechazadas:\n" + data.rejected.join("\n") : "");
    }
  } catch (e){ setStatus("bad", "Error de conexión", e.message); }
  btn.disabled = false;
};

/* =====================================================================
   Inicialización
   ===================================================================== */
async function loadModels(){
  try{
    const r = await fetch("/models");
    const o = await r.json();
    const sel = $("model");
    const models = o.available_models || [];
    sel.innerHTML = models.map(m => `<option value="${escAttr(m)}">${escHtml(m)}</option>`).join("");
    if (o.default_model && models.includes(o.default_model)) sel.value = o.default_model;
  } catch(e){}
}

async function loadSchema(){
  const model = $("model").value;
  if (!model) return;
  try{
    const r = await fetch("/schema?model=" + encodeURIComponent(model));
    if (!r.ok) throw new Error("HTTP " + r.status);
    SCHEMA = await r.json();
    renderForm(); renderTable();
  } catch (e){ setStatus("bad", "Error cargando esquema", e.message); }
}

$("model").onchange = () => {
  state.rows = []; state.results = []; _checkTokens = {};
  renderTable(); renderResults(); loadSchema();
};

loadModels().then(loadSchema);