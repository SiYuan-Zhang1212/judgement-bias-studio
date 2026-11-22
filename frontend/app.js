(() => {
  const state = {
    subjects: [],
    curId: null,
    curStep: 'intake',
    plugins: [],
    intakeData: null,
    locData: null,
    selectedBadChs: new Set(),
    vol3dSeriesMap: new Map(),
    curVolUid: null,
    cameraStream: null,
    step4Available: true,
  };

  function applyStep4Availability(available) {
    state.step4Available = !!available;
    const banner = $('#step4UnavailBanner');
    const btnRun = $('#btnRunLocalization');
    const btnVol = $('#btnLoadVol3d');
    const shell = document.querySelector('#panel-localization .vol3d-shell');
    const volMsg = $('#volMsg');
    const disabled = !state.step4Available;

    if (banner) banner.style.display = disabled ? '' : 'none';
    if (btnRun) {
      btnRun.disabled = disabled;
      btnRun.style.opacity = disabled ? '0.45' : '';
      btnRun.style.cursor = disabled ? 'not-allowed' : '';
      btnRun.title = disabled ? 'Step 4 依赖 seeg-workbench，未安装。' : '';
    }
    if (btnVol) {
      btnVol.disabled = disabled;
      btnVol.style.opacity = disabled ? '0.45' : '';
      btnVol.style.cursor = disabled ? 'not-allowed' : '';
    }
    if (shell) {
      shell.style.opacity = disabled ? '0.45' : '';
      shell.style.pointerEvents = disabled ? 'none' : '';
      shell.querySelectorAll('select, input, button').forEach((ctrl) => {
        ctrl.disabled = disabled;
      });
    }
    if (disabled && volMsg) {
      volMsg.textContent = 'Step 4 依赖 seeg-workbench，未安装。';
    }
  }

  async function loadHealth() {
    const h = await api('/api/health');
    if (typeof h.step4_available === 'boolean') {
      applyStep4Availability(h.step4_available);
    }
  }

  function requireSubject() {
    if (!state.curId) {
      alert('请先在左侧选择一个被试，或点击「+ 导入新被试」。');
      return false;
    }
    return true;
  }

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const PALETTE = ['#ef4444','#22c55e','#a855f7','#eab308','#38bdf8','#ec4899','#84cc16','#6366f1','#f97316','#14b8a6','#d946ef','#a3e635','#3b82f6','#f43f5e','#06b6d4'];

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
  }

  function setBusy(on, text = '正在处理…') {
    $('#busyText').textContent = text;
    $('#busyOverlay').classList.toggle('hidden', !on);
  }

  function setStatus(text) {
    $('#globalStatus').textContent = text;
  }

  function artifactUrl(path) {
    return `/api/artifact?path=${encodeURIComponent(path)}&_=${Date.now()}`;
  }

  // ---------- Interactive SVG Chart Helpers (Step 2 Behavior) ----------
  function svgPsychometric(ps) {
    const W = 440, H = 220, pad = { l: 44, r: 18, t: 16, b: 44 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const xAt = (i) => pad.l + (i / 4) * pw;
    const yAt = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * ph;
    const pts = ps.map((p, i) => [xAt(i), yAt(p.p_pos ?? 0), p]);
    const poly = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    const midL = xAt(0.75).toFixed(1), midW = (xAt(3.25) - xAt(0.75)).toFixed(1);
    return `
      <svg class="chart-svg" viewBox="0 0 ${W} ${H}">
        <rect x="${midL}" y="${pad.t}" width="${midW}" height="${ph}" fill="rgba(56,189,248,0.08)" />
        ${[0, 0.25, 0.5, 0.75, 1].map((g) => `
          <line x1="${pad.l}" y1="${yAt(g)}" x2="${W - pad.r}" y2="${yAt(g)}" stroke="${g === 0.5 ? '#f59e0b' : '#24344f'}" stroke-dasharray="${g === 0.5 ? '4,3' : '2,3'}" />
          <text x="${pad.l - 6}" y="${yAt(g) + 4}" fill="#94a3b8" font-size="10" text-anchor="end">${(g * 100).toFixed(0)}%</text>
        `).join('')}
        <polyline fill="none" stroke="#38bdf8" stroke-width="2.6" points="${poly}" />
        ${pts.map(([x, y, p]) => `
          <circle cx="${x}" cy="${y}" r="5.5" fill="#0ea5e9" stroke="#f8fafc" stroke-width="1.5">
            <title>${p.label} (${p.orig_cls}, ${p.hz} Hz): P(pos) = ${((p.p_pos ?? 0) * 100).toFixed(1)}% (n=${p.n})</title>
          </circle>
          <text x="${x}" y="${y - 10}" fill="#e2e8f0" font-size="10" text-anchor="middle">${((p.p_pos ?? 0) * 100).toFixed(0)}%</text>
          <text x="${x}" y="${H - 22}" fill="#cbd5e1" font-size="10" text-anchor="middle">${p.label}</text>
          <text x="${x}" y="${H - 8}" fill="#64748b" font-size="9.5" text-anchor="middle">(${p.orig_cls})</text>
        `).join('')}
      </svg>`;
  }

  function svgRtBars(ps) {
    const W = 440, H = 220, pad = { l: 48, r: 16, t: 18, b: 44 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const maxRt = Math.max(1200, ...ps.map((p) => (p.rt_mean_ms || 0) + (p.rt_sd_ms || 0))) * 1.08;
    const yAt = (v) => pad.t + (1 - v / maxRt) * ph;
    const bw = pw / 5 * 0.56;
    return `
      <svg class="chart-svg" viewBox="0 0 ${W} ${H}">
        ${[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const val = maxRt * f;
          return `<line x1="${pad.l}" y1="${yAt(val)}" x2="${W - pad.r}" y2="${yAt(val)}" stroke="#24344f" stroke-dasharray="2,3" />
                  <text x="${pad.l - 6}" y="${yAt(val) + 4}" fill="#94a3b8" font-size="10" text-anchor="end">${Math.round(val)}</text>`;
        }).join('')}
        ${ps.map((p, i) => {
          const cx = pad.l + (i + 0.5) * (pw / 5);
          const mean = p.rt_mean_ms || 0, sd = p.rt_sd_ms || 0;
          const yTop = yAt(mean), yBot = yAt(0);
          const col = (i === 0 || i === 4) ? '#64748b' : '#38bdf8';
          return `
            <rect x="${cx - bw / 2}" y="${yTop}" width="${bw}" height="${Math.max(1, yBot - yTop)}" rx="4" fill="${col}" opacity="0.88">
              <title>${p.label} (${p.orig_cls}): ${mean.toFixed(1)} ± ${sd.toFixed(1)} ms</title>
            </rect>
            <line x1="${cx}" y1="${yAt(mean + sd)}" x2="${cx}" y2="${yAt(Math.max(0, mean - sd))}" stroke="#e2e8f0" stroke-width="1.3" />
            <line x1="${cx - 4}" y1="${yAt(mean + sd)}" x2="${cx + 4}" y2="${yAt(mean + sd)}" stroke="#e2e8f0" stroke-width="1.3" />
            <text x="${cx}" y="${yAt(mean + sd) - 5}" fill="#cbd5e1" font-size="9.5" text-anchor="middle">${Math.round(mean)}</text>
            <text x="${cx}" y="${H - 22}" fill="#cbd5e1" font-size="10" text-anchor="middle">${p.label}</text>
            <text x="${cx}" y="${H - 8}" fill="#64748b" font-size="9.5" text-anchor="middle">(${p.orig_cls})</text>`;
        }).join('')}
      </svg>`;
  }

  function svgBlocks(blks) {
    const W = 440, H = 220, pad = { l: 44, r: 18, t: 22, b: 36 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const n = Math.max(1, blks.length);
    const xAt = (i) => pad.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const yAt = (v) => pad.t + ((1 - v) / 2) * ph; // range -1..+1
    const biasPts = blks.map((b, i) => `${xAt(i)},${yAt(b.mids_bias_score ?? 0)}`).join(' ');
    const accPts = blks.map((b, i) => `${xAt(i)},${yAt(b.anchor_acc ?? 0)}`).join(' ');
    return `
      <svg class="chart-svg" viewBox="0 0 ${W} ${H}">
        ${[-1, -0.5, 0, 0.5, 1].map((g) => `
          <line x1="${pad.l}" y1="${yAt(g)}" x2="${W - pad.r}" y2="${yAt(g)}" stroke="${g === 0 ? '#64748b' : '#24344f'}" stroke-dasharray="2,3" />
          <text x="${pad.l - 6}" y="${yAt(g) + 4}" fill="#94a3b8" font-size="10" text-anchor="end">${g.toFixed(1)}</text>
        `).join('')}
        <polyline fill="none" stroke="#34d399" stroke-width="2.4" points="${biasPts}" />
        <polyline fill="none" stroke="#fbbf24" stroke-width="2.0" stroke-dasharray="5,3" points="${accPts}" />
        ${blks.map((b, i) => `
          <circle cx="${xAt(i)}" cy="${yAt(b.mids_bias_score ?? 0)}" r="4.5" fill="#34d399"><title>Block ${b.block} Bias = ${(b.mids_bias_score ?? 0).toFixed(3)}</title></circle>
          <circle cx="${xAt(i)}" cy="${yAt(b.anchor_acc ?? 0)}" r="4" fill="#fbbf24"><title>Block ${b.block} Anchor Acc = ${((b.anchor_acc ?? 0) * 100).toFixed(1)}%</title></circle>
          <text x="${xAt(i)}" y="${H - 12}" fill="#cbd5e1" font-size="10.5" text-anchor="middle">Block ${b.block}</text>
        `).join('')}
        <text x="${W - 18}" y="14" fill="#34d399" font-size="10" text-anchor="end">● Bias (-1..+1)   <tspan fill="#fbbf24">-- Anchor Acc (0..1)</tspan></text>
      </svg>`;
  }

  function svgMood(ms) {
    const W = 440, H = 220, pad = { l: 40, r: 18, t: 18, b: 44 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const n = Math.max(1, ms.length);
    const xAt = (i) => pad.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const yAt = (v) => pad.t + (1 - Math.max(0, Math.min(10, v)) / 10) * ph;
    const pts = ms.map((m, i) => `${xAt(i)},${yAt(m.valence ?? 5)}`).join(' ');
    return `
      <svg class="chart-svg" viewBox="0 0 ${W} ${H}">
        ${[0, 2.5, 5, 7.5, 10].map((g) => `
          <line x1="${pad.l}" y1="${yAt(g)}" x2="${W - pad.r}" y2="${yAt(g)}" stroke="#24344f" stroke-dasharray="2,3" />
          <text x="${pad.l - 6}" y="${yAt(g) + 4}" fill="#94a3b8" font-size="10" text-anchor="end">${g}</text>
        `).join('')}
        <polyline fill="none" stroke="#f472b6" stroke-width="2.4" points="${pts}" />
        ${ms.map((m, i) => `
          <circle cx="${xAt(i)}" cy="${yAt(m.valence ?? 5)}" r="4.8" fill="#f472b6" stroke="#fff" stroke-width="1.2">
            <title>${m.checkpoint}: ${m.valence}/10</title>
          </circle>
          <text x="${xAt(i)}" y="${yAt(m.valence ?? 5) - 9}" fill="#f8fafc" font-size="10" text-anchor="middle">${m.valence ?? '-'}</text>
          <text x="${xAt(i)}" y="${H - 16}" fill="#94a3b8" font-size="9" text-anchor="middle">${(m.checkpoint || '').replace('block_', 'B').replace('training', 'Train')}</text>
        `).join('')}
      </svg>`;
  }

  // ---------- 3D Volume Viewer Integration (Sinovation volume3d.js) ----------
  function getVol3d() {
    return typeof volume3d !== 'undefined' ? volume3d : null;
  }

  function syncElectrodesTo3D(uid) {
    const v3 = getVol3d();
    if (!v3 || !state.locData || !uid) return;
    const isCt = uid === state.locData.ct_uid;
    const pts = (state.locData.contacts || []).map((c) => {
      const xyz = isCt ? c.ct_lps : c.mr_lps;
      return { electrode: c.electrode, index: c.contact_index, x: xyz[0], y: xyz[1], z: xyz[2] };
    });
    v3.setPoints(uid, pts, true);
    const byE = new Map();
    pts.forEach((p) => { if (!byE.has(p.electrode)) byE.set(p.electrode, []); byE.get(p.electrode).push(p); });
    const trajs = [];
    let ci = 0;
    for (const [ename, arr] of byE.entries()) {
      arr.sort((a, b) => a.index - b.index);
      const first = arr[0], last = arr[arr.length - 1];
      trajs.push({
        name: ename,
        color: PALETTE[ci++ % PALETTE.length],
        source: 'detected',
        target_xyz: [first.x, first.y, first.z],
        entry_xyz: [last.x, last.y, last.z],
      });
    }
    v3.setTrajectories(trajs);
  }

  async function openSeriesIn3D(uid) {
    const v3 = getVol3d();
    if (!v3 || !uid) return;
    const meta = state.vol3dSeriesMap.get(uid);
    if (!meta) return;
    state.curVolUid = uid;
    syncElectrodesTo3D(uid);
    await v3.open(meta);
    syncElectrodesTo3D(uid);
  }

  // ---------- Subject Management ----------
  async function loadSubjects() {
    const res = await api('/api/subjects');
    state.subjects = res.subjects || [];
    renderSubjectList();
    if (!state.curId && state.subjects.length) {
      await selectSubject(state.subjects[0].subject_id);
    }
  }

  function renderSubjectList() {
    const el = $('#subjectList');
    if (!state.subjects.length) {
      el.innerHTML = '<div class="muted">暂无被试，请点击上方“导入新被试”。</div>';
      return;
    }
    el.innerHTML = state.subjects.map((s) => {
      const st = s.steps || {};
      const dot = (k, lbl) => `<span class="dot ${st[k]?.status === 'done' ? 'done' : ''}">${lbl}</span>`;
      return `
        <div class="sub-item ${s.subject_id === state.curId ? 'active' : ''}" data-sid="${s.subject_id}">
          <div class="sub-name">${s.display_name}</div>
          <div class="sub-id">${s.subject_id}</div>
          <div class="sub-dots">
            ${dot('intake', '体检')}
            ${dot('clinical', '临床')}
            ${dot('behavior', '行为')}
            ${dot('localization', '配准')}
            ${dot('preprocess', 'SEEG')}
          </div>
        </div>`;
    }).join('');
    $$('.sub-item').forEach((n) => { n.onclick = () => selectSubject(n.dataset.sid); });
  }

  async function selectSubject(sid) {
    state.curId = sid;
    renderSubjectList();
    const sub = state.subjects.find((s) => s.subject_id === sid);
    if (!sub) return;
    $('#curSubjectTitle').textContent = sub.display_name;
    const hasDoc = sub.paths?.electrode_excel ? ' · 含医生电极表' : ' · 无医生电极表(可选)';
    $('#curSubjectMeta').textContent = `ID: ${sub.subject_id}${hasDoc} · 目录: ${sub.paths?.root_dir || '自定义路径'}`;
    await loadCurrentSubjectPanels();
  }

  async function loadCurrentSubjectPanels() {
    if (!state.curId) return;
    const [intake, clin, beh, loc, pre] = await Promise.all([
      api(`/api/subjects/${state.curId}/intake`),
      api(`/api/subjects/${state.curId}/clinical`).catch(() => ({ data: null })),
      api(`/api/subjects/${state.curId}/behavior`),
      api(`/api/subjects/${state.curId}/localization`),
      api(`/api/subjects/${state.curId}/preprocess`),
    ]);
    if (clin && clin.data) renderClinical(clin.data);
    if (intake.exists) {
      state.intakeData = intake.data;
      renderIntake(intake.data);
      populateSeriesSelectors(intake.data.dicom);
      renderBadChannelSelector(intake.data.edf);
    } else {
      state.intakeData = null;
      $('#intakeBody').innerHTML = '<div class="empty-hint">点击右上角“一键体检与触发对齐”开始自动检查该被试的多模态数据。</div>';
      $('#badChannelBox').innerHTML = '';
    }
    if (beh.exists) renderBehavior(beh.data);
    else $('#behaviorBody').innerHTML = '<div class="empty-hint">点击右上角按钮生成交互式行为学图表与指标。</div>';

    if (loc.exists) {
      state.locData = loc.data;
      renderLocalization(loc.data);
    } else {
      state.locData = null;
      $('#localizationBody').innerHTML = '<div class="empty-hint">建议先完成 Step 1 体检，再点击「一键检测与配准」。</div>';
    }

    if (pre.exists) renderPreprocess(pre.data);
    else $('#preprocessBody').innerHTML = '<div class="empty-hint">保持 PNAS 2025 默认参数（或按需微调）后点击一键运行预处理。</div>';
  }

  function populateSeriesSelectors(dicom) {
    if (!dicom) return;
    state.vol3dSeriesMap.clear();
    const mrList = dicom.mr_series || [];
    const ctList = dicom.ct_series || [];
    mrList.forEach((s) => state.vol3dSeriesMap.set(s.uid, { ...s, kind: 'mr' }));
    ctList.forEach((s) => state.vol3dSeriesMap.set(s.uid, { ...s, kind: 'ct' }));

    $('#selMrUid').innerHTML = mrList.map((s) =>
      `<option value="${s.uid}" ${s.uid === dicom.recommended_mr_uid ? 'selected' : ''}>${s.description || 'MR'} (${s.num_slices}层)</option>`
    ).join('');
    $('#selCtUid').innerHTML = ctList.map((s) =>
      `<option value="${s.uid}" ${s.uid === dicom.recommended_ct_uid ? 'selected' : ''}>${s.description || 'CT'} (${s.num_slices}层)</option>`
    ).join('');
    $('#selVol3dSeries').innerHTML = [
      ...mrList.map((s) => `<option value="${s.uid}" ${s.uid === dicom.recommended_mr_uid ? 'selected' : ''}>[MR] ${s.description || 'MR'} (${s.num_slices}层)</option>`),
      ...ctList.map((s) => `<option value="${s.uid}">[CT] ${s.description || 'CT'} (${s.num_slices}层)</option>`),
    ].join('');
  }

  function renderIntake(d) {
    const sync = d.trigger_sync?.summary || {};
    const clk = d.trigger_sync?.clock_model || {};
    const edf = d.edf || {};
    const shafts = edf.shafts || [];
    const doc = d.doctor_table;
    const syncBanner = sync.sync_ok === false
      ? `<div class="err-banner">⛔ <strong>触发对齐或 EDF 连续性未通过（已锁定预处理）：</strong>${sync.sync_error || '请检查触发通道'}</div>`
      : '';
    const recBadge = (sync.tone_wall_clock_recovered || 0) > 0
      ? `<span class="badge badge-amber">墙钟补齐 ${sync.tone_wall_clock_recovered} 个漏标</span>`
      : `<span class="badge badge-green">零漏标</span>`;
    const docHtml = doc && !doc.error
      ? `<span class="badge badge-green">已加载医生表 (${doc.n_electrodes} 根)</span>`
      : `<span class="badge badge-blue">未提供（可选，可继续）</span>`;

    $('#intakeBody').innerHTML = `
      ${syncBanner}
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">SEEG 通道 / 电极杆</div><div class="kpi-val">${edf.n_seeg || 0} / ${shafts.length}</div><div class="kpi-sub">${edf.sfreq || 0} Hz · ${((edf.duration_s || 0) / 60).toFixed(1)} min · ${sync.edf_has_discontinuity ? '⚠有间断' : '连续无间断'}</div></div>
        <div class="kpi"><div class="kpi-label">声音触发对齐</div><div class="kpi-val">${sync.tone_hardware_matched || 0} / ${sync.n_trials || 0}</div><div class="kpi-sub">${recBadge}</div></div>
        <div class="kpi"><div class="kpi-label">按键触发对齐</div><div class="kpi-val">${sync.resp_hardware_matched || 0}</div><div class="kpi-sub">硬件-JSON RT 差 ${sync.median_hw_vs_json_rt_diff_ms ?? '-'} ms</div></div>
        <div class="kpi"><div class="kpi-label">时钟对齐中位残差</div><div class="kpi-val kpi-ok">${sync.median_clock_resid_ms ?? '-'} ms</div><div class="kpi-sub">晶振漂移 ${clk?.clock_drift_ppm ?? '-'} ppm</div></div>
        <div class="kpi"><div class="kpi-label">坏道初筛 / 高幅提醒</div><div class="kpi-val"><span class="${(edf.auto_bad_channels||[]).length ? 'kpi-warn' : 'kpi-ok'}">${(edf.auto_bad_channels||[]).length}</span> / <span class="kpi-warn">${(edf.review_warning_channels||[]).length}</span></div><div class="kpi-sub">仅明确坏道默认剔除</div></div>
        <div class="kpi"><div class="kpi-label">医生电极位置表</div><div class="kpi-val" style="font-size:13px;">${docHtml}</div><div class="kpi-sub">术后核对优先采信</div></div>
      </div>
      <div class="section-label">EDF 电极杆清单（自动识别缺失跳号）</div>
      <table class="data-table">
        <thead><tr><th>电极杆</th><th>半球</th><th>有效触点数</th><th>编号范围</th><th>缺失/跳号触点</th></tr></thead>
        <tbody>
          ${shafts.map((sh) => `
            <tr>
              <td><strong>${sh.shaft}</strong></td>
              <td>${sh.side}</td>
              <td>${sh.n_contacts}</td>
              <td>${sh.min_index} – ${sh.max_index}</td>
              <td>${sh.missing_indices.length ? `<span class="badge badge-amber">${sh.missing_indices.join(', ')}</span>` : '<span class="muted">连续无缺失</span>'}</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
  }

  function renderBehavior(d) {
    const m = d.metrics || {};
    const vm = d.valence_mapping || {};
    const art = d.artifacts || {};
    $('#behaviorBody').innerHTML = `
      <div style="margin-bottom:12px;">
        <span class="badge badge-blue">正性锚点：${vm.pos_anchor_cls} (${vm.pos_anchor_hz} Hz · ${vm.pos_key}键)</span>
        <span class="badge badge-blue" style="margin-left:6px;">负性锚点：${vm.neg_anchor_cls} (${vm.neg_anchor_hz} Hz · ${vm.neg_key}键)</span>
        <span class="badge ${d.qc_pass ? 'badge-green' : 'badge-amber'}" style="margin-left:6px;">${d.qc_pass ? '行为 QC 通过' : '需复核: ' + (d.qc_flags || []).join('; ')}</span>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">判断偏倚 (Bias Score)</div><div class="kpi-val">${m.bias_score != null ? (m.bias_score >= 0 ? '+' : '') + m.bias_score.toFixed(3) : '-'}</div><div class="kpi-sub">P(pos|Mid) = ${m.p_pos_on_mids != null ? (m.p_pos_on_mids * 100).toFixed(1) + '%' : '-'}</div></div>
        <div class="kpi"><div class="kpi-label">锚点辨别力 (d')</div><div class="kpi-val kpi-ok">${m.dprime ?? '-'}</div><div class="kpi-sub">Criterion c = ${m.criterion ?? '-'}</div></div>
        <div class="kpi"><div class="kpi-label">锚点正确率 (Pos / Neg)</div><div class="kpi-val">${m.acc_pos_anchor != null ? (m.acc_pos_anchor * 100).toFixed(1) + '%' : '-'} / ${m.acc_neg_anchor != null ? (m.acc_neg_anchor * 100).toFixed(1) + '%' : '-'}</div><div class="kpi-sub">遗漏率 ${((m.omission_rate || 0) * 100).toFixed(1)}%</div></div>
        <div class="kpi"><div class="kpi-label">反应时冲突效应</div><div class="kpi-val">${m.rt_conflict_ms != null ? (m.rt_conflict_ms >= 0 ? '+' : '') + m.rt_conflict_ms + ' ms' : '-'}</div><div class="kpi-sub">模糊 ${m.rt_mids_mean_ms} vs 锚点 ${m.rt_anchor_mean_ms}</div></div>
        <div class="kpi"><div class="kpi-label">情绪均值 (VAS 0-10)</div><div class="kpi-val">${m.mood_mean ?? '-'}</div><div class="kpi-sub">前后变化 ${m.mood_delta ?? '-'}</div></div>
      </div>

      <!-- Inline Interactive SVG Charts -->
      <div class="charts-grid-2">
        <div class="chart-card"><h4>A. 心理测量曲线（P("正性")，悬停查看数值）</h4>${svgPsychometric(d.psychometric || [])}</div>
        <div class="chart-card"><h4>B. 各音高反应时分布（Mean ± SD, ms）</h4>${svgRtBars(d.psychometric || [])}</div>
        <div class="chart-card"><h4>C. 分 Block 偏倚与锚点正确率稳定性</h4>${svgBlocks(d.by_block || [])}</div>
        <div class="chart-card"><h4>D. 7 点情绪探针轨迹（VAS 0–10）</h4>${svgMood(d.mood_series || [])}</div>
      </div>

      <div class="download-row">
        ${art.overview_png ? `<a class="dl-link" href="${artifactUrl(art.overview_png)}" download>下载高清出版级 2×2 图 (PNG)</a>` : ''}
        ${art.trials_csv ? `<a class="dl-link" href="${artifactUrl(art.trials_csv)}" download>下载标准化试次表 (behavior_trials.csv)</a>` : ''}
        ${art.summary_json ? `<a class="dl-link" href="${artifactUrl(art.summary_json)}" download>下载汇总 JSON</a>` : ''}
      </div>`;
  }

  function renderLocalization(d) {
    const elecs = d.electrodes || [];
    const shafts = d.edf_shafts_available || [];
    const doc = d.doctor_table;
    const art = d.artifacts || {};
    const warns = d.qc_warnings || [];
    const bindOptions = [
      ...((doc && !doc.error ? doc.electrodes : []) || []).map((de) => ({ val: de.name, lbl: `[医生表] ${de.name} (${de.n_contacts}点 · ${de.target || ''}→${de.entry || ''})` })),
      ...shafts.map((sh) => ({ val: sh.shaft, lbl: `[EDF] ${sh.shaft} (${sh.side}, ${sh.n_contacts}触点)` })),
    ];

    $('#localizationBody').innerHTML = `
      ${warns.length ? `<div class="warn-banner">${warns.map((w) => `⚠ ${w}`).join('<br/>')}</div>` : ''}
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">CT 检出电极杆</div><div class="kpi-val">${d.n_electrodes}</div><div class="kpi-sub">HU ≥ ${d.hu_threshold}</div></div>
        <div class="kpi"><div class="kpi-label">配准触点总数</div><div class="kpi-val">${d.n_contacts}</div><div class="kpi-sub">间距 ${d.contact_spacing_mm} mm</div></div>
        <div class="kpi"><div class="kpi-label">医生电极表</div><div class="kpi-val" style="font-size:14px;">${doc && !doc.error ? `已接入 (${doc.n_electrodes}根)` : '未提供（近似 MNI）'}</div><div class="kpi-sub">${d.mr_description || 'T1 MR'}</div></div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
        <div class="section-label">CT 电极杆 → 医生表 / EDF 绑定（保存后按实际触点数重算触点并刷新 3D）</div>
        <button id="btnSaveBindings" class="btn btn-primary btn-sm">保存绑定并重算触点</button>
      </div>
      <table class="data-table">
        <thead><tr><th>CT 编号</th><th>半球</th><th>触点数</th><th>深端 / 浅端距头表</th><th>尖端(#1)脑区</th><th>绑定 医生表 / EDF 电极名</th></tr></thead>
        <tbody>
          ${elecs.map((e) => `
            <tr>
              <td><strong>${e.electrode}</strong></td>
              <td>${e.hemisphere}</td>
              <td>${e.n_contacts} <span class="muted" style="font-size:10.5px;">(${e.contact_count_source === 'doctor_table' ? '医生表' : (e.contact_count_source === 'edf_shaft' ? 'EDF' : '估算')})</span></td>
              <td>${e.tip_surface_depth_mm ?? '-'} mm / ${e.entry_surface_depth_mm ?? '-'} mm</td>
              <td><span class="badge ${e.tip_region.includes('医生核对') ? 'badge-green' : 'badge-blue'}">${e.tip_region}</span></td>
              <td>
                <select class="bind-sel" data-elec="${e.electrode}">
                  <option value="">-- 未绑定 --</option>
                  ${bindOptions.map((o) => `<option value="${o.val}" ${e.edf_shaft_binding === o.val ? 'selected' : ''}>${o.lbl}</option>`).join('')}
                </select>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>
      <div class="download-row">
        ${art.contacts_csv ? `<a class="dl-link" href="${artifactUrl(art.contacts_csv)}" download>下载触点坐标与脑区表 (electrode_contacts_mni.csv)</a>` : ''}
      </div>`;

    const btn = $('#btnSaveBindings');
    if (btn) {
      btn.onclick = async () => {
        const bindings = {};
        $$('.bind-sel').forEach((s) => { if (s.value) bindings[s.dataset.elec] = s.value; });
        const res = await api(`/api/subjects/${state.curId}/localization/bindings`, {
          method: 'POST',
          body: JSON.stringify({ bindings }),
        });
        if (res.data) {
          state.locData = res.data;
          renderLocalization(res.data);
          if (state.curVolUid) syncElectrodesTo3D(state.curVolUid);
        }
        setStatus('绑定已保存：已按实际触点数重算触点并写入医生核对脑区');
      };
    }
  }

  function renderBadChannelSelector(edf) {
    const box = $('#badChannelBox');
    if (!edf || !edf.channel_qc) { box.innerHTML = ''; return; }
    // ONLY auto-select hard_bad channels; NEVER auto-drop physiological high_variance_review (e.g. Hippocampus)!
    state.selectedBadChs = new Set(edf.auto_bad_channels || []);
    const nWarn = (edf.review_warning_channels || []).length;
    box.innerHTML = `
      <div class="section-label">坏道人工确认（<span style="color:var(--red);">红框=明确坏道默认剔除 ${(edf.auto_bad_channels || []).length} 个</span>；<span style="color:var(--amber);">黄框=高幅值提醒 ${nWarn} 个，不默认剔除，如海马通道请结合脑区核对</span>）</div>
      <div class="chip-list" id="badChips">
        ${edf.channel_qc.map((c) => {
          const isSel = state.selectedBadChs.has(c.channel);
          const cls = isSel ? 'selected' : (c.review_warning ? 'warn-only' : '');
          const reg = c.doctor_region ? ` · ${c.doctor_region}` : '';
          const mark = c.auto_bad ? ' ✖' : (c.review_warning ? ' ⚡' : '');
          return `<span class="ch-chip ${cls}" data-ch="${c.channel}" data-warn="${c.review_warning ? '1' : '0'}" title="SD=${c.std_uv}uV | 50Hz比=${c.line_50hz_ratio} | 峰度=${c.kurtosis} | 原因=${(c.reasons||[]).join(',') || '正常'}">${c.channel.replace('POL ', '')}${reg}${mark}</span>`;
        }).join('')}
      </div>`;
    $$('#badChips .ch-chip').forEach((el) => {
      el.onclick = () => {
        const ch = el.dataset.ch;
        if (state.selectedBadChs.has(ch)) {
          state.selectedBadChs.delete(ch);
          el.classList.remove('selected');
          if (el.dataset.warn === '1') el.classList.add('warn-only');
        } else {
          state.selectedBadChs.add(ch);
          el.classList.remove('warn-only');
          el.classList.add('selected');
        }
      };
    });
  }

  function renderPreprocess(d) {
    const art = d.artifacts || {};
    const iqr = d.iqr_stats || {};
    $('#preprocessBody').innerHTML = `
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">保留 SEEG 通道</div><div class="kpi-val kpi-ok">${d.n_channels_retained}</div><div class="kpi-sub">剔除坏道 ${d.n_bad_channels} 个</div></div>
        <div class="kpi"><div class="kpi-label">保留干净试次</div><div class="kpi-val kpi-ok">${d.n_epochs_retained} / ${d.n_epochs_initial}</div><div class="kpi-sub">IQR 剔除 ${d.n_epochs_rejected_iqr} 试次</div></div>
        <div class="kpi"><div class="kpi-label">重参考 / 采样率</div><div class="kpi-val" style="font-size:16px;">${(d.params?.reref || '').toUpperCase()} · ${d.output_sfreq} Hz</div><div class="kpi-sub">50Hz FIR + ${d.params?.hp_freq}Hz HP</div></div>
        <div class="kpi"><div class="kpi-label">逐通道 IQR 剔除</div><div class="kpi-val">w = ${iqr.weight ?? 2.3}</div><div class="kpi-sub">试次最大偏离中位 ${iqr.median_max_iqr_score ?? '-'}×IQR</div></div>
      </div>
      ${art.qc_png ? `<div class="fig-wrap"><img src="${artifactUrl(art.qc_png)}" alt="Preprocess QC" /></div>` : ''}
      <div class="download-row">
        ${art.epochs_fif ? `<a class="dl-link" href="${artifactUrl(art.epochs_fif)}" download>下载标准 MNE Epochs (.fif)</a>` : ''}
        ${art.metadata_csv ? `<a class="dl-link" href="${artifactUrl(art.metadata_csv)}" download>下载试次 Metadata CSV</a>` : ''}
        ${art.qc_json ? `<a class="dl-link" href="${artifactUrl(art.qc_json)}" download>下载预处理 QC 报告 JSON</a>` : ''}
      </div>`;
  }

  async function loadPlugins() {
    const res = await api('/api/plugins');
    state.plugins = res.plugins || [];
    $('#pluginsBody').innerHTML = state.plugins.map((p) => `
      <div class="plugin-card" data-pid="${p.plugin_id}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3>${p.name}</h3>
          <span class="badge badge-blue">v${p.version}</span>
        </div>
        <p class="muted">${p.description}</p>
        <div class="plugin-params">
          ${(p.params_schema || []).map((f) => {
            if (f.type === 'select') {
              return `<label>${f.label}<select data-pkey="${f.key}">${f.options.map((o) => `<option value="${o.value}" ${o.value === f.default ? 'selected' : ''}>${o.label}</option>`).join('')}</select></label>`;
            }
            return `<label>${f.label}<input data-pkey="${f.key}" type="${f.type === 'number' ? 'number' : 'text'}" value="${f.default ?? ''}" step="any" /></label>`;
          }).join('')}
        </div>
        <button class="btn btn-primary btn-run-plugin" data-pid="${p.plugin_id}">运行插件</button>
        <div class="plugin-out" id="pout-${p.plugin_id}"></div>
      </div>`).join('');

    $$('.btn-run-plugin').forEach((btn) => {
      btn.onclick = async () => {
        if (!state.curId) return alert('请先选择被试');
        const pid = btn.dataset.pid;
        const card = btn.closest('.plugin-card');
        const params = {};
        card.querySelectorAll('[data-pkey]').forEach((inp) => {
          params[inp.dataset.pkey] = inp.type === 'number' ? Number(inp.value) : inp.value;
        });
        setBusy(true, `正在运行插件：${pid}…`);
        try {
          const res = await api(`/api/subjects/${state.curId}/plugins/${pid}/run`, {
            method: 'POST',
            body: JSON.stringify({ params }),
          });
          const out = res.result || {};
          $(`#pout-${pid}`).innerHTML = `
            <div class="muted" style="color:var(--green); margin-bottom:6px;">${out.summary || '运行完成'}</div>
            ${out.image_path ? `<div class="fig-wrap"><img src="${artifactUrl(out.image_path)}" /></div>` : ''}`;
          setStatus(`插件 ${pid} 运行完成`);
        } catch (err) { alert(`插件运行失败：${err.message}`); }
        finally { setBusy(false); }
      };
    });
  }

  // ---------- AI Assistant (OpenCode Go + DeepSeek Flash 1017) ----------
  async function loadAiConfig() {
    const cfg = await api('/api/ai/config');
    $('#aiBaseUrl').value = cfg.base_url || 'https://opencode.ai/zen/go/v1';
    $('#aiModel').value = cfg.model || 'deepseek-v4.1-flash-1017';
    $('#aiModelBadge').textContent = `OpenCode Go · ${cfg.model}`;
    $('#aiKeyStatus').textContent = cfg.has_api_key
      ? `当前已保存 Key：${cfg.api_key_masked}（留空保存将保留原 Key）`
      : '尚未配置 API Key，请粘贴您的 OpenCode Go Key。';
  }

  function appendAiMsg(role, text) {
    const box = $('#aiMessages');
    const div = document.createElement('div');
    div.className = `ai-msg ${role}`;
    div.textContent = text;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }

  async function sendAiQuestion(question, stepOverride) {
    if (!question.trim()) return;
    $('#aiDrawer').classList.remove('hidden');
    appendAiMsg('user', question);
    const pending = appendAiMsg('bot', 'DeepSeek 思考中…');
    try {
      const res = await api('/api/ai/ask', {
        method: 'POST',
        body: JSON.stringify({
          subject_id: state.curId || '',
          step: stepOverride || state.curStep,
          question,
        }),
      });
      pending.textContent = res.answer || '（无返回内容）';
    } catch (e) {
      pending.textContent = `请求失败：${e.message}`;
    }
  }

  // ---------- Clinical Data (Photos / Files + Camera + AI Extract) ----------
  function renderClinical(d) {
    if (!d) return;
    const st = d.structured || {};
    $$('[data-ckey]').forEach((el) => {
      el.value = st[el.dataset.ckey] ?? '';
    });
    const noteBox = $('#clinAiNoteBox');
    if (d.ai_extract_notes) {
      noteBox.textContent = `✨ AI 提取说明：${d.ai_extract_notes}`;
      noteBox.classList.remove('hidden');
    } else {
      noteBox.classList.add('hidden');
    }

    const files = d.files || [];
    const listEl = $('#clinFilesList');
    if (!files.length) {
      listEl.innerHTML = '<div class="muted">暂无上传文件，可点击上方「上传文件 / 照片」或「📷 摄像头拍照」。</div>';
      return;
    }
    listEl.innerHTML = files.map((f) => `
      <div class="clin-file-card">
        ${f.is_image
          ? `<a href="${artifactUrl(f.path)}" target="_blank"><img src="${artifactUrl(f.path)}" alt="${f.name}" /></a>`
          : `<a href="${artifactUrl(f.path)}" target="_blank" style="text-decoration:none;"><div class="clin-file-icon">📄</div></a>`}
        <div class="clin-file-meta">
          <span class="clin-file-name" title="${f.name}">${f.name}</span>
          <button class="btn btn-ghost btn-sm btn-del-clin-file" data-fname="${f.name}" title="删除">✕</button>
        </div>
      </div>`).join('');

    $$('.btn-del-clin-file').forEach((btn) => {
      btn.onclick = async () => {
        if (!requireSubject()) return;
        const res = await api(`/api/subjects/${state.curId}/clinical/files/${encodeURIComponent(btn.dataset.fname)}`, { method: 'DELETE' });
        renderClinical(res.data);
        setStatus(`已删除临床文件 ${btn.dataset.fname}`);
      };
    });
  }

  function readAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result || ''));
      r.onerror = () => reject(r.error || new Error('读取文件失败'));
      r.readAsDataURL(file);
    });
  }

  async function uploadClinicalBlobList(fileList) {
    if (!requireSubject() || !fileList || !fileList.length) return;
    setBusy(true, '正在上传临床资料…');
    try {
      const items = [];
      for (const f of fileList) {
        const data_base64 = await readAsDataURL(f);
        items.push({ name: f.name || `photo_${Date.now()}.jpg`, data_base64 });
      }
      const data = await api(`/api/subjects/${state.curId}/clinical/upload`, {
        method: 'POST',
        body: JSON.stringify({ files: items }),
      });
      renderClinical(data.data);
      await loadSubjects();
      setStatus(`已上传 ${data.uploaded.length} 份临床资料`);
    } catch (e) { alert(`上传失败：${e.message}`); }
    finally { setBusy(false); }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
    }
    $('#cameraBox').classList.add('hidden');
  }

  $('#btnClinUpload').onclick = () => {
    if (!requireSubject()) return;
    $('#clinFileInput').value = '';
    $('#clinFileInput').click();
  };
  $('#clinFileInput').onchange = (e) => uploadClinicalBlobList(e.target.files);

  $('#btnClinCamera').onclick = async () => {
    if (!requireSubject()) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return alert('当前浏览器不支持直接调用摄像头，请使用「上传文件 / 照片」选择图片。');
    }
    try {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      $('#cameraVideo').srcObject = state.cameraStream;
      $('#cameraBox').classList.remove('hidden');
    } catch (e) {
      alert(`无法打开摄像头（可在系统设置授权或改用上传照片）：${e.message}`);
    }
  };
  $('#btnStopCamera').onclick = () => stopCamera();
  $('#btnSnapPhoto').onclick = () => {
    const video = $('#cameraVideo');
    const canvas = $('#cameraCanvas');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const file = new File([blob], `clinical_photo_${Date.now()}.jpg`, { type: 'image/jpeg' });
      await uploadClinicalBlobList([file]);
      stopCamera();
    }, 'image/jpeg', 0.9);
  };

  $('#btnClinAiExtract').onclick = async () => {
    if (!requireSubject()) return;
    setBusy(true, '正在调用 OpenCode Go DeepSeek 视觉/文本模型提取临床结构化信息…');
    try {
      const res = await api(`/api/subjects/${state.curId}/clinical/ai_extract`, {
        method: 'POST',
        body: JSON.stringify({
          extra_note: $('#clinExtraNote').value || '',
          overwrite: false,
        }),
      });
      renderClinical(res.data);
      await loadSubjects();
      setStatus(`AI 已自动提取临床信息（模型：${res.model}）`);
    } catch (e) { alert(`AI 提取失败：${e.message}`); }
    finally { setBusy(false); }
  };

  $('#btnSaveClinical').onclick = async () => {
    if (!requireSubject()) return;
    const structured = {};
    $$('[data-ckey]').forEach((el) => { structured[el.dataset.ckey] = el.value.trim(); });
    setBusy(true, '正在保存临床信息…');
    try {
      const res = await api(`/api/subjects/${state.curId}/clinical`, {
        method: 'PUT',
        body: JSON.stringify({ structured }),
      });
      renderClinical(res.data);
      await loadSubjects();
      setStatus('临床信息已保存并同步至汇总表');
    } catch (e) { alert(`保存失败：${e.message}`); }
    finally { setBusy(false); }
  };

  // ---------- Event Bindings ----------
  $$('.step-tab').forEach((btn) => {
    btn.onclick = () => {
      state.curStep = btn.dataset.step;
      $$('.step-tab').forEach((b) => b.classList.toggle('active', b === btn));
      $$('.step-panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${state.curStep}`));
      if (state.curStep === 'localization' && state.step4Available && !state.curVolUid) {
        const targetUid = state.locData?.mr_uid || $('#selVol3dSeries').value;
        if (targetUid) openSeriesIn3D(targetUid);
      }
    };
  });

  $('#btnLoadVol3d').onclick = () => {
    if (!state.step4Available) return;
    openSeriesIn3D($('#selVol3dSeries').value);
  };
  $('#selVol3dSeries').onchange = () => {
    if (!state.step4Available) return;
    openSeriesIn3D($('#selVol3dSeries').value);
  };

  $('#btnRunIntake').onclick = async () => {
    if (!requireSubject()) return;
    setBusy(true, '正在扫描 MRI/CT、解析 EDF 通道、医生表与并口触发码…');
    try {
      const data = await api(`/api/subjects/${state.curId}/intake`, { method: 'POST' });
      state.intakeData = data;
      renderIntake(data);
      populateSeriesSelectors(data.dicom);
      renderBadChannelSelector(data.edf);
      await loadSubjects();
      setStatus('Step 1 体检与触发对齐完成');
    } catch (e) { alert(e.message); }
    finally { setBusy(false); }
  };

  $('#btnRunBehavior').onclick = async () => {
    if (!requireSubject()) return;
    setBusy(true, '正在运行 Judgement Bias 行为分析…');
    try {
      const data = await api(`/api/subjects/${state.curId}/behavior`, { method: 'POST' });
      renderBehavior(data);
      await loadSubjects();
      setStatus('Step 3 行为分析完成');
    } catch (e) { alert(e.message); }
    finally { setBusy(false); }
  };

  $('#btnRunLocalization').onclick = async () => {
    if (!state.step4Available) return;
    if (!requireSubject()) return;
    setBusy(true, '正在执行术后 CT 电极检测、CT→MR 配准与 MNI 脑区标注（约 15-30 秒）…');
    try {
      const data = await api(`/api/subjects/${state.curId}/localization`, {
        method: 'POST',
        body: JSON.stringify({
          mr_uid: $('#selMrUid').value || null,
          ct_uid: $('#selCtUid').value || null,
          hu_threshold: Number($('#inpHu').value),
          contact_spacing_mm: Number($('#inpSpacing').value),
        }),
      });
      state.locData = data;
      renderLocalization(data);
      await loadSubjects();
      if (data.mr_uid) {
        $('#selVol3dSeries').value = data.mr_uid;
        await openSeriesIn3D(data.mr_uid);
      }
      setStatus('Step 4 电极定位与 3D 脑配准完成');
    } catch (e) { alert(e.message); }
    finally { setBusy(false); }
  };

  $('#btnRunPreprocess').onclick = async () => {
    if (!requireSubject()) return;
    setBusy(true, '正在执行 PNAS 2025 标准化 SEEG 预处理…');
    try {
      const data = await api(`/api/subjects/${state.curId}/preprocess`, {
        method: 'POST',
        body: JSON.stringify({
          reref: $('#preReref').value,
          resample_hz: Number($('#preSfreq').value),
          epoch_lock: $('#preLock').value,
          baseline_mode: $('#preBaseline').value,
          tmin: Number($('#preTmin').value),
          tmax: Number($('#preTmax').value),
          iqr_weight: Number($('#preIqrW').value),
          phase_filter: $('#prePhase').value,
          bad_channels: Array.from(state.selectedBadChs),
          use_auto_bad: false,
        }),
      });
      renderPreprocess(data);
      await loadSubjects();
      setStatus('Step 5 SEEG 预处理完成');
    } catch (e) { alert(e.message); }
    finally { setBusy(false); }
  };

  $('#btnRefreshPlugins').onclick = () => loadPlugins();
  $('#btnGroupExport').onclick = async () => {
    setBusy(true, '正在合并导出全课题汇总 CSV…');
    try {
      const res = await api('/api/group/export', { method: 'POST' });
      alert(`导出成功！\n行为汇总被试数: ${res.n_behavior_subjects}\n电极触点总数: ${res.n_electrode_contacts}\n保存位置: judgement-bias-studio/workspace_data/group_exports/`);
    } catch (e) { alert(e.message); }
    finally { setBusy(false); }
  };

  // AI Drawer & Modal
  $('#btnToggleAi').onclick = () => $('#aiDrawer').classList.toggle('hidden');
  $('#btnCloseAi').onclick = () => $('#aiDrawer').classList.add('hidden');
  $('#btnSendAi').onclick = () => {
    const q = $('#aiInput').value;
    $('#aiInput').value = '';
    sendAiQuestion(q);
  };
  $$('.ai-qbtn').forEach((b) => { b.onclick = () => sendAiQuestion(b.dataset.q); });
  $$('.btn-ai-step').forEach((b) => {
    b.onclick = () => sendAiQuestion(`请帮我专业解读当前 ${b.dataset.step} 步骤的结果并给出质控结论和建议。`, b.dataset.step);
  });

  const openAiModal = (v) => $('#aiModalBackdrop').classList.toggle('hidden', !v);
  $('#btnAiSettings').onclick = async () => { await loadAiConfig(); openAiModal(true); };
  $('#btnCloseAiModal').onclick = () => openAiModal(false);
  $('#btnCancelAiModal').onclick = () => openAiModal(false);
  $('#btnSaveAiConfig').onclick = async () => {
    try {
      await api('/api/ai/config', {
        method: 'POST',
        body: JSON.stringify({
          base_url: $('#aiBaseUrl').value.trim(),
          model: $('#aiModel').value.trim(),
          api_key: $('#aiApiKey').value.trim(),
        }),
      });
      $('#aiApiKey').value = '';
      await loadAiConfig();
      openAiModal(false);
      setStatus('AI 配置已保存');
    } catch (e) { alert(e.message); }
  };

  // New Subject Modal
  const openModal = (v) => $('#modalBackdrop').classList.toggle('hidden', !v);
  $('#btnNewSubject').onclick = () => openModal(true);
  $('#btnCloseModal').onclick = () => openModal(false);
  $('#btnCancelModal').onclick = () => openModal(false);
  $('#btnDiscoverFolder').onclick = async () => {
    const folder = $('#mFolder').value.trim();
    if (!folder) return;
    try {
      const p = await api('/api/subjects/discover', {
        method: 'POST',
        body: JSON.stringify({ folder_path: folder }),
      });
      $('#mMri').value = p.mri_dir || '';
      $('#mCt').value = p.ct_dir || '';
      $('#mEdf').value = p.edf_path || '';
      $('#mBeh').value = p.behavior_json || '';
      $('#mExcel').value = p.electrode_excel || '';
      if (!$('#mSubId').value) {
        const base = folder.split('/').filter(Boolean).pop() || 'sub-new';
        $('#mSubId').value = base;
        $('#mName').value = base;
      }
    } catch (e) { alert(e.message); }
  };
  $('#btnSaveSubject').onclick = async () => {
    const sid = $('#mSubId').value.trim();
    if (!sid) return alert('请填写被试编号');
    try {
      await api('/api/subjects', {
        method: 'POST',
        body: JSON.stringify({
          subject_id: sid,
          display_name: $('#mName').value.trim() || sid,
          folder_path: $('#mFolder').value.trim(),
          mri_dir: $('#mMri').value.trim(),
          ct_dir: $('#mCt').value.trim(),
          edf_path: $('#mEdf').value.trim(),
          behavior_json: $('#mBeh').value.trim(),
          electrode_excel: $('#mExcel').value.trim(),
        }),
      });
      openModal(false);
      await loadSubjects();
      await selectSubject(sid);
    } catch (e) { alert(e.message); }
  };

  // Fault-tolerant Boot: subjects ALWAYS load first even if optional endpoints fail
  (async () => {
    try { await loadHealth(); } catch (e) { console.warn('loadHealth failed:', e); }
    try { await loadSubjects(); } catch (e) { console.error('loadSubjects failed:', e); }
    try { await loadPlugins(); } catch (e) { console.warn('loadPlugins failed:', e); }
    try { await loadAiConfig(); } catch (e) { console.warn('loadAiConfig failed:', e); }
  })();
})();
