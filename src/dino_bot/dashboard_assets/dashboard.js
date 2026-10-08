const $ = (id) => document.getElementById(id);
const eventNames = {
  hunt: "成功狩獵",
  hatch: "孵化並取得恐龍",
  nest_collect_batch: "收集所有巢蛋",
  replacement: "親代替換完成",
  autoplace_top: "頂尖自動放置完成",
  autoplace_mass: "量產自動放置完成",
  cull_removed: "洞穴淘汰完成",
  verification_failure: "操作驗證失敗",
  game_restart: "遊戲已重新啟動",
  cull_decision: "洞穴容量判定",
};
const customWorkflowStageLabels = {
  attack: "攻擊親代",
  hp: "HP 親代",
  top: "頂尖配置",
  mass: "量產配置",
  collect: "收集巢蛋",
  cave: "洞穴淘汰",
  hatch: "孵蛋",
  hunt: "狩獵",
};

function setCounter(name, value) {
  const data = value || { session: 0, today: 0, total: 0 };
  $(name + "Session").textContent = data.session;
  $(name + "Today").textContent = data.today;
  $(name + "Total").textContent = data.total;
}

function setRecord(name, record) {
  const key = name[0].toUpperCase() + name.slice(1);
  $("record" + key).textContent = record ? record.value.toLocaleString() : "—";
  $("record" + key + "Meta").textContent = record
    ? `${record.tag || "已讀取"} · ${record.occurred_at.replace("T", " ")}`
    : "尚無資料";
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function formatLongDuration(seconds) {
  const value = Math.max(0, Math.ceil(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = value % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;",
  }[character]));
}

let selectedInstanceId = null;
let knownInstances = [];
let lastOperationUpdatedAt = null;

function renderInstances(items) {
  knownInstances = items || [];
  const root = $("instanceList");
  if (!items || !items.length) {
    root.innerHTML = '<p class="empty">尚未設定 Bot 實例</p>';
    return;
  }
  root.innerHTML = items.map((item) => {
    const active = item.active || {};
    const running = Boolean(active.running);
    const selected = item.id === selectedInstanceId;
    const serial = item.serial || "未設定 ADB";
    const stopTimer = item.stop_timer || {};
    const allowed = new Set(item.allowed_modes || []);
    const startButton = (mode, label) => allowed.has(mode)
      ? `<button type="button" data-instance-action="start-${mode}" data-instance-id="${escapeHtml(item.id)}">${label}</button>`
      : "";
    return `<article class="instance-card${running ? " running" : ""}${selected ? " selected" : ""}" data-instance-select="${escapeHtml(item.id)}">
      <div class="instance-card-head">
        <strong>${escapeHtml(item.name)}</strong>
        <span class="instance-state${running ? " running" : ""}">${running ? "● 執行中" : "○ 已停止"}</span>
      </div>
      <div class="instance-card-meta"><span>${escapeHtml(serial)}</span><span>Port ${item.status_port}</span></div>
      <div class="instance-card-meta"><span>${escapeHtml(active.mode_label || "未啟動")}</span><span>${escapeHtml((active.status || {}).current_stage || "—")}</span>${stopTimer.active ? `<span>停止倒數 ${formatLongDuration(stopTimer.remaining_seconds)}</span>` : ""}</div>
      <div class="instance-card-actions">
        <button type="button" data-instance-action="select" data-instance-id="${escapeHtml(item.id)}">檢視</button>
        ${startButton("hatch-hunt", "孵蛋＋狩獵")}
        ${startButton("custom-workflow", "自訂循環")}
        ${startButton("hunt", "純狩獵")}
        <button type="button" data-instance-action="edit" data-instance-id="${escapeHtml(item.id)}">設定</button>
        <button type="button" class="danger" data-instance-action="stop" data-instance-id="${escapeHtml(item.id)}">停止</button>
      </div>
    </article>`;
  }).join("");
}

function renderActive(active) {
  $("modeLabel").textContent = active.mode_label || "未啟動";
  $("liveDot").classList.toggle("running", Boolean(active.running));
  const workflow = active.workflow || {};
  $("workflowLabel").textContent = active.running ? workflow.label : "Bot 尚未啟動";
  const status = active.status || {};
  $("workflowMeta").textContent = active.running
    ? `狀態：${status.current_stage || "active"} · 操作 ${status.total_actions || 0} · Port ${active.port}`
    : "可以從右側控制區啟動純狩獵或自動孵蛋＋狩獵";
  document.querySelectorAll(".stage-track span").forEach((node) => {
    node.classList.toggle("active", node.dataset.stage === workflow.stage);
  });
  const remaining = workflow.cooldown_remaining_seconds;
  $("cooldown").classList.toggle("hidden", remaining === null || remaining === undefined || remaining <= 0);
  $("cooldownValue").textContent = formatDuration(remaining);
}

function renderTimeline(items) {
  const root = $("timeline");
  if (!items || !items.length) {
    root.innerHTML = '<p class="empty">尚無歷史資料</p>';
    return;
  }
  const max = Math.max(1, ...items.flatMap((item) => [item.hunt, item.hatch]));
  root.innerHTML = `<div class="hour-chart">${items.map((item) => {
    const huntHeight = item.hunt ? Math.max(3, Math.round(item.hunt / max * 105)) : 0;
    const hatchHeight = item.hatch ? Math.max(3, Math.round(item.hatch / max * 105)) : 0;
    const showLabel = item.hour % 3 === 0 || item.hour === 23;
    return `<div class="hour-bars">
      <i class="bar hunt${item.hunt ? "" : " zero"}" style="height:${huntHeight}px" data-value="${item.hunt}"></i>
      <i class="bar hatch${item.hatch ? "" : " zero"}" style="height:${hatchHeight}px" data-value="${item.hatch}"></i>
      <span class="hour-label">${showLabel ? String(item.hour).padStart(2, "0") : ""}</span>
    </div>`;
  }).join("")}</div>`;
}

function eventDetails(event) {
  const details = event.details || {};
  if (event.kind === "replacement") return `${details.tag || "親代"} · ${details.hp}/${details.attack}/${details.speed}`;
  if (event.kind === "cull_decision") return `${details.capacity}/${details.capacity_limit || 350} · ${details.cull ? "執行淘汰" : "安全跳過"}`;
  if (event.kind === "cull_removed") return `${details.before} → ${details.expected_after} · 選取 ${details.selected}`;
  if (event.kind === "verification_failure") return details.target || "未知目標";
  return "已驗證";
}

function renderEvents(items) {
  const root = $("events");
  if (!items || !items.length) {
    root.innerHTML = '<p class="empty">等待事件</p>';
    return;
  }
  root.innerHTML = items.map((item) => `<div class="event-row">
    <time>${item.occurred_at.slice(11)}</time>
    <strong>${eventNames[item.kind] || item.kind}</strong>
    <span>${eventDetails(item)}</span>
  </div>`).join("");
}

function render(data) {
  if (!selectedInstanceId || !data.instances?.some((item) => item.id === selectedInstanceId)) {
    selectedInstanceId = data.selected_instance || data.instances?.[0]?.id || null;
  }
  renderInstances(data.instances || []);
  const selected = (data.instances || []).find((item) => item.id === selectedInstanceId);
  const allowed = new Set(selected?.allowed_modes || []);
  document.querySelectorAll("button[data-mode]").forEach((button) => {
    button.hidden = !allowed.has(button.dataset.mode);
  });
  renderActive(data.active || {});
  const stopTimer = data.stop_timer || {};
  const timerControl = document.querySelector(".stop-timer-control");
  timerControl.classList.toggle("active", stopTimer.active === true);
  $("stopTimerStatus").textContent = stopTimer.active
    ? `剩餘 ${formatLongDuration(stopTimer.remaining_seconds)}`
    : "未設定";
  $("stopTimerDeadline").textContent = stopTimer.active
    ? `預計 ${new Date(stopTimer.deadline_at).toLocaleString("zh-TW", { hour12: false })} 關閉遊戲與 Bot`
    : "指定這台 Bot 要執行多久；到期先關遊戲，再停止 Bot";
  const operation = data.operation;
  if (operation?.updated_at && operation.updated_at !== lastOperationUpdatedAt) {
    lastOperationUpdatedAt = operation.updated_at;
    const result = $("commandResult");
    result.classList.toggle("error", operation.state === "failed");
    const prefix = operation.state === "pending"
      ? "處理中"
      : operation.state === "succeeded" ? "完成" : "失敗";
    result.textContent = `${prefix} [${operation.code}]：${operation.message}`;
  }
  const inventory = data.hatch_boost_inventory || {};
  const stock = Number(inventory.remaining ?? 100);
  $("boostStockValue").textContent = stock;
  $("boostUsedTotal").textContent = Number(inventory.used_total || 0);
  const boostEnabled = inventory.enabled === true;
  const boostWait = Math.max(0, Math.ceil(Number(inventory.next_check_at || 0) - Date.now() / 1000));
  $("boostSchedule").textContent = !boostEnabled
    ? "已關閉自動使用"
    : stock <= 0 ? "等待補充庫存"
    : !data.active?.running ? "等待 Bot 啟動"
    : !["hatch-hunt", "custom-workflow", "hatch-full"].includes(data.active.feature)
      ? "適用於自動孵蛋與自訂循環模式"
    : boostWait > 0
      ? `最早可再檢查：${Math.floor(boostWait / 60)} 分 ${boostWait % 60} 秒${inventory.waiting_reason ? `（${inventory.waiting_reason}）` : ""}`
      : "等待正常開啟孵化器時檢查（不中斷狩獵）";
  const tuning = data.hatch_tuning || {};
  $("capacityLimitValue").textContent = tuning.capacity_limit ?? "—";
  $("cullThresholdValue").textContent = tuning.cull_threshold ?? "—";
  $("screeningIntervalValue").textContent = tuning.screening_growth_interval ?? "—";
  const customWorkflow = data.custom_workflow || {};
  const selectedStages = new Set(customWorkflow.stages || ["collect", "hatch", "hunt"]);
  const orderedStages = Object.keys(customWorkflowStageLabels)
    .filter((stage) => selectedStages.has(stage));
  $("customWorkflowSummary").textContent = orderedStages
    .map((stage) => customWorkflowStageLabels[stage])
    .join(" → ");
  syncSettingsForm(data);
  const metrics = data.metrics || {};
  const counters = metrics.counters || {};
  setCounter("hunt", counters.hunt);
  setCounter("hatch", counters.hatch);
  setCounter("replacement", counters.replacement);
  setCounter("collect", counters.nest_collect_batch);
  setCounter("cull", counters.cull_removed);
  setRecord("hp", metrics.records?.hp);
  setRecord("attack", metrics.records?.attack);
  setRecord("speed", metrics.records?.speed);
  $("failureToday").textContent = counters.verification_failure?.today || 0;
  $("trendScope").textContent = metrics.timeline_date
    ? `${metrics.timeline_date} · 每小時`
    : "今天 · 每小時";
  renderTimeline(metrics.timeline);
  renderEvents(metrics.recent_events);
  $("lastUpdate").textContent = `更新 ${new Date().toLocaleTimeString("zh-TW", { hour12: false })}`;
}

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const query = selectedInstanceId ? `?instance=${encodeURIComponent(selectedInstanceId)}` : "";
    const response = await fetch(`/api/overview${query}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("modeLabel").textContent = "儀表板離線";
    $("lastUpdate").textContent = String(error);
    $("liveDot").classList.remove("running");
  } finally {
    refreshing = false;
  }
}

async function invokeControl(button) {
  const action = button.dataset.action;
  const instanceId = button.dataset.instanceId || selectedInstanceId;
  if (button.dataset.confirm && !window.confirm(button.dataset.confirm)) return;
  document.querySelectorAll("button[data-action]").forEach((item) => { item.disabled = true; });
  const result = $("commandResult");
  result.classList.remove("error");
  result.textContent = "指令執行中…";
  try {
    const suffix = instanceId ? `?instance=${encodeURIComponent(instanceId)}` : "";
    const response = await fetch(`/api/control/${action}${suffix}`, {
      method: "POST",
      headers: { "X-Dino-Dashboard": "1" },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    result.textContent = payload.message || `指令已接受：${action}`;
    window.setTimeout(refresh, 1200);
  } catch (error) {
    result.classList.add("error");
    result.textContent = String(error.message || error);
  } finally {
    document.querySelectorAll("button[data-action]").forEach((item) => { item.disabled = false; });
  }
}

document.querySelectorAll("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => invokeControl(button));
});

$("stopTimerSet").addEventListener("click", async () => {
  const hours = Number($("stopTimerHours").value);
  const minutes = Number($("stopTimerMinutes").value);
  const durationSeconds = (hours * 60 + minutes) * 60;
  const result = $("commandResult");
  if (
    !Number.isInteger(hours) || !Number.isInteger(minutes) ||
    hours < 0 || hours > 168 || minutes < 0 || minutes > 59 ||
    durationSeconds < 60 || durationSeconds > 7 * 24 * 60 * 60
  ) {
    result.classList.add("error");
    result.textContent = "請設定 1 分鐘到 7 天之間的執行時間。";
    return;
  }
  const button = $("stopTimerSet");
  button.disabled = true;
  result.classList.remove("error");
  result.textContent = "設定停止計時器…";
  try {
    const suffix = selectedInstanceId ? `?instance=${encodeURIComponent(selectedInstanceId)}` : "";
    const response = await fetch(`/api/control/set-stop-timer${suffix}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Dino-Dashboard": "1",
      },
      body: JSON.stringify({ duration_seconds: durationSeconds }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    result.textContent = payload.message;
    await refresh();
  } catch (error) {
    result.classList.add("error");
    result.textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
});

function openNewInstanceForm() {
  const form = $("instanceForm");
  delete form.dataset.editingId;
  form.reset();
  $("instancePort").value = String(8775 + knownInstances.length - 1);
  form.querySelector("button.primary").textContent = "建立實例";
  form.classList.remove("hidden");
}

function openEditInstanceForm(instanceId) {
  const item = knownInstances.find((candidate) => candidate.id === instanceId);
  if (!item) return;
  if (instanceId !== selectedInstanceId && !confirmDiscardSettings()) return;
  const form = $("instanceForm");
  form.dataset.editingId = instanceId;
  $("instanceName").value = item.name || "";
  $("instanceSerial").value = item.serial || "";
  $("instancePort").value = item.status_port || "";
  form.querySelector("button.primary").textContent = "儲存並重新啟動";
  form.classList.remove("hidden");
  selectedInstanceId = instanceId;
  renderInstances(knownInstances);
}

function renderAdbScan(payload) {
  const box = $("adbScanResult");
  const devices = payload.devices || [];
  if (!devices.length) {
    box.innerHTML = `<p class="muted">${escapeHtml(payload.message)}</p>`;
    box.classList.remove("hidden");
    return;
  }
  // 一台就直接填入;多台一定要使用者自己挑,替他選會連錯模擬器。
  const ready = devices.filter((device) => device.ready);
  if (ready.length === 1) $("instanceSerial").value = ready[0].serial;
  const rows = devices
    .map((device) => {
      const notes = [device.hint, device.state === "device" ? "" : device.state]
        .filter(Boolean)
        .join(" · ");
      return `<button type="button" class="scan-pick" data-serial="${escapeHtml(device.serial)}"
        ${device.ready ? "" : "disabled"}>
        <span>${escapeHtml(device.serial)}</span>
        <span class="muted">${escapeHtml(notes)}</span>
      </button>`;
    })
    .join("");
  box.innerHTML = `<p class="muted">${escapeHtml(payload.message)}</p>${rows}`;
  box.classList.remove("hidden");
}

$("scanAdb").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "掃描中…";
  try {
    const editingId = $("instanceForm").dataset.editingId || selectedInstanceId;
    const suffix = editingId ? `?instance=${encodeURIComponent(editingId)}` : "";
    const response = await fetch(`/api/control/scan-adb${suffix}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Dino-Dashboard": "1" },
      body: "{}",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    renderAdbScan(payload);
  } catch (error) {
    const box = $("adbScanResult");
    box.innerHTML = `<p class="muted">掃描失敗：${escapeHtml(String(error.message || error))}</p>`;
    box.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
});

$("adbScanResult").addEventListener("click", (event) => {
  const pick = event.target.closest(".scan-pick");
  if (!pick) return;
  $("instanceSerial").value = pick.dataset.serial;
});

function closeInstanceForm() {
  const form = $("instanceForm");
  delete form.dataset.editingId;
  const box = $("adbScanResult");
  box.classList.add("hidden");
  box.innerHTML = "";
  form.reset();
  form.querySelector("button.primary").textContent = "建立實例";
  form.classList.add("hidden");
}

$("instanceList").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-instance-action]");
  const card = event.target.closest("[data-instance-select]");
  const instanceId = button?.dataset.instanceId || card?.dataset.instanceSelect;
  if (!instanceId) return;
  if (button?.dataset.instanceAction === "edit") {
    openEditInstanceForm(instanceId);
    return;
  }
  if (!button || button.dataset.instanceAction === "select") {
    if (instanceId !== selectedInstanceId && !confirmDiscardSettings()) return;
    selectedInstanceId = instanceId;
    refresh();
    return;
  }
  invokeControl({
    dataset: { action: button.dataset.instanceAction, instanceId },
    disabled: false,
  });
});

$("addInstanceToggle").addEventListener("click", () => {
  if ($("instanceForm").classList.contains("hidden")) openNewInstanceForm();
  else closeInstanceForm();
});
$("addInstanceCancel").addEventListener("click", () => {
  closeInstanceForm();
});
$("instanceForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button.primary");
  button.disabled = true;
  const editingId = form.dataset.editingId;
  const action = editingId ? "update-instance" : "add-instance";
  const suffix = editingId ? `?instance=${encodeURIComponent(editingId)}` : "";
  try {
    const response = await fetch(`/api/control/${action}${suffix}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Dino-Dashboard": "1" },
      body: JSON.stringify({
        id: editingId,
        name: $("instanceName").value,
        serial: $("instanceSerial").value,
        status_port: Number($("instancePort").value),
        restart: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    selectedInstanceId = payload.instance_id;
    closeInstanceForm();
    $("commandResult").classList.remove("error");
    $("commandResult").textContent = payload.message || `已儲存實例：${payload.name}`;
    await refresh();
  } catch (error) {
    $("commandResult").classList.add("error");
    $("commandResult").textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
});

// 運行設定表單:輪詢只在沒有未儲存的修改時回填,改好後一次送出。原本每個欄位
// 各自儲存,而每 2.5 秒的輪詢會把沒有焦點的欄位蓋回 Bot 的值——勾了第二個
// 選項,第一個就被還原。
let savedSettings = null;
let settingsInstanceId = null;
let settingsDirty = false;
let settingsSaving = false;
// 儲存結果或驗證錯誤要留到下一次修改,不能被下一輪輪詢的預設狀態文字蓋掉。
let settingsNotice = null;

const tuningInputs = {
  capacity_limit: "capacityLimitInput",
  cull_threshold: "cullThresholdInput",
  screening_growth_interval: "screeningIntervalInput",
};

function settingsFromServer(data) {
  const inventory = data.hatch_boost_inventory || {};
  const tuning = data.hatch_tuning || {};
  const stages = new Set(data.custom_workflow?.stages || ["collect", "hatch", "hunt"]);
  return {
    stages: Array.from(document.querySelectorAll("input[data-custom-stage]"))
      .filter((input) => input.disabled ? input.checked : stages.has(input.dataset.customStage))
      .map((input) => input.dataset.customStage),
    boostEnabled: inventory.enabled === true,
    boostRemaining: Number(inventory.remaining ?? 100),
    tuning: {
      capacity_limit: tuning.capacity_limit ?? null,
      cull_threshold: tuning.cull_threshold ?? null,
      screening_growth_interval: tuning.screening_growth_interval ?? null,
      allow_extreme_specialization_parent: tuning.allow_extreme_specialization_parent === true,
      auto_place_specializations: tuning.auto_place_specializations === true,
    },
  };
}

function settingsFromForm() {
  const number = (id) => ($(id).value === "" ? null : Number($(id).value));
  return {
    stages: Array.from(document.querySelectorAll("input[data-custom-stage]"))
      .filter((input) => input.checked)
      .map((input) => input.dataset.customStage),
    boostEnabled: $("boostEnabled").checked,
    boostRemaining: number("boostStockInput"),
    tuning: {
      capacity_limit: number(tuningInputs.capacity_limit),
      cull_threshold: number(tuningInputs.cull_threshold),
      screening_growth_interval: number(tuningInputs.screening_growth_interval),
      allow_extreme_specialization_parent: $("extremeParentProtection").checked,
      auto_place_specializations: $("specializationAutoPlace").checked,
    },
  };
}

function fillSettingsForm(settings) {
  const stages = new Set(settings.stages);
  document.querySelectorAll("input[data-custom-stage]").forEach((input) => {
    if (!input.disabled) input.checked = stages.has(input.dataset.customStage);
  });
  $("boostEnabled").checked = settings.boostEnabled;
  $("boostStockInput").value = settings.boostRemaining;
  for (const [key, id] of Object.entries(tuningInputs)) {
    $(id).value = settings.tuning[key] ?? "";
  }
  $("extremeParentProtection").checked = settings.tuning.allow_extreme_specialization_parent;
  $("specializationAutoPlace").checked = settings.tuning.auto_place_specializations;
}

function changedSettingsSections(form, saved) {
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  return {
    stages: !same(form.stages, saved.stages),
    boostEnabled: form.boostEnabled !== saved.boostEnabled,
    boostRemaining: form.boostRemaining !== saved.boostRemaining,
    tuning: !same(form.tuning, saved.tuning),
  };
}

function updateToggleLabels() {
  $("boostEnabledLabel").textContent = $("boostEnabled").checked
    ? "孵化時自動檢查加速：開啟"
    : "孵化時自動檢查加速：關閉";
  $("extremeParentProtectionLabel").textContent = $("extremeParentProtection").checked
    ? "10/1/1 極端親代保護：開啟"
    : "10/1/1 極端親代保護：關閉";
  $("specializationAutoPlaceLabel").textContent = $("specializationAutoPlace").checked
    ? "攻擊／HP 自動放置＋純化：開啟"
    : "攻擊／HP 自動放置＋純化：關閉";
}

function renderSettingsState(message, isError = false) {
  if (message !== undefined) settingsNotice = { message, isError };
  const count = savedSettings
    ? Object.values(changedSettingsSections(settingsFromForm(), savedSettings)).filter(Boolean).length
    : 0;
  settingsDirty = count > 0;
  $("settingsForm").classList.toggle("dirty", settingsDirty);
  $("settingsSave").disabled = settingsSaving || !settingsDirty;
  $("settingsDiscard").disabled = settingsSaving || !settingsDirty;
  const state = $("settingsState");
  state.classList.toggle("error", settingsNotice?.isError === true);
  state.textContent = settingsNotice?.message
    || (settingsDirty ? `有 ${count} 個區塊尚未儲存` : "設定與 Bot 一致");
  updateToggleLabels();
}

function syncSettingsForm(data) {
  const dataInstance = data.selected_instance || selectedInstanceId;
  if (dataInstance !== selectedInstanceId) return;
  savedSettings = settingsFromServer(data);
  if (settingsInstanceId !== selectedInstanceId) {
    settingsInstanceId = selectedInstanceId;
    settingsDirty = false;
    settingsNotice = null;
  }
  if (!settingsDirty && !settingsSaving) fillSettingsForm(savedSettings);
  if (!settingsSaving) renderSettingsState();
}

function confirmDiscardSettings() {
  if (!settingsDirty) return true;
  if (!window.confirm("這台的設定還沒儲存，切換會放棄這些修改。確定切換？")) return false;
  settingsDirty = false;
  return true;
}

function validateSettings(form) {
  const positiveInteger = (value) => Number.isInteger(value) && value > 0;
  if (!Number.isInteger(form.boostRemaining) || form.boostRemaining < 0 || form.boostRemaining > 100) {
    return "加速券庫存必須是 0 到 100 的整數。";
  }
  const { capacity_limit: limit, cull_threshold: threshold, screening_growth_interval: interval } = form.tuning;
  if (!positiveInteger(limit) || !positiveInteger(threshold) || !positiveInteger(interval)) {
    return "上限人口、安全人口與篩選間隔都必須是大於 0 的整數。";
  }
  if (threshold > limit) return "安全人口不能大於上限人口。";
  return null;
}

async function postSetting(action, body) {
  const suffix = selectedInstanceId ? `?instance=${encodeURIComponent(selectedInstanceId)}` : "";
  const response = await fetch(`/api/control/${action}${suffix}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Dino-Dashboard": "1" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function onSettingsEdited() {
  settingsNotice = null;
  renderSettingsState();
}

$("settingsForm").addEventListener("input", onSettingsEdited);
$("settingsForm").addEventListener("change", onSettingsEdited);

$("settingsDiscard").addEventListener("click", () => {
  if (savedSettings) fillSettingsForm(savedSettings);
  onSettingsEdited();
});

$("settingsForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!savedSettings || settingsSaving) return;
  const form = settingsFromForm();
  const problem = validateSettings(form);
  if (problem) {
    renderSettingsState(problem, true);
    return;
  }
  const changed = changedSettingsSections(form, savedSettings);
  const steps = [
    changed.stages && ["自訂流程", "set-custom-workflow", { stages: form.stages }],
    changed.boostEnabled && ["自動加速", "set-boost-enabled", { enabled: form.boostEnabled }],
    changed.boostRemaining && ["加速券庫存", "set-boost-stock", { remaining: form.boostRemaining }],
    changed.tuning && ["孵蛋參數", "set-hatch-tuning", form.tuning],
  ].filter(Boolean);
  if (!steps.length) return;
  settingsSaving = true;
  renderSettingsState("儲存中…");
  const done = [];
  let restartRequired = false;
  let failure = null;
  try {
    for (const [label, action, body] of steps) {
      try {
        const payload = await postSetting(action, body);
        restartRequired = restartRequired || payload.restart_required === true;
        done.push(label);
      } catch (error) {
        failure = `${label}儲存失敗：${error.message || error}`;
        break;
      }
    }
  } finally {
    settingsSaving = false;
  }
  const summary = done.length ? `已儲存：${done.join("、")}` : "";
  const message = failure
    ? [summary, failure, "其餘修改仍保留在表單上"].filter(Boolean).join("；")
    : `${summary}${restartRequired ? "；重新啟動 Bot 後生效" : ""}`;
  const result = $("commandResult");
  result.classList.toggle("error", Boolean(failure));
  result.textContent = message;
  if (!failure) settingsDirty = false;
  await refresh();
  renderSettingsState(failure ? message : summary, Boolean(failure));
});

window.addEventListener("beforeunload", (event) => {
  if (!settingsDirty) return;
  event.preventDefault();
  event.returnValue = "";
});

refresh();
window.setInterval(refresh, 2500);
