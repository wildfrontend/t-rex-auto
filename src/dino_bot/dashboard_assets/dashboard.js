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
    return `<article class="instance-card${running ? " running" : ""}${selected ? " selected" : ""}" data-instance-select="${escapeHtml(item.id)}">
      <div class="instance-card-head">
        <strong>${escapeHtml(item.name)}</strong>
        <span class="instance-state${running ? " running" : ""}">${running ? "● 執行中" : "○ 已停止"}</span>
      </div>
      <div class="instance-card-meta"><span>${escapeHtml(serial)}</span><span>Port ${item.status_port}</span></div>
      <div class="instance-card-meta"><span>${escapeHtml(active.mode_label || "未啟動")}</span><span>${escapeHtml((active.status || {}).current_stage || "—")}</span></div>
      <div class="instance-card-actions">
        <button type="button" data-instance-action="select" data-instance-id="${escapeHtml(item.id)}">檢視</button>
        <button type="button" data-instance-action="start-hatch-beginner" data-instance-id="${escapeHtml(item.id)}">新手孵蛋</button>
        <button type="button" data-instance-action="start-hatch-hunt" data-instance-id="${escapeHtml(item.id)}">孵蛋＋狩獵</button>
        <button type="button" data-instance-action="start-hunt" data-instance-id="${escapeHtml(item.id)}">純狩獵</button>
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
  if (event.kind === "cull_decision") return `${details.capacity}/350 · ${details.cull ? "執行淘汰" : "安全跳過"}`;
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
  renderActive(data.active || {});
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
  $("boostEnabled").checked = boostEnabled;
  $("boostEnabledLabel").textContent = boostEnabled
    ? "下一輪孵化使用：開啟"
    : "下一輪孵化使用：關閉";
  if (document.activeElement !== $("boostStockInput")) {
    $("boostStockInput").value = stock;
  }
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
  $("topTotal").textContent = counters.autoplace_top?.total || 0;
  $("massTotal").textContent = counters.autoplace_mass?.total || 0;
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

$("boostStockUpdate").addEventListener("click", async () => {
  const remaining = Number($("boostStockInput").value);
  const result = $("commandResult");
  if (!Number.isInteger(remaining) || remaining < 0 || remaining > 100) {
    result.classList.add("error");
    result.textContent = "Bot 額度必須是 0 到 100 的整數。";
    return;
  }
  $("boostStockUpdate").disabled = true;
  result.classList.remove("error");
  result.textContent = "更新本機庫存…";
  try {
    const suffix = selectedInstanceId ? `?instance=${encodeURIComponent(selectedInstanceId)}` : "";
    const response = await fetch(`/api/control/set-boost-stock${suffix}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Dino-Dashboard": "1",
      },
      body: JSON.stringify({ remaining }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    result.textContent = payload.message;
    await refresh();
  } catch (error) {
    result.classList.add("error");
    result.textContent = String(error.message || error);
  } finally {
    $("boostStockUpdate").disabled = false;
  }
});

$("boostEnabled").addEventListener("change", async () => {
  const checkbox = $("boostEnabled");
  const result = $("commandResult");
  checkbox.disabled = true;
  result.classList.remove("error");
  result.textContent = checkbox.checked
    ? "設定下一輪孵化使用加速券…"
    : "關閉加速券使用…";
  try {
    const suffix = selectedInstanceId ? `?instance=${encodeURIComponent(selectedInstanceId)}` : "";
    const response = await fetch(`/api/control/set-boost-enabled${suffix}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Dino-Dashboard": "1",
      },
      body: JSON.stringify({ enabled: checkbox.checked }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "HTTP " + response.status);
    result.textContent = payload.message;
    await refresh();
  } catch (error) {
    checkbox.checked = !checkbox.checked;
    result.classList.add("error");
    result.textContent = String(error.message || error);
  } finally {
    checkbox.disabled = false;
  }
});

refresh();
window.setInterval(refresh, 2500);
