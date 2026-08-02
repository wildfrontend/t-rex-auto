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
  renderActive(data.active || {});
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
    const response = await fetch("/api/overview", { cache: "no-store" });
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
  if (button.dataset.confirm && !window.confirm(button.dataset.confirm)) return;
  document.querySelectorAll("button[data-action]").forEach((item) => { item.disabled = true; });
  const result = $("commandResult");
  result.classList.remove("error");
  result.textContent = "指令執行中…";
  try {
    const response = await fetch(`/api/control/${action}`, {
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
refresh();
window.setInterval(refresh, 2500);
