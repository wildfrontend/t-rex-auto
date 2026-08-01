# 孵蛋（Auto Hatch）素材

完整孵蛋流程使用下列 template（`type` 名稱必須完全一致，
對應 `src/dino_bot/hatch.py` 的常數）。截圖用
`dino-bot snapshot`，裁切用 `dino-bot template --type <type> --name <name>`，
並把 `--config` 指到含 `hatch.manifest` 的設定（manifest 預設
`assets/hatch/manifest.json`）。

| type | 來源畫面 | 說明 |
| --- | --- | --- |
| `hatch_home_anchor` | 主頁面 | 判斷「在主頁面」的錨點；建議裁左上選單第 4 個蛋巢圖示（草巢白蛋，紅點會變，裁不含紅點的區域） |
| `hatch_incubator_title` | 孵化器 | 「孵化器」標題文字 |
| `hatch_label` | 孵化器 | 蛋下方的「孵化」字樣（點擊目標；黃/藍光圈不影響） |
| `hatch_button` | 蛋詳細頁 | 青色「孵化」按鈕 |
| `hatch_claim_button` | 孵化結果頁 | 青色「獲取」按鈕 |
| `hatch_expel_button` | 孵化結果頁 | 紅色「驅逐」按鈕——**只用於防呆辨識，絕不點擊** |
| `hatch_close_button` | 孵化器 | 紅色 X 關閉鈕 |

注意：

- 大蛋堆平台**不做 template**——樣式會變，用 `hatch.egg_pile` 座標
  （900 寬參考座標系）點擊，進場後以 `hatch_incubator_title` 驗證。
- 「獲取」「驅逐」與各彈窗的「是」「否」按鈕配色相近，threshold
  建議 ≥0.9，並先用 `--mode debug` 驗證不互相誤匹配。
- Phase B（蛋巢管理）與 Phase C（清除多餘恐龍）模板已收在同一份
  `manifest.json`。所有按鈕模板都只在對應畫面標題／提示同時可見時使用；
  洞穴容量讀值必須通過 `/350` 分母驗證才會觸發清理。
