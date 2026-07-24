# Android USB 實機實驗版

此版本只存在於 `feature/android-usb-device` branch 與獨立 worktree，不會修改正式版
`main` 或正式執行目錄 `.runtime-macos`。

## 使用條件

1. Android 手機開啟「開發人員選項 → USB 偵錯」。
2. 使用 USB 連接 Mac，並在手機上允許這台電腦。
3. 手機保持解鎖，使用者自行開啟 Dino Mutant 並停在主地圖或採集地圖。
4. Finder 雙擊 `start-android-usb.command`。

實驗版使用：

- 獨立執行目錄：`.runtime-android-usb`
- 獨立本機狀態埠：`8865`
- 預設速度：`safe`
- 裝置策略：只接受唯一 USB Android 實機，不會回退到模擬器

停止時雙擊 `stop-android-usb.command`。

## 完整移除

先停止實驗版，再從正式版 repository 執行：

```bash
git worktree remove /Users/macintosh/Dev/Personal/t-rex-auto-android-usb
git branch -D feature/android-usb-device
```

這不會影響正式版的來源、commit 或 `.runtime-macos`。
