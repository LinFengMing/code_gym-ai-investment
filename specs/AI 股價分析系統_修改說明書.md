# AI 股價分析系統修改說明書

> **用途說明**: 本說明書用於對現有系統進行客製化修改。請將現有程式碼、原始說明書和本客製化需求一起提交給AI，AI將根據您的需求對系統進行升級改造。

## 📋 基本資訊

### 原始系統
- **系統名稱**: AI 股票趨勢分析系統
- **修改目標**: 系統名稱修改為 AI 股票趨勢分析系統 (美股與台股)

### 修改類型
請說明本次修改的主要類型：
- 新增功能
- 增強現有功能
- 改善界面設計

## 🎯 客製化需求描述

### 需求 1：股票代碼改為輸入數字是台股，輸入英文是美股
**要達成什麼**：
股票代碼改為輸入數字時切換輸入 FindMind API Key，輸入英文時切換輸入 FMP API Key

**數據來源與說明**:
- FindMind API: 獲取股票歷史價格數據，包含開盤、最高、最低、收盤價和成交量
- API網址範例: `https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={symbol}&start_date={from}&end_date={to}`
- Headers Authorization 是 Bearer token {api_key}（若 API Key 為空則不加入此 Header）

### 需求 2：API Key 可選填
**要達成什麼**：
- FindMind API Key 如果為空，就不使用 Headers Authorization（允許使用免費額度）
- AI API Key 如果為空，就不執行 AI 分析，改為顯示提示訊息

**實作說明**:
- 移除 API Key 的必填驗證
- 在發送 FindMind API 請求時，只有在 API Key 不為空時才加入 Authorization Header
- 在執行 AI 分析前檢查 AI API Key 是否存在
- 若無 AI API Key，顯示友善提示訊息，但不影響其他功能正常運作

### 需求 3：支援 Google Gemini AI
**要達成什麼**：
- 新增 Google Gemini API 支援
- 使用者可選擇使用 OpenAI 或 Google Gemini 進行 AI 分析
- 使用 **gemini-2.5-pro** 高階推理模型

**數據來源與說明**:
- Google Gemini API: 使用 google-generativeai 套件
- 模型：gemini-2.5-pro（Gemini 2.5 系列的 Pro 版本，推理能力強，適合深度技術分析）
- API Key 取得：https://aistudio.google.com/app/apikey

**實作說明**:
- 在側邊欄新增「選擇 AI 提供商」下拉選單
- 根據選擇動態顯示對應的 API Key 輸入框
- 修改 `generate_ai_insights()` 函數支援雙 AI 提供商
- 統一使用相同的提示語，確保分析品質一致
- 新增 `google-generativeai` 套件依賴
- 優化 Gemini 回應處理，增加輸出 token 限制至 8000
- 只傳送最近 30 天數據以減少輸入 token

### 需求 4：預設使用 OpenAI
**要達成什麼**：
- 系統預設 AI 提供商為 **OpenAI**
- 調整選項順序，OpenAI 在前，Google Gemini 在後
- 保留 Gemini 選項供使用者切換

**實作原因**:
- OpenAI 穩定可靠，回應品質一致
- GPT-4o-mini 適合日常技術分析
- 商業級穩定性，適合長期使用
- 使用者可根據需求切換至 Gemini

**實作說明**:
- AI 提供商選項順序為 `["openai", "gemini"]`
- 設定 `index=0` 確保預設選擇 OpenAI
- 保留 Gemini 選項供使用者切換

### 特殊要求
**其他技術要求**：
- 如需新增套件，請告知新增套件名稱
支援自訂參數、要有錯誤處理

## 🚀 AI實作指令
1. **保持原有功能**：確保所有現有功能正常運作
2. **實現新需求**：根據上述需求描述實現新功能
3. **整合界面**：將新功能整合到現有界面中
4. **更新AI分析**：根據需求更新AI分析內容
5. **添加註釋**：為新增代碼添加清楚的中文註釋
6. **錯誤處理**：為新功能實施適當的錯誤處理