# AI 股票趨勢分析系統 - 修改日誌

## 版本 2.1.2 (2026-01-13)

### 🎯 預設設定調整

#### 預設 AI 提供商改回 OpenAI
- ✅ 系統預設 AI 提供商改回 **OpenAI (GPT-4o-mini)**
- ✅ 選項順序調整：OpenAI 在前，Gemini 在後
- ✅ 更新修改說明書和規格說明書

**變更原因**：
- OpenAI 穩定可靠，回應品質一致
- GPT-4o-mini 適合日常技術分析
- 商業級穩定性，適合長期使用
- 使用者仍可切換至 Gemini

**文檔更新**：
- ✅ `specs/AI 股價分析系統_修改說明書.md` - 更新需求 4
- ✅ `specs/AI 股價分析系統_規格說明書.md` - 更新預設設定說明
- ✅ `src/app.py` - 預設選項調整

---

## 版本 2.1.1 (2026-01-13)

### 🎯 優化改進

#### 預設 AI 提供商變更（已改回 OpenAI）
- ✅ 系統預設 AI 提供商改為 **Google Gemini**（已改回 OpenAI）
- ✅ 選項順序調整：Gemini 在前，OpenAI 在後（已改回）
- ✅ 強調 Gemini 為推薦選項（⭐標記）（已移除）
- ✅ 更新修改說明書，新增「需求 4：預設使用 Google Gemini」（已更新為 OpenAI）

**變更原因**：
- Google Gemini 免費額度更高
- gemini-2.5-pro 推理能力強
- 無需信用卡，新手友善
- 適合大多數使用場景

**文檔更新**：
- ✅ `specs/AI 股價分析系統_修改說明書.md` - 新增需求 4
- ✅ `src/app.py` - 預設選項和 UI 優化
- ✅ `README.md` - 功能說明和使用指南更新
- ✅ `CHANGELOG.md` - 版本記錄

---

## 版本 2.1.0 (2026-01-13)

### 🎉 新增功能

#### Google Gemini AI 支援
- ✅ 新增 Google Gemini API 整合
- ✅ 支援 **gemini-2.5-pro** 高階模型
- ✅ 使用者可選擇 AI 提供商：OpenAI 或 Google Gemini
- ✅ 動態切換 API Key 輸入框
- ✅ 統一的 AI 分析介面
- ✅ 優化提示語，減少輸入 token，只傳送最近30天數據

**功能說明**：
- 在側邊欄新增「選擇 AI 提供商」下拉選單
- 可選擇「OpenAI (GPT-4o-mini)」或「Google Gemini (gemini-2.5-pro)」
- 根據選擇自動顯示對應的 API Key 輸入框
- 兩種 AI 使用相同的提示語，確保分析品質一致
- gemini-2.5-pro 提供更強大的推理能力和完整的分析內容
- 輸出限制提升至 8000 tokens，確保分析完整性

### 🔧 技術改進
- 修改 `generate_ai_insights()` 函數支援雙 AI 提供商
- 新增 `google-generativeai` 套件依賴
- 統一錯誤處理機制
- 改善使用者提示訊息

### 📦 依賴更新
- 新增：`google-generativeai>=0.8.0`
- 版本號更新：0.1.0 → 0.2.0

---

## 版本 2.0.1 (2026-01-13)

### 🎉 新增功能

#### API Key 可選填支援
- ✅ FindMind API Key 可選填，為空時不加入 Authorization Header（可使用免費額度）
- ✅ OpenAI API Key 可選填，為空時跳過 AI 分析並顯示友善提示
- ✅ 移除強制 API Key 驗證，提升使用彈性
- ✅ 優化用戶體驗，即使沒有 OpenAI API Key 也能查看圖表和技術指標

### 🐛 問題修復

#### FindMind API 400 錯誤修復
- ✅ 修正 FindMind API 欄位名稱映射問題
- ✅ 更新欄位對應：`max` → `high`, `min` → `low`
- ✅ 新增詳細的錯誤處理和調試資訊
- ✅ 加入瀏覽器模擬的 User-Agent Header
- ✅ 改善 API 請求超時處理（30秒）

**問題說明**：
- FindMind API 實際回應的欄位名稱為小寫（`max`, `min`）而非大寫
- 原映射使用 `Max`, `Min` 導致欄位無法正確對應
- 已修正為正確的欄位名稱映射

### 🔧 技術改進
- 修改 `get_taiwan_stock_data()` 函數，只在 API Key 不為空時才加入 Authorization Header
- 修改主分析邏輯，移除 API Key 必填驗證
- 在 AI 分析區塊加入條件判斷，只在有 OpenAI API Key 時執行分析
- 新增詳細的 API 回應欄位調試資訊
- 改善錯誤訊息，提供更清楚的問題診斷資訊

---

## 版本 2.0.0 (2026-01-12)

### 🎉 主要功能更新

#### 1. 雙市場支援
- ✅ 新增台股支援，系統現在同時支援美股與台股分析
- ✅ 自動偵測股票代碼類型：
  - 數字代碼（如：2330）→ 台股 🇹🇼
  - 英文代碼（如：TSLA）→ 美股 🇺🇸

#### 2. 動態 API Key 切換
- ✅ 根據股票代碼類型自動切換 API Key 輸入框
- ✅ 台股使用 FindMind API Key
- ✅ 美股使用 FMP API Key
- ✅ 提供清晰的視覺提示顯示當前偵測到的股票類型

#### 3. 新增 FindMind API 整合
- ✅ 實作 `get_taiwan_stock_data()` 函數
- ✅ 支援 FindMind API 獲取台股歷史數據
- ✅ API 端點：`https://api.finmindtrade.com/api/v4/data`
- ✅ 支援 Authorization Bearer Token 認證
- ✅ 完整的錯誤處理機制

#### 4. 資料格式統一
- ✅ 統一處理台股與美股的資料格式
- ✅ 標準化欄位名稱：date, open, high, low, close, volume
- ✅ 自動資料型態轉換與驗證
- ✅ 處理缺失值與異常資料

#### 5. UI/UX 改進
- ✅ 更新系統標題為「AI 股票趨勢分析系統 (美股與台股)」
- ✅ 圖表標題顯示股票市場類型（台股/美股）
- ✅ 貨幣單位自動切換：
  - 台股顯示 NT$（新台幣）
  - 美股顯示 $（美元）
- ✅ 更新使用說明，加入台股操作指引
- ✅ 新增股票代碼範例（台股與美股）

### 🔧 技術改進

#### 新增函數
1. **`is_taiwan_stock(symbol)`**
   - 判斷股票代碼是否為台股（純數字）
   - 返回布林值

2. **`get_taiwan_stock_data(symbol, api_key, start_date, end_date)`**
   - 從 FindMind API 獲取台股歷史數據
   - 處理資料格式轉換
   - 實施完整錯誤處理

#### 修改函數
1. **`create_enhanced_chart()`**
   - 新增 `is_taiwan` 參數
   - 根據市場類型調整圖表標題和貨幣單位

2. **主要分析邏輯**
   - 根據股票類型動態選擇 API
   - 統一的資料處理流程
   - 改進的錯誤訊息

### 📦 依賴套件

無需新增套件，現有依賴已足夠：
- `streamlit>=1.52.2`
- `openai>=2.15.0`
- `plotly>=6.5.1`
- `requests` (由 streamlit 提供)
- `pandas` (由 streamlit 提供)
- `numpy` (由 pandas 提供)

### 🔍 使用範例

#### 台股分析
```
股票代碼：2330
FindMind API Key：您的 FindMind API Key
OpenAI API Key：您的 OpenAI API Key
```

#### 美股分析
```
股票代碼：TSLA
FMP API Key：您的 FMP API Key
OpenAI API Key：您的 OpenAI API Key
```

### ⚠️ 注意事項

1. **API Key 需求**
   - 台股需要 FindMind API Key (https://finmindtrade.com/)
   - 美股需要 FMP API Key (https://financialmodelingprep.com/)
   - 兩者都需要 OpenAI API Key (https://platform.openai.com)

2. **股票代碼格式**
   - 台股：純數字（如：2330, 2317, 2454）
   - 美股：包含英文（如：TSLA, MSFT, GOOGL）

3. **資料來源**
   - 台股：FindMind API
   - 美股：Financial Modeling Prep API

### 🐛 錯誤處理

系統提供完整的錯誤處理：
- API 連線失敗提示
- 股票代碼驗證
- 資料格式驗證
- 欄位缺失檢查
- 友善的錯誤訊息

### 🎯 未來改進建議

1. 支援更多技術指標（MACD, KD 等）
2. 加入即時報價功能
3. 支援多股票比較
4. 匯出分析報告功能
5. 加入港股、A股支援

---

**修改完成日期**：2026-01-12
**修改依據**：AI 股價分析系統修改說明書
**開發者**：AI Assistant
