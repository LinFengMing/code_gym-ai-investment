# 📈 AI 股票趨勢分析系統 (美股與台股)

專業的股票技術分析工具，支援美股與台股市場，提供 K 線圖、技術指標和 AI 智能分析。

## ✨ 功能特色

### 🌍 雙市場支援
- **台股**：支援台灣證券交易所上市股票
- **美股**：支援美國主要交易所股票
- 自動偵測股票代碼類型並切換對應 API

### 📊 專業技術分析
- **K線圖**：互動式燭台圖表，支援縮放和懸停顯示
- **移動平均線**：MA5, MA10, MA20, MA60
- **RSI 指標**：相對強弱指標，分析超買超賣狀態
- **成交量分析**：視覺化交易量變化

### 🤖 AI 智能分析
- 支援 **OpenAI** 和 **Google Gemini** 雙 AI 引擎
- **預設使用 OpenAI GPT-4o-mini** - 穩定可靠，回應品質一致
- 備選 Google Gemini 2.5 Pro - 高階推理模型，免費額度高
- 深度技術面分析與客觀的歷史數據解讀
- 專業的技術指標說明
- 純教育性分析，不提供投資建議

### 🎯 彈性配置
- API Key 可選填，降低使用門檻
- 自訂 RSI 計算週期
- 靈活的日期範圍選擇

## 🚀 快速開始

### 環境需求

- Python 3.14+
- uv (Python 套件管理工具)

### 安裝步驟

```bash
# 1. 克隆專案
git clone <repository-url>
cd code_gym-ai-investment

# 2. 安裝依賴
uv sync

# 3. 啟動應用程式
uv run streamlit run src/app.py
```

### 瀏覽器訪問

應用程式啟動後，開啟瀏覽器訪問：
```
http://localhost:8501
```

## 📝 使用說明

### 分析台股

1. 輸入股票代碼（純數字，如：`2330`）
2. 系統自動偵測為台股
3. 輸入 FindMind API Key（可選，留空使用免費額度）
    4. 選擇 AI 提供商（預設：**OpenAI**）
    5. 輸入對應的 AI API Key（可選，用於 AI 分析）
6. 選擇日期範圍
7. 調整 RSI 計算週期（預設 14 天）
8. 點擊「🚀 開始分析」

**範例**：
```
股票代碼：2330 (台積電)
FindMind API Key：(留空或填寫)
AI 提供商：OpenAI（預設）
OpenAI API Key：(留空或填寫)
```

### 分析美股

1. 輸入股票代碼（包含英文，如：`TSLA`）
2. 系統自動偵測為美股
3. 輸入 FMP API Key（必填）
    4. 選擇 AI 提供商（預設：**OpenAI**）
    5. 輸入對應的 AI API Key（可選，用於 AI 分析）
6. 選擇日期範圍
7. 調整 RSI 計算週期（預設 14 天）
8. 點擊「🚀 開始分析」

**範例**：
```
股票代碼：TSLA (特斯拉)
FMP API Key：您的 API Key
AI 提供商：OpenAI（預設）
OpenAI API Key：(留空或填寫)
```

## 🔑 API Key 取得

### 台股 - FindMind API
- 註冊網址：https://finmindtrade.com/
- 免費方案：可不填寫 API Key，使用免費額度
- 付費方案：提供更多資料和請求次數

### 美股 - Financial Modeling Prep (FMP)
- 註冊網址：https://financialmodelingprep.com/
- 需要註冊並取得 API Key
- 免費方案：每日有請求次數限制

### AI 分析

**OpenAI**（系統預設）
- 註冊網址：https://platform.openai.com/
- 模型：GPT-4o-mini
- 穩定可靠，回應品質一致
- 適合日常技術分析
- 需要 API Key 和足夠的額度
- 需要綁定信用卡

**Google Gemini**（備選）
- 註冊網址：https://aistudio.google.com/app/apikey
- 模型：**gemini-2.5-pro**（高階推理模型）
- 免費額度高，適合新手和頻繁使用
- 推理能力強，適合深度技術分析
- 無需信用卡，容易取得

**注意**：兩個 AI 提供商可選其一使用，不使用 AI 分析時可留空。系統預設使用 OpenAI。

## 📊 技術指標說明

### 移動平均線 (MA)
- **MA5**：5 日移動平均線，短期趨勢指標
- **MA10**：10 日移動平均線，短中期趨勢指標
- **MA20**：20 日移動平均線，中期趨勢指標
- **MA60**：60 日移動平均線，長期趨勢指標

### RSI 相對強弱指標
- **範圍**：0-100
- **超買**：RSI > 70，可能面臨回調壓力
- **超賣**：RSI < 30，可能出現反彈機會
- **正常**：RSI 30-70，技術面相對平衡

## 🧪 測試工具

專案提供測試工具用於驗證 API 連線和診斷問題：

```bash
# 測試 FindMind API（台股）
python test_finmind_api.py

# 測試特定股票
python test_finmind_api.py --symbol 2330 --days 30
```

詳細說明請參考：[測試工具使用說明](README_TESTING.md)

## 📁 專案結構

```
code_gym-ai-investment/
├── src/
│   └── app.py                    # 主應用程式
├── specs/
│   ├── AI 股價分析系統_規格說明書.md
│   └── AI 股價分析系統_修改說明書.md
├── test_finmind_api.py          # API 測試工具
├── CHANGELOG.md                  # 修改日誌
├── TESTING_GUIDE.md             # 測試指南
├── README_TESTING.md            # 測試工具說明
├── pyproject.toml               # 專案配置
├── uv.lock                      # 依賴鎖定檔
└── README.md                    # 本文檔

```

## 🔧 技術棧

- **前端框架**：Streamlit
- **圖表視覺化**：Plotly
- **資料處理**：Pandas, NumPy
- **AI 模型**：
  - OpenAI GPT-4o-mini（預設）
  - Google Gemini gemini-2.5-pro（備選）
- **API 整合**：
  - FindMind API (台股)
  - Financial Modeling Prep API (美股)

## ⚠️ 免責聲明

本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。

## 📚 相關文件

- [完整測試指南](TESTING_GUIDE.md) - 詳細的功能測試案例
- [測試工具說明](README_TESTING.md) - API 測試工具使用方式
- [修改日誌](CHANGELOG.md) - 版本更新記錄
- [系統規格說明書](specs/AI%20股價分析系統_規格說明書.md) - 完整系統規格
- [修改說明書](specs/AI%20股價分析系統_修改說明書.md) - 客製化需求

## 🐛 問題回報

如遇到問題，請提供：
1. 錯誤訊息截圖
2. 使用的股票代碼和參數
3. 系統環境資訊（作業系統、Python 版本）
4. 測試工具的輸出結果

## 📞 技術支援

- 執行測試腳本診斷問題：`python test_finmind_api.py`
- 查看詳細錯誤訊息和 API 回應
- 參考測試指南進行故障排除

## 🎯 常見範例

### 台股熱門股票
- **2330** - 台積電
- **2317** - 鴻海
- **2454** - 聯發科
- **2308** - 台達電
- **2881** - 富邦金

### 美股熱門股票
- **TSLA** - 特斯拉
- **MSFT** - 微軟
- **GOOGL** - Google
- **AMZN** - 亞馬遜
- **NVDA** - 輝達

## 📈 版本資訊

- **當前版本**：2.1.0
- **最後更新**：2026-01-13
- **主要功能**：雙市場支援、RSI 指標、雙 AI 引擎（OpenAI + Gemini）、API Key 可選填

詳細更新記錄請參考 [CHANGELOG.md](CHANGELOG.md)

---

**開始您的技術分析之旅吧！** 📈

Made with ❤️ by Code Gym
