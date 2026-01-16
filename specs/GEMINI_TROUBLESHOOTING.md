# Google Gemini API 故障排除指南

## 🔍 問題：AI 技術分析沒有內容

如果您在使用 Google Gemini 時看到「AI技術分析（Google Gemini）- 台股」但沒有任何分析內容，請按照以下步驟排查。

---

## 📋 快速檢查清單

### 1. API Key 驗證 ✓

**檢查項目**：
- [ ] API Key 已正確輸入（無多餘空格）
- [ ] API Key 來自 https://aistudio.google.com/app/apikey
- [ ] API Key 是有效的（未過期）

**如何驗證**：
```bash
# 運行測試腳本
uv run python test_gemini_api.py

# 或手動輸入 API Key
uv run python test_gemini_api.py --api-key "您的API_KEY"
```

### 2. 網路連線 ✓

**檢查項目**：
- [ ] 電腦可以連接網際網路
- [ ] 可以訪問 Google 服務
- [ ] 防火牆未封鎖 Gemini API

**測試方法**：
```bash
# 在瀏覽器中訪問
https://generativelanguage.googleapis.com/
```

### 3. API 配額 ✓

**檢查項目**：
- [ ] 未超過每分鐘請求限制（15 次/分鐘）
- [ ] 未超過每日配額
- [ ] 帳號狀態正常

**如何檢查**：
- 訪問 https://aistudio.google.com/
- 查看 API 使用統計

### 4. 模型名稱 ✓

**當前使用**：`gemini-2.5-flash`

**可用模型**：
- `gemini-2.5-flash` ✅（快速版本）
- `gemini-2.5-pro`（進階版本）
- `gemini-2.0-flash-exp`（實驗版本）

---

## 🛠️ 診斷步驟

### 步驟 1：執行測試腳本

```bash
cd /path/to/code_gym-ai-investment
uv run python test_gemini_api.py
```

**預期輸出**：
```
[OK] 模型創建成功: gemini-2.5-flash
[OK] API 請求成功
[OK] response.text 存在
[*] 回應內容:
技術分析是透過研究歷史價格和成交量數據...
```

**如果失敗**，查看錯誤訊息並參考下方「常見錯誤」。

### 步驟 2：檢查應用程式日誌

在 Streamlit 應用程式中：
1. 查看錯誤訊息（紅色框）
2. 展開「🔍 查看詳細錯誤信息」
3. 記錄完整錯誤訊息

### 步驟 3：驗證 API Key

```python
# 簡單驗證腳本
import google.generativeai as genai

api_key = "您的API_KEY"
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content("Hello")
print(response.text)
```

---

## ❌ 常見錯誤與解決方案

### 錯誤 1：API Key 無效

**錯誤訊息**：
```
API key not valid. Please pass a valid API key.
```

**解決方案**：
1. 前往 https://aistudio.google.com/app/apikey
2. 確認 API Key 狀態
3. 如需要，創建新的 API Key
4. 重新複製並貼到應用程式中

### 錯誤 2：超過配額

**錯誤訊息**：
```
Resource has been exhausted (e.g. check quota).
```

**解決方案**：
1. 等待 1 分鐘後重試
2. 檢查 API 使用統計
3. 考慮升級到付費方案

### 錯誤 3：內容被安全過濾

**錯誤訊息**：
```
The response was blocked due to SAFETY.
```

**解決方案**：
1. 檢查股票數據是否包含敏感內容
2. 調整提示語
3. 使用不同的股票進行測試

### 錯誤 4：回應為空

**症狀**：
- 沒有錯誤訊息
- 但也沒有分析內容

**解決方案**：
1. 執行 `test_gemini_api.py` 檢查 API 是否正常
2. 查看應用程式是否顯示警告訊息
3. 檢查是否有安全過濾問題

### 錯誤 5：模型不存在

**錯誤訊息**：
```
models/gemini-xxx does not exist
```

**解決方案**：
1. 確認使用 `gemini-2.5-flash`
2. 更新 `google-generativeai` 套件：
```bash
uv add google-generativeai --upgrade
```

---

## 🔧 進階故障排除

### 檢查套件版本

```bash
uv run python -c "import google.generativeai as genai; print(genai.__version__)"
```

**建議版本**：>= 0.8.0

### 啟用詳細日誌

在 `app.py` 中加入：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 測試不同模型

修改 `app.py` 中的模型名稱：
```python
# 嘗試其他模型
model = genai.GenerativeModel('gemini-2.0-flash-exp')
```

---

## 📞 問題回報

如果以上步驟都無法解決問題，請提供：

1. **測試腳本輸出**
   ```bash
   uv run python test_gemini_api.py > gemini_test.txt 2>&1
   ```

2. **應用程式錯誤訊息**
   - 截圖或複製完整錯誤

3. **環境資訊**
   - 作業系統
   - Python 版本
   - `google-generativeai` 版本

4. **重現步驟**
   - 使用的股票代碼
   - 選擇的日期範圍
   - 其他參數設定

---

## ✅ 成功案例確認

如果 Gemini API 正常工作，您應該看到：

1. **測試腳本成功**
   ```
   [OK] 成功獲取分析內容
   [*] 回應內容:
   [完整的技術分析文字...]
   ```

2. **應用程式顯示分析**
   - 清楚的技術分析標題
   - 完整的分析內容（多段落）
   - 包含趨勢、RSI、風險評估等

3. **無錯誤訊息**
   - 沒有紅色錯誤框
   - 沒有警告訊息

---

## 💡 最佳實踐

### 1. API Key 管理
- 不要在程式碼中硬編碼 API Key
- 定期更換 API Key
- 使用環境變數存儲

### 2. 錯誤處理
- 總是檢查 API 回應
- 實施重試機制
- 記錄錯誤以便調試

### 3. 配額管理
- 避免短時間內大量請求
- 實施請求快取
- 監控使用量

---

## 🔗 相關資源

- **Google AI Studio**: https://aistudio.google.com/
- **API 文檔**: https://ai.google.dev/docs
- **測試腳本**: `test_gemini_api.py`
- **主程式**: `src/app.py`

---

**更新日期**：2026-01-13
**版本**：1.0
