# 測試工具使用說明

## 📋 概述

本專案提供測試工具來驗證 API 連線和資料格式。

## 🛠️ 可用測試工具

### 1. FindMind API 測試腳本

**檔案**: `test_finmind_api.py`

**用途**: 測試 FindMind API 連線、驗證資料格式、診斷問題

#### 基本使用

```bash
# 使用預設參數（股票：2317，天數：90天）
python test_finmind_api.py

# 或使用 uv
uv run python test_finmind_api.py
```

#### 進階使用

```bash
# 測試其他股票
python test_finmind_api.py --symbol 2330

# 指定查詢天數
python test_finmind_api.py --symbol 2330 --days 30

# 完整參數
python test_finmind_api.py --symbol 2454 --days 60
```

#### 測試項目

測試腳本會執行以下檢查：

1. ✅ **基本請求測試**
   - 不帶額外 Header 的最基本請求
   - 驗證 API 回應狀態
   - 檢查資料格式

2. ✅ **瀏覽器模擬測試**
   - 帶完整 User-Agent Header
   - 驗證與瀏覽器行為一致性

3. ✅ **Dataset 測試**
   - TaiwanStockPrice（每日收盤價）
   - TaiwanStockPriceTick（分時資料）
   - TaiwanStockInfo（股票基本資訊）

4. ✅ **欄位驗證**
   - 檢查必要欄位是否存在
   - 顯示實際欄位名稱
   - 提供資料範例

#### 輸出範例

```
================================================================================
FindMind API 測試工具
================================================================================

[*] 測試股票: 2317
[*] 日期範圍: 2025-10-15 到 2026-01-13
[*] API URL: https://api.finmindtrade.com/api/v4/data

================================================================================
測試 1: 基本請求（不帶額外 Header）
================================================================================
[OK] 狀態碼: 200
[OK] JSON 解析成功
[OK] 資料筆數: 61
[*] 欄位名稱: ['date', 'stock_id', 'Trading_Volume', ...]

[*] 欄位驗證:
  [OK] date: 存在
  [OK] open: 存在
  [OK] max: 存在
  [OK] min: 存在
  [OK] close: 存在
  [OK] Trading_Volume: 存在
```

## 🐛 故障排除

### 問題 1: 編碼錯誤

**錯誤訊息**:
```
UnicodeEncodeError: 'cp950' codec can't encode character
```

**解決方案**:
腳本已包含 UTF-8 編碼設定，如仍有問題，請在 PowerShell 執行：
```powershell
$env:PYTHONIOENCODING="utf-8"
python test_finmind_api.py
```

### 問題 2: 400 錯誤

**可能原因**:
- 股票代碼不存在
- Dataset 名稱錯誤
- 日期範圍無效

**解決方案**:
- 確認股票代碼正確（如：2330, 2317）
- 使用 `TaiwanStockPrice` dataset
- 確認日期範圍合理

### 問題 3: 無資料回應

**可能原因**:
- 選擇的日期範圍內沒有交易日
- 股票已下市或暫停交易

**解決方案**:
- 調整日期範圍
- 確認股票仍在市場交易

## 📊 常用測試案例

### 台股熱門股票

```bash
# 台積電
python test_finmind_api.py --symbol 2330

# 鴻海
python test_finmind_api.py --symbol 2317

# 聯發科
python test_finmind_api.py --symbol 2454

# 台達電
python test_finmind_api.py --symbol 2308
```

### 不同時間範圍

```bash
# 最近 30 天
python test_finmind_api.py --symbol 2330 --days 30

# 最近 6 個月
python test_finmind_api.py --symbol 2330 --days 180

# 最近 1 年
python test_finmind_api.py --symbol 2330 --days 365
```

## 💡 使用建議

1. **新增功能前先測試**
   - 在修改主程式前，使用測試腳本驗證 API 行為
   - 確認資料格式符合預期

2. **診斷問題時使用**
   - 遇到 API 錯誤時，執行測試腳本查看詳細資訊
   - 比對測試結果與主程式行為

3. **記錄測試結果**
   - 可將輸出重定向到檔案：
   ```bash
   python test_finmind_api.py > test_result.txt
   ```

4. **自訂測試**
   - 可以修改腳本測試其他 API 參數
   - 加入您需要的特定驗證邏輯

## 🔗 相關資源

- **FindMind API 文檔**: https://finmindtrade.com/
- **完整測試指南**: 參考 `TESTING_GUIDE.md`
- **修改記錄**: 參考 `CHANGELOG.md`

## 📞 技術支援

如果測試腳本發現問題，請記錄：
1. 完整的測試輸出
2. 使用的參數（股票代碼、日期範圍）
3. 預期結果 vs 實際結果
4. 系統環境資訊

---

**最後更新**: 2026-01-13
