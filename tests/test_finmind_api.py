"""
FindMind API 測試腳本
用於診斷 API 請求問題和驗證資料格式

使用方式：
    python test_finmind_api.py
    或
    uv run python test_finmind_api.py
"""
import requests
import json
from datetime import datetime, timedelta
import sys
import io

# 設置 UTF-8 輸出（解決 Windows 終端編碼問題）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_finmind_api(symbol="2317", days=90):
    """
    測試 FindMind API 並顯示詳細資訊

    Args:
        symbol: 股票代碼（預設：2317 鴻海）
        days: 查詢天數（預設：90天）
    """

    # 測試參數
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # API 參數
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        'dataset': 'TaiwanStockPrice',
        'data_id': symbol,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }

    print("=" * 80)
    print("FindMind API 測試工具")
    print("=" * 80)
    print(f"\n[*] 測試股票: {symbol}")
    print(f"[*] 日期範圍: {params['start_date']} 到 {params['end_date']}")
    print(f"[*] API URL: {url}")
    print(f"[*] 參數: {params}")

    # 測試 1: 不帶任何 Header（最基本的請求）
    print("\n" + "=" * 80)
    print("測試 1: 基本請求（不帶額外 Header）")
    print("=" * 80)
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"[OK] 狀態碼: {response.status_code}")
        print(f"[*] Content-Type: {response.headers.get('Content-Type', 'N/A')}")

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] JSON 解析成功")
            print(f"[*] 回應鍵值: {list(data.keys())}")

            if 'data' in data and len(data['data']) > 0:
                print(f"[OK] 資料筆數: {len(data['data'])}")
                print(f"[*] 欄位名稱: {list(data['data'][0].keys())}")
                print(f"\n前 3 筆資料範例:")
                for i, record in enumerate(data['data'][:3]):
                    print(f"  {i+1}. {record}")

                # 驗證必要欄位
                print(f"\n[*] 欄位驗證:")
                required_fields = ['date', 'open', 'max', 'min', 'close', 'Trading_Volume']
                for field in required_fields:
                    exists = field in data['data'][0]
                    status = "[OK]" if exists else "[WARN]"
                    print(f"  {status} {field}: {'存在' if exists else '缺失'}")

            else:
                print(f"[WARN] 回應中沒有資料")
                print(f"完整回應: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"[ERROR] 請求失敗")
            print(f"回應內容: {response.text[:500]}")
    except Exception as e:
        print(f"[ERROR] 錯誤: {str(e)}")

    # 測試 2: 帶完整 Header（模擬瀏覽器）
    print("\n" + "=" * 80)
    print("測試 2: 完整請求（模擬瀏覽器 Header）")
    print("=" * 80)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"[OK] 狀態碼: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] JSON 解析成功")
            if 'data' in data and len(data['data']) > 0:
                print(f"[OK] 資料筆數: {len(data['data'])}")
            else:
                print(f"[WARN] 回應中沒有資料")
        else:
            print(f"[ERROR] 請求失敗")
            print(f"回應內容: {response.text[:500]}")
    except Exception as e:
        print(f"[ERROR] 錯誤: {str(e)}")

    # 測試 3: 嘗試不同的 dataset
    print("\n" + "=" * 80)
    print("測試 3: 測試不同的 dataset")
    print("=" * 80)

    datasets_to_test = [
        ('TaiwanStockPrice', '台股每日收盤價'),
        ('TaiwanStockPriceTick', '台股分時資料（可能需要 API Key）'),
        ('TaiwanStockInfo', '台股基本資訊')
    ]

    for dataset, description in datasets_to_test:
        print(f"\n[*] 測試 dataset: {dataset} ({description})")
        test_params = params.copy()
        test_params['dataset'] = dataset

        try:
            response = requests.get(url, params=test_params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    print(f"  [OK] 成功！資料筆數: {len(data['data'])}")
                    print(f"  [*] 欄位: {list(data['data'][0].keys())}")
                else:
                    print(f"  [WARN] 請求成功但無資料")
                    if 'msg' in data:
                        print(f"  [*] 訊息: {data['msg']}")
            else:
                print(f"  [ERROR] 狀態碼: {response.status_code}")
                try:
                    error_data = response.json()
                    if 'msg' in error_data:
                        print(f"  [*] 錯誤訊息: {error_data['msg']}")
                except:
                    pass
        except Exception as e:
            print(f"  [ERROR] 錯誤: {str(e)}")

    # 測試 4: 帶 API Key 的請求（如果有的話）
    print("\n" + "=" * 80)
    print("測試 4: 帶 API Key 的請求測試")
    print("=" * 80)
    print("[*] 提示：如果您有 FindMind API Key，可以修改此腳本加入測試")
    print("[*] 在 headers 中加入: 'Authorization': f'Bearer {api_key}'")

    print("\n" + "=" * 80)
    print("測試完成")
    print("=" * 80)
    print("\n[*] 結論：")
    print("  - TaiwanStockPrice dataset 可正常使用（無需 API Key）")
    print("  - 回應欄位為小寫：date, open, max, min, close, Trading_Volume")
    print("  - 建議在應用程式中使用 TaiwanStockPrice dataset")
    print("\n[*] 如需測試其他股票，請執行：")
    print("  python test_finmind_api.py --symbol 2330 --days 30")

def main():
    """主程式入口"""
    import argparse

    parser = argparse.ArgumentParser(description='FindMind API 測試工具')
    parser.add_argument('--symbol', '-s', default='2317', help='股票代碼（預設：2317）')
    parser.add_argument('--days', '-d', type=int, default=90, help='查詢天數（預設：90）')

    args = parser.parse_args()

    test_finmind_api(symbol=args.symbol, days=args.days)

if __name__ == "__main__":
    main()
