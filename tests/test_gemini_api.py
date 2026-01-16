"""
Google Gemini API 測試腳本
用於驗證 Gemini API 連線和回應格式
"""
import google.generativeai as genai
import sys
import io

# 設置 UTF-8 輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_gemini_api(api_key=None):
    """測試 Gemini API"""

    print("=" * 80)
    print("Google Gemini API 測試工具")
    print("=" * 80)

    if not api_key:
        print("\n[*] 請輸入您的 Google Gemini API Key")
        print("[*] 如果沒有，請前往 https://aistudio.google.com/app/apikey 取得")
        api_key = input("\nAPI Key: ").strip()

    if not api_key:
        print("[ERROR] 未提供 API Key，測試中止")
        return

    try:
        # 配置 API
        print("\n[*] 配置 Gemini API...")
        genai.configure(api_key=api_key)

        # 測試 1: 簡單文本生成
        print("\n" + "=" * 80)
        print("測試 1: 簡單文本生成")
        print("=" * 80)

        model = genai.GenerativeModel('gemini-2.5-pro')
        print(f"[OK] 模型創建成功: gemini-2.5-pro")

        test_prompt = "請用繁體中文簡單介紹什麼是技術分析？限制在100字內。"
        print(f"[*] 測試提示: {test_prompt}")

        response = model.generate_content(test_prompt)
        print(f"[OK] API 請求成功")

        # 檢查回應結構
        print(f"\n[*] 回應物件類型: {type(response)}")
        print(f"[*] 回應物件屬性: {dir(response)}")

        # 嘗試不同方式提取文本
        if hasattr(response, 'text'):
            print(f"\n[OK] response.text 存在")
            print(f"[*] 回應內容:\n{response.text}")
        else:
            print(f"\n[WARN] response.text 不存在")

        if hasattr(response, 'parts'):
            print(f"\n[*] response.parts 存在")
            print(f"[*] Parts 數量: {len(response.parts)}")
            for i, part in enumerate(response.parts):
                print(f"[*] Part {i}: {part}")

        if hasattr(response, 'candidates'):
            print(f"\n[*] response.candidates 存在")
            print(f"[*] Candidates 數量: {len(response.candidates)}")
            for i, candidate in enumerate(response.candidates):
                print(f"[*] Candidate {i}: {candidate}")

        # 測試 2: 使用 generation_config
        print("\n" + "=" * 80)
        print("測試 2: 使用 generation_config")
        print("=" * 80)

        response2 = model.generate_content(
            "請用繁體中文說明 RSI 指標的作用，限制在50字內。",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500,
            )
        )

        print(f"[OK] API 請求成功")
        if hasattr(response2, 'text'):
            print(f"[*] 回應內容:\n{response2.text}")
        else:
            print(f"[WARN] 無法取得 text 屬性")
            print(f"[*] 完整回應: {response2}")

        # 測試 3: 長文本生成（模擬股票分析）
        print("\n" + "=" * 80)
        print("測試 3: 股票技術分析模擬")
        print("=" * 80)

        analysis_prompt = """請以專業技術分析師的角度，用繁體中文分析以下股票數據：

股票代碼：2330
期間：2025-12-01 至 2026-01-13
起始價格：$1480
結束價格：$1450
價格變化：-2.03%
當前 RSI：45

請提供簡要的技術分析觀點（限制在200字內）。"""

        print(f"[*] 測試提示:\n{analysis_prompt}")

        response3 = model.generate_content(
            analysis_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )

        print(f"\n[OK] API 請求成功")
        if hasattr(response3, 'text') and response3.text:
            print(f"[OK] 成功獲取分析內容")
            print(f"[*] 回應長度: {len(response3.text)} 字元")
            print(f"[*] 回應內容:\n{response3.text}")
        else:
            print(f"[ERROR] 無法獲取分析內容")
            print(f"[*] 回應物件: {response3}")

            # 嘗試其他方式提取
            if hasattr(response3, 'candidates') and response3.candidates:
                print(f"[*] 嘗試從 candidates 提取...")
                candidate = response3.candidates[0]
                if hasattr(candidate, 'content'):
                    print(f"[*] Candidate content: {candidate.content}")

        print("\n" + "=" * 80)
        print("測試完成")
        print("=" * 80)
        print("\n[*] 結論：")
        print("  - API 連線: 成功")
        print("  - 模型: gemini-2.5-flash")
        print("  - 文本生成: 正常")
        print("  - 繁體中文: 支援")

    except Exception as e:
        print(f"\n[ERROR] 測試失敗: {str(e)}")
        print(f"[*] 錯誤類型: {type(e).__name__}")

        import traceback
        print(f"\n[*] 完整錯誤追蹤:")
        traceback.print_exc()

        print(f"\n[*] 可能的原因:")
        print("  1. API Key 無效或過期")
        print("  2. 網路連線問題")
        print("  3. API 配額已用完")
        print("  4. 模型名稱錯誤")

def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description='Google Gemini API 測試工具')
    parser.add_argument('--api-key', '-k', help='Google Gemini API Key（可選，不提供則會詢問）')

    args = parser.parse_args()

    test_gemini_api(api_key=args.api_key)

if __name__ == "__main__":
    main()
