import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import re

# 嘗試使用新版 google.genai，如果失敗則使用舊版
try:
    from google import genai
    USING_NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    USING_NEW_GENAI = False

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統 (美股與台股)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("📈 AI 股票趨勢分析系統 (美股與台股)")
st.divider()

def is_taiwan_stock(symbol):
    """
    判斷股票代碼是否為台股（數字代碼）

    Args:
        symbol: 股票代碼

    Returns:
        bool: True 表示台股（純數字），False 表示美股（包含英文）
    """
    # 移除空白並轉換為大寫
    symbol = symbol.strip().upper()
    # 判斷是否為純數字（台股）
    return symbol.isdigit()

def get_taiwan_stock_data(symbol, api_key, start_date, end_date):
    """
    從 FindMind API 獲取台股歷史數據

    Args:
        symbol: 台股股票代碼（數字）
        api_key: FindMind API金鑰（可為空）
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        DataFrame: 包含股票歷史數據的DataFrame
    """
    try:
        # 構建API請求URL
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }

        # 設置請求標頭（模擬瀏覽器）
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        }

        # 只有在 API Key 不為空時才加入 Authorization
        if api_key and api_key.strip():
            headers['Authorization'] = f'Bearer {api_key}'

        # 發送API請求
        response = requests.get(url, params=params, headers=headers, timeout=30)

        # 詳細的錯誤處理
        if response.status_code != 200:
            error_msg = f"FindMind API 請求失敗 (狀態碼: {response.status_code})"
            try:
                error_data = response.json()
                if 'msg' in error_data:
                    error_msg += f"\n錯誤訊息: {error_data['msg']}"
            except:
                error_msg += f"\n回應內容: {response.text[:200]}"
            st.error(error_msg)
            return None

        data = response.json()

        # 檢查API響應
        if 'data' not in data or len(data['data']) == 0:
            st.warning(f"FindMind API 回應中沒有股票 {symbol} 的數據。請確認：\n1. 股票代碼是否正確\n2. 日期範圍內是否有交易數據\n3. 是否需要 API Key")
            return None

        # 轉換為DataFrame
        df = pd.DataFrame(data['data'])

        # 調試資訊：顯示實際收到的欄位
        if len(df) > 0:
            st.info(f"📊 成功獲取 {len(df)} 筆資料。欄位：{', '.join(df.columns.tolist())}")

        # FindMind API 的資料欄位映射與處理
        # 需要將欄位名稱統一為標準格式：date, open, high, low, close, volume

        # 嘗試轉換日期欄位
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            st.error("資料中缺少 'date' 欄位")
            return None

        # FindMind API 的欄位名稱映射（根據實際 API 回應）
        # 實際欄位: date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover
        column_mapping = {
            'Trading_Volume': 'volume',      # 成交量
            'Trading_money': 'trading_money', # 交易金額
            'max': 'high',                    # 最高價
            'min': 'low',                     # 最低價
            'spread': 'spread',               # 漲跌幅
            'Trading_turnover': 'turnover'    # 週轉率
        }

        # 檢查並重命名欄位
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        # 確保必要欄位存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"資料欄位缺失：{', '.join(missing_columns)}")
            st.info(f"可用欄位：{', '.join(df.columns.tolist())}")
            return None

        # 選擇需要的欄位並排序
        df = df[required_columns].copy()
        df = df.sort_values('date').reset_index(drop=True)

        # 轉換資料型態
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # 移除包含 NaN 的行
        df = df.dropna()

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"FindMind API請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"台股數據處理錯誤：{str(e)}")
        return None

def get_stock_data(symbol, api_key, start_date, end_date):
    """
    從Financial Modeling Prep API獲取股票歷史數據

    Args:
        symbol: 股票代碼
        api_key: FMP API金鑰
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        DataFrame: 包含股票歷史數據的DataFrame
    """
    try:
        # 構建API請求URL
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            'symbol': symbol,
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        # 發送API請求
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # 檢查API響應 - 新API直接回傳陣列
        if not isinstance(data, list) or len(data) == 0:
            st.error(f"無法獲取股票 {symbol} 的數據，請檢查股票代碼是否正確。")
            return None

        # 轉換為DataFrame - 新API直接回傳歷史數據陣列
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"API請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"數據處理錯誤：{str(e)}")
        return None

def filter_by_date_range(df, start_date, end_date):
    """
    根據日期範圍過濾數據

    Args:
        df: 股票數據DataFrame
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        DataFrame: 過濾後的數據
    """
    if df is None:
        return None

    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask].copy()

    return filtered_df.reset_index(drop=True)

def calculate_rsi(df, period=14):
    """
    計算RSI相對強弱指標

    Args:
        df: 股票數據DataFrame
        period: RSI計算週期，預設為14天

    Returns:
        DataFrame: 包含RSI指標的數據
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    # 計算價格變化
    df['price_change'] = df['close'].diff()

    # 分離漲幅和跌幅
    df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
    df['loss'] = -df['price_change'].where(df['price_change'] < 0, 0)

    # 計算平均漲幅和平均跌幅
    df['avg_gain'] = df['gain'].rolling(window=period, min_periods=1).mean()
    df['avg_loss'] = df['loss'].rolling(window=period, min_periods=1).mean()

    # 計算相對強度 (RS) 和 RSI
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + df['rs']))

    # 處理初始值的NaN
    df['rsi'] = df['rsi'].fillna(50)  # 初始值設為中性50

    return df

def get_moving_averages(df):
    """
    計算移動平均線（MA5, MA10, MA20, MA60）

    Args:
        df: 股票數據DataFrame

    Returns:
        DataFrame: 包含移動平均線的數據
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    # 計算移動平均線
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()

    return df

def create_enhanced_chart(df, symbol, rsi_period=14, is_taiwan=False):
    """
    創建包含K線圖、移動平均線和RSI指標的綜合圖表

    Args:
        df: 包含股票數據、移動平均線和RSI的DataFrame
        symbol: 股票代碼
        rsi_period: RSI計算週期
        is_taiwan: 是否為台股

    Returns:
        plotly.graph_objects.Figure: 互動式圖表
    """
    # 創建子圖表：價格圖、成交量圖、RSI圖
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('價格與移動平均線', '成交量', 'RSI相對強弱指標'),
        row_heights=[0.5, 0.2, 0.3]
    )

    # === 第一行：K線圖和移動平均線 ===
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ),
        row=1, col=1
    )

    # 移動平均線
    ma_colors = {
        'MA5': '#ff6b6b',
        'MA10': '#4ecdc4',
        'MA20': '#45b7d1',
        'MA60': '#96ceb4'
    }

    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[ma],
                mode='lines',
                name=ma,
                line=dict(color=ma_colors[ma], width=2)
            ),
            row=1, col=1
        )

    # === 第二行：成交量 ===
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color='#a55eea',
            opacity=0.6
        ),
        row=2, col=1
    )

    # === 第三行：RSI指標 ===
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['rsi'],
            mode='lines',
            name=f'RSI({rsi_period})',
            line=dict(color='#3742fa', width=2)
        ),
        row=3, col=1
    )

    # RSI超買線（70）
    fig.add_hline(
        y=70,
        line=dict(color='red', width=1, dash='dash'),
        annotation_text="超買區（70）",
        annotation_position="bottom right",
        row=3, col=1
    )

    # RSI超賣線（30）
    fig.add_hline(
        y=30,
        line=dict(color='green', width=1, dash='dash'),
        annotation_text="超賣區（30）",
        annotation_position="top right",
        row=3, col=1
    )

    # RSI中線（50）
    fig.add_hline(
        y=50,
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text="中線（50）",
        annotation_position="bottom right",
        row=3, col=1
    )

    # 添加RSI超買超賣背景色
    # 超買區域背景（RSI > 70）
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        row=3, col=1
    )

    # 超賣區域背景（RSI < 30）
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below",
        line_width=0,
        row=3, col=1
    )

    # 更新佈局
    stock_market = "台股" if is_taiwan else "美股"
    fig.update_layout(
        title=f'{symbol} 股價技術分析圖表（含RSI指標）- {stock_market}',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )

    # 更新各軸標籤
    currency_label = "價格 (TWD)" if is_taiwan else "價格 (USD)"
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text=currency_label, row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig

def get_rsi_signal(current_rsi):
    """
    根據RSI值判斷超買超賣狀態

    Args:
        current_rsi: 當前RSI值

    Returns:
        tuple: (狀態, 顏色)
    """
    if current_rsi >= 70:
        return "超買狀態", "🔴"
    elif current_rsi <= 30:
        return "超賣狀態", "🟢"
    else:
        return "正常區間", "🟡"

def generate_ai_insights(symbol, stock_data, ai_api_key, ai_provider, start_date, end_date):
    """
    使用 AI 進行技術分析（支援 OpenAI 和 Google Gemini）

    Args:
        symbol: 股票代碼
        stock_data: 股票數據DataFrame
        ai_api_key: AI API金鑰
        ai_provider: AI 提供商 ('openai' 或 'gemini')
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        str: AI分析結果
    """
    try:

        # 準備數據
        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100

        # 獲取最新RSI值
        current_rsi = stock_data['rsi'].iloc[-1]
        rsi_signal, rsi_icon = get_rsi_signal(current_rsi)

        # 準備關鍵數據摘要（避免傳送過多數據）
        # 只傳送最近30筆和關鍵統計數據
        recent_data = stock_data.tail(30)
        data_json = recent_data.to_json(orient='records', date_format='iso')

        # 計算關鍵統計數據
        price_high = stock_data['close'].max()
        price_low = stock_data['close'].min()
        avg_volume = stock_data['volume'].mean()
        current_ma5 = stock_data['MA5'].iloc[-1]
        current_ma20 = stock_data['MA20'].iloc[-1]
        current_ma60 = stock_data['MA60'].iloc[-1]

        # 構建AI提示語
        system_message = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識
5. 專業解讀RSI相對強弱指標的歷史表現

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{first_date} 至 {last_date}（共 {len(stock_data)} 個交易日）
- 起始價格：${start_price:.2f}
- 結束價格：${end_price:.2f}
- 期間價格變化：{price_change:.2f}%
- 期間最高價：${price_high:.2f}
- 期間最低價：${price_low:.2f}
- 平均成交量：{avg_volume:,.0f}

### 當前技術指標
- 當前 RSI：{current_rsi:.2f} ({rsi_signal})
- MA5：${current_ma5:.2f}
- MA20：${current_ma20:.2f}
- MA60：${current_ma60:.2f}
- 價格相對 MA20：{'上方' if end_price > current_ma20 else '下方'}

### 最近30日交易數據
以下是最近30個交易日的詳細數據（包含價格、成交量、移動平均線和RSI）：
{data_json}

### 請提供以下分析（請完整回答每個部分）：

1. **趨勢分析**：整體方向、支撐阻力位
2. **技術指標**：MA均線關係、RSI狀態、成交量分析
3. **價格行為**：關鍵突破點、波動性
4. **風險評估**：當前風險等級、支撐阻力區間
5. **技術觀察**：短中期觀察重點

請確保分析內容完整且詳細（至少800字），包含具體數據支撐，使用繁體中文，條理清晰。"""

        # 根據選擇的 AI 提供商調用對應 API
        if ai_provider == "gemini":
            # 使用 Google Gemini API
            genai.configure(api_key=ai_api_key)

            # Gemini 使用單一提示語（結合 system 和 user message）
            combined_prompt = f"{system_message}\n\n{user_prompt}"

            # 根據使用的套件版本選擇不同的調用方式
            if USING_NEW_GENAI:
                # 使用新版 google.genai
                model = genai.GenerativeModel('gemini-2.5-pro')
                response = model.generate_content(
                    combined_prompt,
                    config={
                        'temperature': 0.3,
                        'max_output_tokens': 8000,  # 增加到 8000
                    }
                )
            else:
                # 使用舊版 google.generativeai
                model = genai.GenerativeModel('gemini-2.5-pro')
                response = model.generate_content(
                    combined_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=8000,  # 增加到 8000
                    )
                )

            # 檢查回應是否有內容
            if response and hasattr(response, 'text'):
                text_content = response.text.strip()
                if text_content and len(text_content) > 50:
                    return text_content
                else:
                    st.warning(f"Gemini API 回應內容過短（{len(text_content)} 字元），可能不完整")

            # 嘗試從 parts 中提取文本
            if response and hasattr(response, 'parts'):
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text.strip())

                if text_parts:
                    combined_text = '\n'.join(text_parts)
                    if len(combined_text) > 50:
                        return combined_text
                    else:
                        st.warning(f"Gemini API 回應內容過短（{len(combined_text)} 字元）")

            # 檢查是否有候選回應
            if response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                        if text_parts:
                            combined_text = '\n'.join(text_parts)
                            if combined_text.strip():
                                return combined_text

            # 如果都沒有內容，返回錯誤訊息
            st.error("⚠️ Gemini API 回應中沒有有效內容")
            st.info("可能的原因：\n- 內容被安全過濾\n- API 回應格式變更\n- 網路連線問題")
            return "Gemini AI 分析暫時無法生成完整內容。建議：\n1. 稍後重試\n2. 嘗試其他股票\n3. 切換到 OpenAI"

        else:  # OpenAI (預設)
            # 使用 OpenAI API
            client = OpenAI(api_key=ai_api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.3
            )

            return response.choices[0].message.content

    except Exception as e:
        provider_name = "Google Gemini" if ai_provider == "gemini" else "OpenAI"
        error_msg = str(e)

        # 顯示詳細錯誤信息
        st.error(f"{provider_name} AI 分析失敗：{error_msg}")

        # 針對常見錯誤提供解決建議
        if "API key" in error_msg or "authentication" in error_msg.lower():
            st.warning("💡 提示：請檢查您的 API Key 是否正確")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            st.warning("💡 提示：您可能已超過 API 使用配額，請稍後再試")
        elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            st.warning("💡 提示：內容可能觸發安全過濾，請調整查詢參數")

        # 顯示完整錯誤以便調試
        with st.expander("🔍 查看詳細錯誤信息"):
            st.code(error_msg)

        return f"{provider_name} AI 分析暫時無法使用，請檢查上述錯誤訊息。"

# 側邊欄設置
st.sidebar.markdown("## 🔧 分析設定")
st.sidebar.divider()

# 輸入控制項
symbol = st.sidebar.text_input(
    "股票代碼",
    value="TSLA",
    help="輸入股票代碼：數字代碼為台股（如：2330），英文代碼為美股（如：TSLA, MSFT）"
)

# 判斷股票類型並動態顯示對應的 API Key 輸入框
is_tw_stock = is_taiwan_stock(symbol) if symbol.strip() else False

if is_tw_stock:
    st.sidebar.info("🇹🇼 偵測到台股代碼，請輸入 FindMind API Key")
    api_key = st.sidebar.text_input(
        "FindMind API Key",
        type="password",
        help="請輸入您的 FindMind API 金鑰",
        key="finmind_api_key"
    )
    stock_type = "台股"
else:
    st.sidebar.info("偵測到美股代碼，請輸入 FMP API Key")
    api_key = st.sidebar.text_input(
        "FMP API Key",
        type="password",
        help="請輸入您的 Financial Modeling Prep API 金鑰",
        key="fmp_api_key"
    )
    stock_type = "美股"

# AI 分析設定
st.sidebar.markdown("### 🤖 AI 分析設定")
ai_provider = st.sidebar.selectbox(
    "選擇 AI 提供商",
    options=["openai", "gemini"],  # 調整順序，openai 在前
    index=0,  # 預設選擇第一個（openai）
    format_func=lambda x: "OpenAI (GPT-4o-mini)" if x == "openai" else "Google Gemini (gemini-2.5-pro)",
    help="選擇用於技術分析的 AI 模型"
)

if ai_provider == "gemini":
    ai_api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="請輸入您的 Google Gemini API 金鑰",
        key="gemini_api_key"
    )
else:
    ai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="請輸入您的 OpenAI API 金鑰",
        key="openai_api_key"
    )

# RSI參數設定（新增）
st.sidebar.markdown("### 📊 RSI指標設定")
rsi_period = st.sidebar.slider(
    "RSI計算週期",
    min_value=5,
    max_value=30,
    value=14,
    help="RSI相對強弱指標的計算週期，標準為14天"
)

# 日期選擇
st.sidebar.markdown("### 📅 日期設定")
default_start_date = datetime.now() - timedelta(days=90)
default_end_date = datetime.now()

start_date = st.sidebar.date_input(
    "起始日期",
    value=default_start_date,
    help="選擇分析的起始日期"
)

end_date = st.sidebar.date_input(
    "結束日期",
    value=default_end_date,
    help="選擇分析的結束日期"
)

# 分析按鈕
analyze_button = st.sidebar.button("🚀 開始分析", type="primary")

# 免責聲明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")

# 主要分析邏輯
if analyze_button:
    # 輸入驗證
    if not symbol.strip():
        st.error("請輸入股票代碼")
    elif start_date >= end_date:
        st.error("起始日期不能晚於或等於結束日期")
    else:
        # 開始分析流程
        with st.spinner(f"正在獲取{stock_type}數據..."):
            # 根據股票類型獲取數據
            if is_tw_stock:
                stock_data = get_taiwan_stock_data(symbol, api_key, start_date, end_date)
            else:
                stock_data = get_stock_data(symbol.upper(), api_key, start_date, end_date)

            if stock_data is not None and len(stock_data) > 0:
                st.success(f"成功獲取 {len(stock_data)} 筆交易數據")

                # 過濾數據
                filtered_data = filter_by_date_range(stock_data, start_date, end_date)

                if filtered_data is not None and len(filtered_data) > 0:
                    # 計算移動平均線
                    with st.spinner("正在計算技術指標..."):
                        data_with_ma = get_moving_averages(filtered_data)

                    # 計算RSI指標（新增）
                    with st.spinner("正在計算RSI指標..."):
                        data_with_indicators = calculate_rsi(data_with_ma, period=rsi_period)

                    if data_with_indicators is not None:
                        # 顯示綜合技術分析圖表（包含RSI）
                        st.markdown(f"### 📊 {stock_type}股價K線圖與技術指標（含RSI）")
                        display_symbol = symbol if is_tw_stock else symbol.upper()
                        chart = create_enhanced_chart(data_with_indicators, display_symbol, rsi_period, is_tw_stock)
                        st.plotly_chart(chart, use_container_width=True)

                        # 基本統計資訊
                        st.markdown(f"### 📈 基本統計資訊 ({stock_type})")
                        col1, col2, col3, col4 = st.columns(4)

                        start_price = data_with_indicators['close'].iloc[0]
                        end_price = data_with_indicators['close'].iloc[-1]
                        price_change = end_price - start_price
                        price_change_pct = (price_change / start_price) * 100
                        current_rsi = data_with_indicators['rsi'].iloc[-1]
                        rsi_signal, rsi_icon = get_rsi_signal(current_rsi)

                        currency_symbol = "NT$" if is_tw_stock else "$"

                        with col1:
                            st.metric(
                                "起始價格",
                                f"{currency_symbol}{start_price:.2f}",
                                help="分析期間第一個交易日的收盤價"
                            )

                        with col2:
                            st.metric(
                                "結束價格",
                                f"{currency_symbol}{end_price:.2f}",
                                help="分析期間最後一個交易日的收盤價"
                            )

                        with col3:
                            st.metric(
                                "價格變化",
                                f"{currency_symbol}{price_change:.2f}",
                                f"{price_change_pct:.2f}%",
                                help="期間內的價格變化金額和百分比"
                            )

                        # RSI狀態顯示（新增）
                        with col4:
                            st.metric(
                                f"RSI({rsi_period})",
                                f"{current_rsi:.2f}",
                                f"{rsi_icon} {rsi_signal}",
                                help=f"相對強弱指標：超買>70，超賣<30"
                            )

                        # RSI狀態警告（新增）
                        if current_rsi >= 70:
                            st.warning(f"🔴 RSI警告：當前RSI值為 {current_rsi:.2f}，處於超買狀態（>70），歷史上此狀態可能伴隨價格回調風險。")
                        elif current_rsi <= 30:
                            st.info(f"🟢 RSI提示：當前RSI值為 {current_rsi:.2f}，處於超賣狀態（<30），歷史上此狀態可能出現反彈機會。")
                        else:
                            st.success(f"🟡 RSI狀態：當前RSI值為 {current_rsi:.2f}，處於正常區間（30-70），技術面相對平衡。")

                        # AI技術分析（僅在有 AI API Key 時執行）
                        if ai_api_key and ai_api_key.strip():
                            provider_name = "Google Gemini" if ai_provider == "gemini" else "OpenAI"
                            st.markdown(f"### 🤖 AI技術分析（{provider_name}）- {stock_type}")
                            with st.spinner(f"{provider_name} AI 正在分析中..."):
                                ai_analysis = generate_ai_insights(
                                    display_symbol,
                                    data_with_indicators,
                                    ai_api_key,
                                    ai_provider,
                                    start_date,
                                    end_date
                                )

                            if ai_analysis:
                                st.markdown(ai_analysis)
                        else:
                            provider_name = "Google Gemini API Key" if ai_provider == "gemini" else "OpenAI API Key"
                            st.info(f"💡 提示：輸入 {provider_name} 可獲得 AI 技術分析報告")

                        # 歷史數據表格
                        st.markdown("### 📋 歷史數據表格（含RSI指標）")
                        # 顯示最近10筆數據
                        display_data = data_with_indicators.tail(10).copy()
                        display_data = display_data.sort_values('date', ascending=False)

                        # 格式化數據（包含RSI）
                        display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60', 'rsi']
                        display_data_formatted = display_data[display_columns].copy()

                        # 重命名欄位
                        display_data_formatted.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60', f'RSI({rsi_period})']

                        # 數值格式化
                        for col in ['開盤', '最高', '最低', '收盤', 'MA5', 'MA10', 'MA20', 'MA60']:
                            display_data_formatted[col] = display_data_formatted[col].round(2)
                        display_data_formatted[f'RSI({rsi_period})'] = display_data_formatted[f'RSI({rsi_period})'].round(2)

                        st.dataframe(
                            display_data_formatted,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.success(f"✅ {stock_type}分析完成！包含RSI技術指標分析")

                else:
                    st.warning("所選日期範圍內沒有交易數據，請調整日期範圍。")
            else:
                st.error(f"無法獲取{stock_type}數據，請檢查股票代碼和API金鑰。")

# 初始頁面說明
if not analyze_button:
    st.markdown("""
    ## 歡迎使用 AI 股票趨勢分析系統 (美股與台股) 👋

    ### 🚀 功能特色
    - **雙市場支援**: 同時支援美股與台股分析 🆕
    - **專業K線圖表**: 互動式價格圖表，包含移動平均線技術指標
    - **RSI相對強弱指標**: 新增RSI技術指標，分析超買超賣狀態
    - **AI智能分析**: 使用先進AI模型進行深度技術面分析（含RSI解讀）
    - **歷史數據**: 詳細的股票歷史價格和成交量數據
    - **教育導向**: 客觀的技術分析，僅供學習研究使用

    ### 📝 使用方法
    1. 在左側輸入股票代碼：
       - **台股**：輸入數字代碼（如：2330, 2317）🇹🇼
       - **美股**：輸入英文代碼（如：TSLA, MSFT, GOOGL）🇺🇸
    2. 系統會自動偵測股票類型並顯示對應的 API Key 輸入框
    3. 輸入對應的 API 金鑰：
       - **台股**：FindMind API Key（可選填）
       - **美股**：FMP API Key（必填）
    4. 選擇 AI 提供商（預設：OpenAI）
    5. 輸入對應的 AI API Key（用於 AI 分析，可選填）
    6. 調整 RSI 計算週期（預設 14 天）
    7. 選擇分析的日期範圍
    8. 點擊「開始分析」按鈕

    ### 💡 技術指標說明
    - **MA5**: 5日移動平均線，短期趨勢指標
    - **MA10**: 10日移動平均線，短中期趨勢指標
    - **MA20**: 20日移動平均線，中期趨勢指標
    - **MA60**: 60日移動平均線，長期趨勢指標
    - **RSI**: 相對強弱指標，分析超買超賣狀態 🆕
        - RSI > 70：超買狀態，可能面臨回調壓力
        - RSI < 30：超賣狀態，可能出現反彈機會
        - RSI 30-70：正常區間，技術面相對平衡

    ### 🔍 RSI指標詳解 🆕
    **RSI（Relative Strength Index）相對強弱指標**是由技術分析師J. Welles Wilder開發的動量振盪器，用於：
    - **測量價格變動的速度和幅度**：RSI在0-100之間波動
    - **識別超買超賣條件**：幫助判斷股票是否被過度買入或賣出
    - **動量分析**：評估價格上漲或下跌的力道強弱
    - **背離信號**：當價格與RSI走勢出現背離時，可能預示趨勢轉變

    **計算公式**：RSI = 100 - (100 / (1 + RS))
    其中 RS = 平均漲幅 / 平均跌幅（通常使用14日期間）

    ### 🔑 API金鑰獲取
    - **台股 FindMind API**: 前往 [FinMind](https://finmindtrade.com/) 註冊 🇹🇼
    - **美股 FMP API**: 前往 [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs) 註冊 🇺🇸
    - **AI 分析**（擇一使用）：
      - **OpenAI API**: 前往 [OpenAI Platform](https://platform.openai.com) 註冊
        - 模型：GPT-4o-mini（預設）
        - 穩定可靠，回應品質一致
      - **Google Gemini API**: 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 取得
        - 模型：gemini-2.5-pro（高階推理模型）
        - 免費額度高，適合新手
        - 推理能力強，分析深入

    ### 🎯 範例
    - **台股範例**：2330（台積電）、2317（鴻海）、2454（聯發科）
    - **美股範例**：TSLA（蘋果）、MSFT（微軟）、GOOGL（Google）、TSLA（特斯拉）

    ---
    **開始您的技術分析之旅吧！** 📈
    """)