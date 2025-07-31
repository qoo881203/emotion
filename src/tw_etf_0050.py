import requests
import pandas as pd
import yfinance as yf



def fetch_0050_components() -> list:
    """Fetch component stock codes for ETF 0050 from Goodinfo."""
    url = "https://goodinfo.tw/tw/StockIdxDetail.asp?STOCK_ID=0050"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": url,
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = "utf-8"
    tables = pd.read_html(resp.text)
    component_table = None
    for table in tables:
        cols = list(table.columns)
        if len(cols) > 0 and ("代號" in str(cols[0]) or "股票" in str(cols[0])):
            component_table = table
            break
    if component_table is None:
        raise RuntimeError("Failed to locate component stock table")
    codes = component_table.iloc[:, 0].astype(str).str.zfill(4).tolist()
    return codes



def fetch_history(symbol: str, period: str = "30d") -> pd.DataFrame:
    """Download historical daily price for given stock symbol."""
    ticker = f"{symbol}.TW"
    df = yf.download(ticker, period=period)
    return df



def add_indicators(df: pd.DataFrame, ma_window: int = 5, rsi_period: int = 14) -> pd.DataFrame:
    """Append MA and RSI indicators to the dataframe."""
    df = df.copy()
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df[f"RSI{rsi_period}"] = 100 - (100 / (1 + rs))
    return df



def main():
    codes = fetch_0050_components()
    result = {}
    for code in codes:
        hist = fetch_history(code)
        hist = add_indicators(hist)
        result[code] = hist.tail(1)[["Close", "MA5", "RSI14"]].iloc[0].to_dict()
    print(pd.DataFrame.from_dict(result, orient="index"))


if __name__ == "__main__":
    main()
