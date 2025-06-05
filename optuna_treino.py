import pandas as pd
import numpy as np
import optuna
import ta

# === Função de backtest vetorizado ===
def vectorized_backtest(data, buy_entry, sell_entry, buy_exit, sell_exit, min_return=0.004):
    position = 0  # 1 = comprado, -1 = vendido, 0 = neutro
    entry_price = 0.0
    net_profit = 0.0
    for i in range(1, len(data)):
        price = data['close'].iloc[i]

        if position == 0:
            if buy_entry.iloc[i]:
                position = 1
                entry_price = price
            elif sell_entry.iloc[i]:
                position = -1
                entry_price = price

        elif position == 1:
            if buy_exit.iloc[i] or (price - entry_price) / entry_price >= min_return:
                net_profit += (price - entry_price) / entry_price
                position = 0

        elif position == -1:
            if sell_exit.iloc[i] or (entry_price - price) / entry_price >= min_return:
                net_profit += (entry_price - price) / entry_price
                position = 0

    return {"net_profit": net_profit}

# === Função objetivo ===
def objective(trial):
    df = pd.read_csv("XRPUSDT_5m.csv")
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

    # Parâmetros a otimizar
    rsi_period = trial.suggest_int("rsi_period", 5, 20)
    rsi_oversold = trial.suggest_int("rsi_oversold", 20, 40)
    rsi_overbought = trial.suggest_int("rsi_overbought", 60, 90)
    rsi_exit_long = trial.suggest_int("rsi_exit_long", 60, 90)
    rsi_exit_short = trial.suggest_int("rsi_exit_short", 10, 40)
    ema_fast = trial.suggest_int("ema_fast", 5, 20)
    ema_slow = trial.suggest_int("ema_slow", 21, 50)
    macd_fast = trial.suggest_int("macd_fast", 8, 15)
    macd_slow = trial.suggest_int("macd_slow", 20, 30)
    macd_signal = trial.suggest_int("macd_signal", 5, 10)
    bb_window = trial.suggest_int("bb_window", 10, 30)
    volume_ma = trial.suggest_int("volume_ma", 5, 20)

    # Indicadores técnicos
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], ema_fast).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], ema_slow).ema_indicator()
    macd = ta.trend.MACD(df['close'], macd_fast, macd_slow, macd_signal)
    df['macd_diff'] = macd.macd_diff()
    df['bb_high'] = ta.volatility.BollingerBands(df['close'], bb_window).bollinger_hband()
    df['bb_low'] = ta.volatility.BollingerBands(df['close'], bb_window).bollinger_lband()
    df['volume_ma'] = df['volume'].rolling(volume_ma).mean()

    df = df.dropna()

    # Regras de entrada
    buy_entry = (
        (df['rsi'] < rsi_oversold) &
        (df['ema_fast'] > df['ema_slow']) &
        (df['macd_diff'] > 0) &
        (df['close'] < df['bb_low']) &
        (df['volume'] > df['volume_ma'])
    )

    sell_entry = (
        (df['rsi'] > rsi_overbought) &
        (df['ema_fast'] < df['ema_slow']) &
        (df['macd_diff'] < 0) &
        (df['close'] > df['bb_high']) &
        (df['volume'] > df['volume_ma'])
    )

    # Regras de saída
    buy_exit = (
        (df['rsi'] > rsi_exit_long) |
        (df['ema_fast'] < df['ema_slow']) |
        (df['macd_diff'] < 0)
    )

    sell_exit = (
        (df['rsi'] < rsi_exit_short) |
        (df['ema_fast'] > df['ema_slow']) |
        (df['macd_diff'] > 0)
    )

    result = vectorized_backtest(
        df, buy_entry, sell_entry, buy_exit, sell_exit, min_return=0.004
    )

    return result["net_profit"]

# === Executar script ===
if __name__ == "__main__":
    print("Iniciando otimização com Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("\nMelhores parâmetros encontrados:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print("Lucro líquido:", study.best_value)

    results_df = pd.DataFrame([study.best_params])
    results_df["net_profit"] = study.best_value
    results_df.to_csv("melhores_parametros_scalping.csv", index=False)
    print("Parâmetros salvos em 'melhores_parametros_scalping.csv'.")
