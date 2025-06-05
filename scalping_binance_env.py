import gymnasium as gym
import numpy as np
import pandas as pd
import ccxt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import time

from scalping_config import (
    ENV_VERSION, SYMBOL, ALL_INDICATORS, REQUIRED_INDICATORS,
    TRADING_FEE, MIN_PROFIT_THRESHOLD
)

FIXED_INDICATORS = [
    'close', 'volume', 'ema', 'rsi', 'macd', 'macd_signal', 
    'bb_mavg', 'bb_high', 'bb_low', 'vwap', 'atr', 'obv'
]

NUM_INDICATORS = 12

class IndicatorCache:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.last_data_hash = None
        self.cached_df = None
        self.last_update_time = 0
        
    def get_cached_data(self, df):
        if self.cached_df is None:
            return None
        current_hash = hash(str(df.shape) + str(df.iloc[-1].values.tolist() if len(df) > 0 else "empty"))
        if current_hash == self.last_data_hash:
            return self.cached_df
        return None
        
    def update_cache(self, df):
        self.last_data_hash = hash(str(df.shape) + str(df.iloc[-1].values.tolist() if len(df) > 0 else "empty"))
        self.cached_df = df.copy()
        self.last_update_time = time.time()


class ScalpingBinanceEnv(gym.Env):
    def __init__(self, min_profit_threshold=None):
        super(ScalpingBinanceEnv, self).__init__()
        self.client = ccxt.binance()
        self.symbol = SYMBOL
        self.timeframe = '5m'
        self.indicator_cache = IndicatorCache()
        self.min_profit_threshold = min_profit_threshold if min_profit_threshold is not None else MIN_PROFIT_THRESHOLD
        self.indicators = FIXED_INDICATORS
        self.data = self.load_data()
        self.action_space = gym.spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(NUM_INDICATORS,),
            dtype=np.float32
        )
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        actual_dim = len(self.data.columns)
        if actual_dim != NUM_INDICATORS:
            raise ValueError(f"Dimensão incorreta: {actual_dim} (esperado {NUM_INDICATORS})")

    def calculate_indicators(self, df):
        cached_df = self.indicator_cache.get_cached_data(df)
        if cached_df is not None:
            return cached_df
        if df is None or len(df) == 0:
            raise ValueError("DataFrame vazio ou None")
        required_base_columns = ['close', 'volume']
        missing_base = [col for col in required_base_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"DataFrame não contém colunas básicas necessárias: {missing_base}")
        result_df = df.copy()
        try:
            if 'ema' not in result_df.columns:
                result_df['ema'] = EMAIndicator(close=result_df['close'], window=9).ema_indicator()
            if 'rsi' not in result_df.columns:
                result_df['rsi'] = RSIIndicator(close=result_df['close'], window=14).rsi()
            if 'macd' not in result_df.columns or 'macd_signal' not in result_df.columns:
                macd = MACD(close=result_df['close'])
                result_df['macd'] = macd.macd()
                result_df['macd_signal'] = macd.macd_signal()
            if 'bb_mavg' not in result_df.columns or 'bb_high' not in result_df.columns or 'bb_low' not in result_df.columns:
                bb = BollingerBands(close=result_df['close'], window=20, window_dev=2)
                result_df['bb_mavg'] = bb.bollinger_mavg()
                result_df['bb_high'] = bb.bollinger_hband()
                result_df['bb_low'] = bb.bollinger_lband()
            if 'vwap' not in result_df.columns:
                if 'high' in result_df.columns and 'low' in result_df.columns:
                    vwap = VolumeWeightedAveragePrice(
                        high=result_df['high'], low=result_df['low'], 
                        close=result_df['close'], volume=result_df['volume']
                    )
                    result_df['vwap'] = vwap.volume_weighted_average_price()
                else:
                    result_df['vwap'] = result_df['close']
            if 'atr' not in result_df.columns:
                if 'high' in result_df.columns and 'low' in result_df.columns:
                    atr = AverageTrueRange(high=result_df['high'], low=result_df['low'], close=result_df['close'], window=14)
                    result_df['atr'] = atr.average_true_range()
                else:
                    result_df['atr'] = result_df['close'].rolling(window=14).std()
            if 'obv' not in result_df.columns:
                obv = OnBalanceVolumeIndicator(close=result_df['close'], volume=result_df['volume'])
                result_df['obv'] = obv.on_balance_volume()
        except Exception:
            pass
        missing_indicators = set(FIXED_INDICATORS) - set(result_df.columns)
        for indicator in missing_indicators:
            result_df[indicator] = 0.0
        extra_indicators = set(result_df.columns) - set(FIXED_INDICATORS)
        if extra_indicators:
            result_df = result_df.drop(columns=list(extra_indicators))
        result_df = result_df[FIXED_INDICATORS]
        if result_df.isnull().any().any():
            result_df = result_df.bfill().ffill().fillna(0)
        self.indicator_cache.update_cache(result_df)
        return result_df

    def load_data(self):
        ohlcv_5m = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=2000)
        df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_5m = self.calculate_indicators(df_5m)
        df_5m.reset_index(drop=True, inplace=True)
        return df_5m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        if not hasattr(self, 'data') or self.data is None or len(self.data) == 0:
            raise ValueError("DataFrame não inicializado ou vazio")
        missing_columns = set(FIXED_INDICATORS) - set(self.data.columns)
        if missing_columns:
            self.data = self.calculate_indicators(self.data)
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        obs = self.data.iloc[self.current_step][FIXED_INDICATORS].values.astype(np.float32)
        if len(obs) != NUM_INDICATORS:
            temp_df = pd.DataFrame([obs], columns=self.data.columns)
            temp_df = self.calculate_indicators(temp_df)
            obs = temp_df.iloc[0].values.astype(np.float32)
        return obs

    def set_external_data(self, df):
        self.indicator_cache.reset()
        df_with_indicators = self.calculate_indicators(df)
        self.data = df_with_indicators
        return True

    def step(self, action):
        reward = 0
        price = self.data.iloc[self.current_step]['close']
        info = {}

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price * (1 + TRADING_FEE)
            info['entry_fee'] = price * TRADING_FEE
            
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price * (1 - TRADING_FEE)
            info['entry_fee'] = price * TRADING_FEE
            
        elif action == 0 and self.position != 0:
            exit_fee = price * TRADING_FEE
            
            if self.position == 1:
                effective_exit_price = price * (1 - TRADING_FEE)
                base_reward = effective_exit_price - self.entry_price
                profit_percentage = base_reward / self.entry_price if self.entry_price != 0 else 0
                
                if base_reward > 0:
                    if profit_percentage >= self.min_profit_threshold:
                        reward = base_reward * 1.5
                    else:
                        reward = base_reward * 0.5
                else:
                    reward = base_reward
                
            elif self.position == -1:
                effective_exit_price = price * (1 + TRADING_FEE)
                base_reward = self.entry_price - effective_exit_price
                profit_percentage = base_reward / self.entry_price if self.entry_price != 0 else 0
                
                if base_reward > 0:
                    if profit_percentage >= self.min_profit_threshold:
                        reward = base_reward * 1.5
                    else:
                        reward = base_reward * 0.5
                else:
                    reward = base_reward
            
            info['exit_fee'] = exit_fee
            info['base_reward'] = base_reward
            info['adjusted_reward'] = reward
            info['profit_percentage'] = profit_percentage
            info['profit_threshold'] = self.min_profit_threshold
            info['profit_threshold_met'] = profit_percentage >= self.min_profit_threshold
            
            self.position = 0
            self.entry_price = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        pass
