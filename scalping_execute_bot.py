"""
Execução do bot de trading usando o modelo treinado.
Conecta-se à Binance Testnet Futures via WebSockets para realizar operações de trading em tempo real.
Versão final otimizada usando a biblioteca oficial binance-futures-connector.
VERSÃO CORRIGIDA - Calcula todos os 12 indicadores técnicos necessários.
"""

import numpy as np
import pandas as pd
import os
import time
import logging
import traceback
import json
from datetime import datetime
from threading import Thread
from queue import Queue
from collections import deque

# Bibliotecas para modelo de IA
from stable_baselines3 import PPO

# Biblioteca oficial da Binance para Futures
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.error import ClientError

# Bibliotecas para indicadores técnicos
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# Importar ambiente e indicadores
from scalping_binance_env import ScalpingBinanceEnv
import warnings

# Importar configurações centralizadas
from scalping_config import (
    ENV_VERSION, FORCE_DIM, SYMBOL, TRADE_AMOUNT, LEVERAGE,
    MODEL_DIR, CONFIG_DIR, LOG_DIR, MODEL_PATH, INDICATORS_CONFIG_PATH,
    TRADING_FEE, MIN_PROFIT_THRESHOLD,
    check_version_compatibility
)

# Importar logger centralizado
from scalping_logging import setup_logger

# Configurar logger
logger = setup_logger("scalping_execute_ws")

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Verificar compatibilidade de versão
if not check_version_compatibility(ENV_VERSION):
    logger.error(f"Versão do ambiente incompatível. Esperado: {ENV_VERSION}")
    raise ValueError(f"Versão do ambiente incompatível")

# Configurações de execução
SIMULATION_MODE = False  # FORÇADO para modo real na Testnet

# Credenciais da API Binance Testnet (atualizadas)
# Se não quiser definir aqui, configure como variáveis de ambiente:
# export BINANCE_API_KEY="sua_api_key_da_testnet"
# export BINANCE_API_SECRET="sua_api_secret_da_testnet"
BINANCE_API_KEY = "96690166dc18d7a6783ae185e79e9403ecde1221977a436a70983b70e7dc34ec"
BINANCE_API_SECRET = "3263a45cb82b1138c13a39fe72a91a91f877a7d774cb87920b05a19547d65abe"

# Configurações específicas para WebSocket
KLINE_INTERVAL = '5m'  # Intervalo de tempo para os candles
WEBSOCKET_TIMEOUT = 60  # Timeout para reconexão do WebSocket em segundos
DATA_BUFFER_SIZE = 1000  # Número de candles a manter no buffer
TRADE_UPDATE_INTERVAL = 1  # Intervalo em segundos para verificar sinais de trading

# URLs específicas para Testnet Futures - CORRIGIDAS
TESTNET_FUTURES_BASE_URL = 'https://testnet.binancefuture.com'


class FuturesManager:
    """
    Gerenciador de conexões com a Binance Futures Testnet usando a biblioteca oficial.
    """
    def __init__(self, api_key, api_secret, symbol):
        """
        Inicializa o gerenciador para Futures.
        
        Args:
            api_key: Chave de API da Binance
            api_secret: Chave secreta da Binance
            symbol: Símbolo de trading (ex: 'ETHUSDT')
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol.replace('/', '')  # Remover '/' se presente
        self.client = None
        self.ws_client = None
        self.last_kline = None
        self.klines_buffer = deque(maxlen=DATA_BUFFER_SIZE)
        self.df = None
        self.ws_connected = False
        self.callbacks_registered = False
        
        # Inicializar conexão
        self._initialize_client()
        
    def _initialize_client(self):
        """
        Inicializa o cliente da Binance Futures Testnet usando a biblioteca oficial.
        """
        try:
            # Configurar cliente específico para Futures Testnet
            self.client = UMFutures(
                key=self.api_key,
                secret=self.api_secret,
                base_url=TESTNET_FUTURES_BASE_URL
            )
            
            logger.info(f"Cliente Binance Futures Testnet inicializado com base_url: {TESTNET_FUTURES_BASE_URL}")
            
            # Verificar conexão
            try:
                server_time = self.client.time()
                logger.info(f"Conexão com Binance Futures Testnet estabelecida. Hora do servidor: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            except ClientError as e:
                logger.error(f"Erro ao verificar hora do servidor Futures: {e}")
                raise ConnectionError(f"Falha na conexão com Binance Futures Testnet: {e}")
            
            # Verificar informações da conta
            try:
                account_info = self.client.account()
                logger.info(f"Conta Futures Testnet acessada com sucesso. Saldo disponível: {account_info.get('availableBalance', 'N/A')} USDT")
            except ClientError as e:
                logger.error(f"Erro ao obter informações da conta Futures Testnet: {e}")
                logger.error("Verifique se sua API key tem permissões para Futures e foi gerada na Testnet Futures")
                raise ConnectionError(f"Falha ao acessar conta Futures Testnet: {e}")
            
            # Configurar alavancagem
            try:
                leverage_info = self.client.change_leverage(
                    symbol=self.symbol,
                    leverage=LEVERAGE
                )
                logger.info(f"Alavancagem configurada: {leverage_info.get('leverage', LEVERAGE)}x")
            except ClientError as e:
                logger.error(f"Erro ao configurar alavancagem: {e}")
                logger.error("Verifique se sua API key tem permissões para Futures e foi gerada na Testnet Futures")
                raise ConnectionError(f"Falha ao configurar alavancagem: {e}")
            
            # Inicializar cliente WebSocket
            self.ws_client = UMFuturesWebsocketClient()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Binance Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            raise ConnectionError(f"Falha na conexão com Binance Futures Testnet: {e}")
    
    def start_kline_socket(self, interval=KLINE_INTERVAL):
        """
        Inicia o socket para receber klines (candles).
        
        Args:
            interval: Intervalo de tempo para os candles
        """
        try:
            # Carregar dados históricos primeiro
            self._load_historical_data(interval)
            
            # Registrar callbacks para WebSocket
            if not self.callbacks_registered:
                # CORREÇÃO: Não chamar start(), apenas registrar callbacks diretamente
                # A biblioteca binance-futures-connector inicia automaticamente a conexão
                
                # Registrar callbacks
                self.ws_client.kline(
                    symbol=self.symbol.lower(),
                    id=1,
                    interval=interval,
                    callback=self._handle_kline_message
                )
                
                self.callbacks_registered = True
                self.ws_connected = True
                
                logger.info(f"Socket de klines Futures Testnet iniciado para {self.symbol} ({interval})")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar socket de klines Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            raise ConnectionError(f"Falha ao iniciar WebSocket Futures Testnet: {e}")
    
    def _handle_kline_message(self, message):
        """
        Manipula mensagens de kline recebidas via WebSocket.
        
        Args:
            message: Mensagem recebida
        """
        try:
            # Verificar se é uma mensagem de kline
            if 'k' not in message:
                return
            
            kline = message['k']
            
            # Verificar se o candle está fechado
            is_closed = kline['x']
            
            # Criar dicionário com dados do candle
            candle_data = {
                'timestamp': kline['t'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closed': is_closed
            }
            
            # Atualizar último kline
            self.last_kline = candle_data
            
            # Se o candle estiver fechado, adicionar ao buffer
            if is_closed:
                self.klines_buffer.append(candle_data)
                logger.debug(f"Candle fechado: {datetime.fromtimestamp(candle_data['timestamp']/1000)} - Close: {candle_data['close']}")
                
                # Atualizar DataFrame
                self._update_dataframe()
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem de kline Futures Testnet: {e}")
            logger.error(f"Mensagem: {message}")
            logger.error(traceback.format_exc())
    
    def _load_historical_data(self, interval):
        """
        Carrega dados históricos de klines para Futures Testnet.
        
        Args:
            interval: Intervalo de tempo para os candles
        """
        try:
            logger.info(f"Carregando dados históricos Futures Testnet para {self.symbol} ({interval})...")
            
            # Obter klines históricos para Futures
            klines = self.client.klines(
                symbol=self.symbol,
                interval=interval,
                limit=DATA_BUFFER_SIZE
            )
            logger.info("Dados históricos obtidos via klines")
            
            # Processar klines
            for k in klines:
                candle_data = {
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'closed': True
                }
                self.klines_buffer.append(candle_data)
            
            logger.info(f"Dados históricos Futures Testnet carregados: {len(self.klines_buffer)} candles")
            
            # Criar DataFrame inicial
            self._update_dataframe()
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados históricos Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            raise ConnectionError(f"Falha ao carregar dados históricos Futures Testnet: {e}")
    
    def _update_dataframe(self):
        """
        Atualiza o DataFrame com os dados do buffer.
        """
        try:
            # Converter buffer para DataFrame
            df = pd.DataFrame(list(self.klines_buffer))
            
            if len(df) > 0:
                # Converter timestamp para datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Ordenar por timestamp
                df.sort_index(inplace=True)
                
                # Atualizar DataFrame
                self.df = df
                logger.debug(f"DataFrame Futures Testnet atualizado: {len(df)} registros")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar DataFrame Futures Testnet: {e}")
            logger.error(traceback.format_exc())
    
    def get_current_dataframe(self):
        """
        Retorna o DataFrame atual com os dados de klines.
        
        Returns:
            pd.DataFrame: DataFrame com os dados de klines
        """
        return self.df.copy() if self.df is not None else None
    
    def get_current_price(self):
        """
        Retorna o preço atual do ativo no mercado Futures Testnet.
        
        Returns:
            float: Preço atual
        """
        if self.last_kline is not None:
            return float(self.last_kline['close'])
        
        # Fallback: obter preço via API REST Futures
        try:
            ticker = self.client.ticker_price(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao obter preço atual Futures Testnet: {e}")
            return None
    
    def stop(self):
        """
        Para o WebSocket.
        """
        if not self.ws_connected:
            return
        
        try:
            if self.ws_client:
                # CORREÇÃO: Usar o método correto para parar o WebSocket
                self.ws_client.stop()
            
            self.ws_connected = False
            logger.info("WebSocket Futures Testnet parado")
            
        except Exception as e:
            logger.error(f"Erro ao parar WebSocket Futures Testnet: {e}")
            logger.error(traceback.format_exc())
    
    def create_order(self, side, quantity):
        """
        Cria uma ordem no mercado Futures Testnet.
        
        Args:
            side: Lado da ordem (BUY ou SELL)
            quantity: Quantidade a ser negociada
            
        Returns:
            dict: Informações da ordem criada
        """
        try:
            logger.info(f"Criando ordem Futures Testnet: {side} {quantity} {self.symbol}")
            
            # Criar ordem via API Futures
            order = self.client.new_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            
            logger.info(f"Ordem Futures Testnet criada: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Erro ao criar ordem Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            raise


class ScalpingBotFuturesOfficial:
    """
    Bot de trading que utiliza a biblioteca oficial da Binance para obter dados em tempo real do mercado Futures Testnet.
    VERSÃO CORRIGIDA - Calcula todos os 12 indicadores técnicos necessários.
    """
    
    def __init__(self, api_key=None, api_secret=None, simulation_mode=SIMULATION_MODE):
        """
        Inicializa o bot de trading.
        
        Args:
            api_key: Chave de API da Binance
            api_secret: Chave secreta da Binance
            simulation_mode: Se True, executa em modo de simulação
        """
        self.simulation_mode = simulation_mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = SYMBOL
        self.trade_amount = TRADE_AMOUNT
        self.leverage = LEVERAGE
        
        # Verificar e obter credenciais
        self._get_credentials()
        
        # Inicializar Futures Manager
        self.futures_manager = FuturesManager(
            api_key=self.api_key,
            api_secret=self.api_secret,
            symbol=self.symbol
        )
        
        # Carregar modelo e indicadores
        self.model = self._load_model()
        self.indicators = self._load_indicators()
        
        # Criar ambiente
        self.env = self._create_environment()
        
        # Inicializar estado
        self.position = 0  # 0 = neutro, 1 = comprado, -1 = vendido
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.total_gain = 0
        self.total_loss = 0
        self.current_step = 0
        self.entry_step = 0
        
        # Iniciar WebSocket
        self.futures_manager.start_kline_socket()
        
        logger.info("Bot de trading Futures Testnet com WebSocket oficial inicializado com sucesso")
    
    def _get_credentials(self):
        """
        Obtém as credenciais da API da Binance.
        """
        if not self.api_key or not self.api_secret:
            logger.warning("API Key e Secret não encontrados. Tentando usar variáveis de ambiente.")
            self.api_key = os.environ.get('BINANCE_API_KEY', BINANCE_API_KEY)
            self.api_secret = os.environ.get('BINANCE_API_SECRET', BINANCE_API_SECRET)
            
            if not self.api_key or not self.api_secret:
                logger.error("API Key e Secret são necessários para modo real na Testnet")
                raise ValueError("API Key e Secret são necessários para modo real na Testnet")
    
    def _load_model(self):
        """
        Carrega o modelo treinado.

        Returns:
            PPO: Modelo carregado
        """
        # Garante que MODEL_PATH não termine com .zip antes de usar
        model_base_path = MODEL_PATH
        if model_base_path.endswith(".zip"):
             model_base_path = model_base_path[:-4] # Remove .zip se existir

        logger.info(f"Carregando modelo de {model_base_path}")

        # Caminho esperado do arquivo (com .zip)
        model_file_path = f"{model_base_path}.zip"

        if not os.path.exists(model_file_path):
            logger.error(f"Modelo não encontrado em {model_file_path}")
            # Levanta o erro com o caminho esperado (com .zip)
            raise FileNotFoundError(f"Modelo não encontrado: {model_file_path}")

        # Carrega usando o caminho base (sem .zip), SB3 adiciona automaticamente
        model = PPO.load(model_base_path)
        logger.info("Modelo carregado com sucesso")

        return model
    
    def _load_indicators(self):
        """
        Carrega os indicadores utilizados no treinamento.
        
        Returns:
            list: Lista de indicadores
        """
        logger.info(f"Carregando indicadores de {INDICATORS_CONFIG_PATH}")
        
        if not os.path.exists(INDICATORS_CONFIG_PATH):
            logger.error(f"Arquivo de indicadores não encontrado: {INDICATORS_CONFIG_PATH}")
            raise FileNotFoundError(f"Arquivo de indicadores não encontrado: {INDICATORS_CONFIG_PATH}")
        
        with open(INDICATORS_CONFIG_PATH, "r") as f:
            indicators = eval(f.read().strip())
        
        logger.info(f"Indicadores carregados: {indicators}")
        return indicators
    
    def _create_environment(self):
        """
        Cria o ambiente de trading com os indicadores carregados.
        
        Returns:
            ScalpingBinanceEnv: Ambiente de trading
        """
        logger.info("Criando ambiente com indicadores carregados")
        env = ScalpingBinanceEnv(indicators=self.indicators)
        return env
    
    def _verify_dimensions(self, observation):
        """
        Verifica se as dimensões da observação e do modelo são compatíveis.
        
        Args:
            observation: Observação a ser verificada
        """
        expected_shape = self.model.observation_space.shape[0]
        
        if observation.shape[0] != expected_shape:
            logger.error(f"Incompatibilidade de dimensões! Modelo espera {expected_shape}, ambiente fornece {observation.shape[0]}")
            raise ValueError(f"Incompatibilidade de dimensões entre modelo e ambiente")
        
        logger.debug(f"Dimensões compatíveis: {observation.shape[0]}")
    
    def _prepare_observation(self, df):
        """
        Prepara a observação para o modelo, calculando todos os 12 indicadores necessários.
        VERSÃO CORRIGIDA: Calcula todos os indicadores no DataFrame do WebSocket.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV do WebSocket
            
        Returns:
            np.ndarray: Array com 12 features para o modelo
        """
        try:
            logger.info(f"Preparando observação com DataFrame de shape {df.shape}")
            logger.info(f"Colunas disponíveis: {list(df.columns)}")
            
            # Verificar se temos dados suficientes
            if len(df) < 50:  # Precisamos de pelo menos 50 candles para calcular indicadores
                logger.warning(f"Dados insuficientes: {len(df)} candles. Mínimo: 50")
                # Usar dados do ambiente se não temos dados suficientes
                return self.env._get_obs()
            
            # Fazer uma cópia do DataFrame para não modificar o original
            df_work = df.copy()
            
            # Certifique-se de que temos as colunas OHLCV básicas
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_work.columns:
                    raise ValueError(f"Coluna {col} não encontrada no DataFrame")
            
            logger.info("Calculando indicadores técnicos para observação...")
            
            # 1. EMA (Exponential Moving Average)
            logger.debug("Calculando EMA...")
            try:
                ema_indicator = EMAIndicator(close=df_work['close'], window=9)
                df_work['ema'] = ema_indicator.ema_indicator()
                logger.debug(f"EMA calculado: últimos 3 valores = {df_work['ema'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular EMA: {e}")
                df_work['ema'] = df_work['close']  # Fallback para close
            
            # 2. RSI (Relative Strength Index)
            logger.debug("Calculando RSI...")
            try:
                rsi_indicator = RSIIndicator(close=df_work['close'], window=14)
                df_work['rsi'] = rsi_indicator.rsi()
                logger.debug(f"RSI calculado: últimos 3 valores = {df_work['rsi'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular RSI: {e}")
                df_work['rsi'] = 50.0  # Fallback para valor neutro
            
            # 3. MACD (Moving Average Convergence Divergence)
            logger.debug("Calculando MACD...")
            try:
                macd_indicator = MACD(close=df_work['close'])
                df_work['macd'] = macd_indicator.macd()
                df_work['macd_signal'] = macd_indicator.macd_signal()
                logger.debug(f"MACD calculado: últimos 3 valores = {df_work['macd'].tail(3).values}")
                logger.debug(f"MACD Signal calculado: últimos 3 valores = {df_work['macd_signal'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular MACD: {e}")
                df_work['macd'] = 0.0
                df_work['macd_signal'] = 0.0
            
            # 4. Bollinger Bands
            logger.debug("Calculando Bollinger Bands...")
            try:
                bb_indicator = BollingerBands(close=df_work['close'], window=20, window_dev=2)
                df_work['bb_mavg'] = bb_indicator.bollinger_mavg()
                df_work['bb_high'] = bb_indicator.bollinger_hband()
                df_work['bb_low'] = bb_indicator.bollinger_lband()
                logger.debug(f"BB Middle calculado: últimos 3 valores = {df_work['bb_mavg'].tail(3).values}")
                logger.debug(f"BB Upper calculado: últimos 3 valores = {df_work['bb_high'].tail(3).values}")
                logger.debug(f"BB Lower calculado: últimos 3 valores = {df_work['bb_low'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular Bollinger Bands: {e}")
                df_work['bb_mavg'] = df_work['close']
                df_work['bb_high'] = df_work['close'] * 1.02
                df_work['bb_low'] = df_work['close'] * 0.98
            
            # 5. VWAP (Volume Weighted Average Price)
            logger.debug("Calculando VWAP...")
            try:
                vwap_indicator = VolumeWeightedAveragePrice(
                    high=df_work['high'], 
                    low=df_work['low'], 
                    close=df_work['close'], 
                    volume=df_work['volume']
                )
                df_work['vwap'] = vwap_indicator.volume_weighted_average_price()
                logger.debug(f"VWAP calculado: últimos 3 valores = {df_work['vwap'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular VWAP: {e}")
                df_work['vwap'] = df_work['close']  # Fallback para close
            
            # 6. ATR (Average True Range)
            logger.debug("Calculando ATR...")
            try:
                atr_indicator = AverageTrueRange(
                    high=df_work['high'], 
                    low=df_work['low'], 
                    close=df_work['close'], 
                    window=14
                )
                df_work['atr'] = atr_indicator.average_true_range()
                logger.debug(f"ATR calculado: últimos 3 valores = {df_work['atr'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular ATR: {e}")
                df_work['atr'] = (df_work['high'] - df_work['low']).rolling(14).mean()
            
            # 7. OBV (On Balance Volume)
            logger.debug("Calculando OBV...")
            try:
                obv_indicator = OnBalanceVolumeIndicator(
                    close=df_work['close'], 
                    volume=df_work['volume']
                )
                df_work['obv'] = obv_indicator.on_balance_volume()
                logger.debug(f"OBV calculado: últimos 3 valores = {df_work['obv'].tail(3).values}")
            except Exception as e:
                logger.error(f"Erro ao calcular OBV: {e}")
                df_work['obv'] = df_work['volume'].cumsum()  # Fallback simples
            
            # Verificar se temos todos os indicadores necessários
            required_indicators = ['close', 'volume', 'ema', 'rsi', 'macd', 'macd_signal', 
                                  'bb_mavg', 'bb_high', 'bb_low', 'vwap', 'atr', 'obv']
            
            missing = [ind for ind in required_indicators if ind not in df_work.columns]
            if missing:
                logger.error(f"Indicadores faltando após cálculo: {missing}")
                raise ValueError(f"Indicadores faltando após cálculo: {missing}")
            
            logger.info(f"Todos os indicadores calculados. Colunas disponíveis: {list(df_work.columns)}")
            
            # Selecionar apenas os indicadores necessários na ordem correta
            df_indicators = df_work[required_indicators].copy()
            
            logger.info(f"DataFrame de indicadores antes da limpeza: {df_indicators.shape}")
            logger.info(f"NaN por coluna:\n{df_indicators.isna().sum()}")
            
            # Tratar valores NaN
            # Para os primeiros valores que podem ser NaN devido ao cálculo dos indicadores,
            # vamos usar forward fill e backward fill
            df_indicators = df_indicators.fillna(method='ffill').fillna(method='bfill')
            
            # Se ainda houver NaN, preencher com valores padrão
            if df_indicators.isna().any().any():
                logger.warning("Ainda há valores NaN após preenchimento. Usando valores padrão.")
                df_indicators = df_indicators.fillna({
                    'close': df_indicators['close'].mean(),
                    'volume': df_indicators['volume'].mean(),
                    'ema': df_indicators['close'].mean(),
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'bb_mavg': df_indicators['close'].mean(),
                    'bb_high': df_indicators['close'].mean() * 1.02,
                    'bb_low': df_indicators['close'].mean() * 0.98,
                    'vwap': df_indicators['close'].mean(),
                    'atr': df_indicators['close'].std(),
                    'obv': df_indicators['volume'].sum()
                })
            
            logger.info(f"DataFrame de indicadores após limpeza: {df_indicators.shape}")
            
            if len(df_indicators) == 0:
                logger.error("DataFrame de indicadores está vazio após limpeza")
                raise ValueError("DataFrame de indicadores está vazio após limpeza")
            
            # Atualizar o DataFrame do ambiente com os novos indicadores
            # Manter apenas os dados mais recentes para não sobrecarregar a memória
            max_history = 1000
            if len(df_indicators) > max_history:
                df_indicators = df_indicators.tail(max_history).reset_index(drop=True)
            
            # Atualizar o ambiente
            self.env.data = df_indicators
            self.env.current_step = len(df_indicators) - 1  # Usar o último ponto
            
            logger.info(f"Ambiente atualizado. Data shape: {self.env.data.shape}, current_step: {self.env.current_step}")
            
            # Obter a observação do ambiente
            observation = self.env._get_obs()
            
            logger.info(f"Observação preparada com sucesso: shape {observation.shape}")
            logger.debug(f"Valores da observação: {observation}")
            
            # Verificação final
            if len(observation) != 12:
                logger.error(f"Observação tem {len(observation)} features, esperado 12")
                raise ValueError(f"Observação tem {len(observation)} features, esperado 12")
            
            # Verificar se há valores inválidos
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                logger.warning("Observação contém valores inválidos (NaN ou Inf). Corrigindo...")
                observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return observation
            
        except Exception as e:
            logger.error(f"Erro ao preparar observação Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback: tentar usar a observação do ambiente como está
            try:
                logger.info("Tentando fallback para observação do ambiente...")
                fallback_obs = self.env._get_obs()
                logger.info(f"Fallback bem-sucedido: shape {fallback_obs.shape}")
                return fallback_obs
            except Exception as fallback_error:
                logger.error(f"Fallback também falhou: {fallback_error}")
                # Último recurso: retornar array de zeros
                logger.warning("Retornando array de zeros como último recurso")
                return np.zeros(12, dtype=np.float32)

    def _update_environment_data(self, df):
        """
        Método auxiliar para atualizar os dados do ambiente de forma segura.
        
        Args:
            df (pd.DataFrame): DataFrame com os indicadores calculados
        """
        try:
            # Verificar se o DataFrame tem as colunas corretas
            required_indicators = ['close', 'volume', 'ema', 'rsi', 'macd', 'macd_signal', 
                                  'bb_mavg', 'bb_high', 'bb_low', 'vwap', 'atr', 'obv']
            
            if not all(col in df.columns for col in required_indicators):
                missing = [col for col in required_indicators if col not in df.columns]
                raise ValueError(f"Colunas faltando no DataFrame: {missing}")
            
            # Atualizar o ambiente
            self.env.data = df[required_indicators].copy()
            self.env.current_step = len(self.env.data) - 1
            
            logger.info(f"Ambiente atualizado com sucesso. Shape: {self.env.data.shape}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar dados do ambiente: {e}")
            raise

    def _validate_observation(self, observation):
        """
        Valida se a observação está no formato correto.
        
        Args:
            observation (np.ndarray): Array da observação
            
        Returns:
            np.ndarray: Observação validada e corrigida se necessário
        """
        try:
            # Verificar se é um array numpy
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            
            # Verificar dimensões
            if len(observation) != 12:
                raise ValueError(f"Observação tem {len(observation)} features, esperado 12")
            
            # Verificar valores inválidos
            if np.any(np.isnan(observation)):
                logger.warning("Observação contém NaN. Substituindo por 0.")
                observation = np.nan_to_num(observation, nan=0.0)
            
            if np.any(np.isinf(observation)):
                logger.warning("Observação contém valores infinitos. Limitando valores.")
                observation = np.nan_to_num(observation, posinf=1e6, neginf=-1e6)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro na validação da observação: {e}")
            # Retornar array de zeros como fallback
            return np.zeros(12, dtype=np.float32)
    
    def execute_trade(self, action, price):
        """
        Executa uma operação de trading no mercado Futures Testnet.
        
        Args:
            action: Ação a ser executada (0=Hold, 1=Buy, 2=Sell)
            price: Preço atual
            
        Returns:
            float: Recompensa da operação
        """
        reward = 0
        
        if action == 1 and self.position == 0:  # Comprar
            effective_entry_price = price * (1 + TRADING_FEE)
            entry_fee = price * TRADING_FEE
            
            # Exibir informações da operação
            logger.info("=" * 50)
            logger.info(f"INICIANDO COMPRA FUTURES TESTNET @ {price:.2f}")
            logger.info(f"Preço efetivo com taxa: {effective_entry_price:.2f}")
            logger.info(f"Taxa de entrada: {entry_fee:.4f}")
            
            if not self.simulation_mode:
                try:
                    logger.info(f"Enviando ordem de compra Futures Testnet para {self.trade_amount} {self.symbol.split('/')[0]}...")
                    
                    # Criar ordem via API Futures
                    order = self.futures_manager.create_order(
                        side="BUY",
                        quantity=self.trade_amount
                    )
                    
                    self.entry_price = float(order['avgPrice']) if 'avgPrice' in order else effective_entry_price
                    logger.info(f"Ordem de compra Futures Testnet executada: ID={order.get('orderId', 'N/A')}")
                    logger.info(f"Detalhes: {order}")
                except Exception as e:
                    logger.error(f"Erro ao executar ordem de compra Futures Testnet: {e}")
                    logger.error(f"Detalhes: {traceback.format_exc()}")
                    self.entry_price = effective_entry_price
            else:
                logger.info(f"SIMULAÇÃO: Executando ordem de compra Futures Testnet @ {price:.2f}")
                self.entry_price = effective_entry_price
            
            logger.info(f"Posição COMPRADA Futures Testnet aberta com preço de entrada: {self.entry_price:.2f}")
            logger.info("=" * 50)
            
            self.entry_step = self.current_step
            self.position = 1
            
        elif action == 2 and self.position == 0:  # Vender
            effective_entry_price = price * (1 - TRADING_FEE)
            entry_fee = price * TRADING_FEE
            
            # Exibir informações da operação
            logger.info("=" * 50)
            logger.info(f"INICIANDO VENDA FUTURES TESTNET @ {price:.2f}")
            logger.info(f"Preço efetivo com taxa: {effective_entry_price:.2f}")
            logger.info(f"Taxa de entrada: {entry_fee:.4f}")
            
            if not self.simulation_mode:
                try:
                    logger.info(f"Enviando ordem de venda Futures Testnet para {self.trade_amount} {self.symbol.split('/')[0]}...")
                    
                    # Criar ordem via API Futures
                    order = self.futures_manager.create_order(
                        side="SELL",
                        quantity=self.trade_amount
                    )
                    
                    self.entry_price = float(order['avgPrice']) if 'avgPrice' in order else effective_entry_price
                    logger.info(f"Ordem de venda Futures Testnet executada: ID={order.get('orderId', 'N/A')}")
                    logger.info(f"Detalhes: {order}")
                except Exception as e:
                    logger.error(f"Erro ao executar ordem de venda Futures Testnet: {e}")
                    logger.error(f"Detalhes: {traceback.format_exc()}")
                    self.entry_price = effective_entry_price
            else:
                logger.info(f"SIMULAÇÃO: Executando ordem de venda Futures Testnet @ {price:.2f}")
                self.entry_price = effective_entry_price
            
            logger.info(f"Posição VENDIDA Futures Testnet aberta com preço de entrada: {self.entry_price:.2f}")
            logger.info("=" * 50)
            
            self.entry_step = self.current_step
            self.position = -1
            
        elif action == 0 and self.position != 0:  # Fechar posição
            trade_duration = self.current_step - self.entry_step
            exit_fee = price * TRADING_FEE
            
            if self.position == 1:  # Fechar compra (long)
                effective_exit_price = price * (1 - TRADING_FEE)
                reward = effective_exit_price - self.entry_price
                profit_percentage = reward / self.entry_price if self.entry_price != 0 else 0
                meets_threshold = profit_percentage >= MIN_PROFIT_THRESHOLD
                
                # Exibir informações da operação
                logger.info("=" * 50)
                logger.info(f"ANALISANDO FECHAMENTO DE COMPRA FUTURES TESTNET @ {price:.2f}")
                logger.info(f"Preço de entrada: {self.entry_price:.2f}")
                logger.info(f"Preço efetivo de saída: {effective_exit_price:.2f}")
                logger.info(f"Lucro bruto: {price - self.entry_price / (1 + TRADING_FEE):.4f}")
                logger.info(f"Taxas totais: {exit_fee + self.entry_price - self.entry_price / (1 + TRADING_FEE):.4f}")
                logger.info(f"Lucro líquido: {reward:.4f} ({profit_percentage*100:.2f}%)")
                logger.info(f"Duração: {trade_duration} steps")
                logger.info(f"Limiar mínimo: {MIN_PROFIT_THRESHOLD*100:.2f}%")
                logger.info(f"Limiar atingido: {'Sim' if meets_threshold else 'Não'}")
                
                if meets_threshold or reward <= 0:
                    if not self.simulation_mode:
                        try:
                            logger.info(f"Enviando ordem para fechar posição COMPRADA Futures Testnet...")
                            
                            # Criar ordem via API Futures
                            order = self.futures_manager.create_order(
                                side="SELL",
                                quantity=self.trade_amount
                            )
                            
                            logger.info(f"Ordem de fechamento Futures Testnet executada: ID={order.get('orderId', 'N/A')}")
                            logger.info(f"Detalhes: {order}")
                        except Exception as e:
                            logger.error(f"Erro ao fechar posição comprada Futures Testnet: {e}")
                            logger.error(f"Detalhes: {traceback.format_exc()}")
                    else:
                        logger.info(f"SIMULAÇÃO: Fechando posição comprada Futures Testnet @ {price:.2f}")
                    
                    logger.info(f"POSIÇÃO COMPRADA FUTURES TESTNET FECHADA com {'LUCRO' if reward > 0 else 'PREJUÍZO'} de {reward:.4f}")
                    self.position = 0
                    self.entry_price = 0
                    self.total_trades += 1
                    if reward > 0:
                        self.profitable_trades += 1
                        self.total_gain += reward
                    else:
                        self.losing_trades += 1
                        self.total_loss += abs(reward)
                else:
                    logger.info(f"MANTENDO POSIÇÃO COMPRADA FUTURES TESTNET - Lucro atual abaixo do limiar mínimo")
                    reward = 0
                
                logger.info("=" * 50)
                    
            elif self.position == -1:  # Fechar venda (short)
                effective_exit_price = price * (1 + TRADING_FEE)
                reward = self.entry_price - effective_exit_price
                profit_percentage = reward / self.entry_price if self.entry_price != 0 else 0
                meets_threshold = profit_percentage >= MIN_PROFIT_THRESHOLD
                
                # Exibir informações da operação
                logger.info("=" * 50)
                logger.info(f"ANALISANDO FECHAMENTO DE VENDA FUTURES TESTNET @ {price:.2f}")
                logger.info(f"Preço de entrada: {self.entry_price:.2f}")
                logger.info(f"Preço efetivo de saída: {effective_exit_price:.2f}")
                logger.info(f"Lucro bruto: {self.entry_price / (1 - TRADING_FEE) - price:.4f}")
                logger.info(f"Taxas totais: {exit_fee + self.entry_price / (1 - TRADING_FEE) - self.entry_price:.4f}")
                logger.info(f"Lucro líquido: {reward:.4f} ({profit_percentage*100:.2f}%)")
                logger.info(f"Duração: {trade_duration} steps")
                logger.info(f"Limiar mínimo: {MIN_PROFIT_THRESHOLD*100:.2f}%")
                logger.info(f"Limiar atingido: {'Sim' if meets_threshold else 'Não'}")
                
                if meets_threshold or reward <= 0:
                    if not self.simulation_mode:
                        try:
                            logger.info(f"Enviando ordem para fechar posição VENDIDA Futures Testnet...")
                            
                            # Criar ordem via API Futures
                            order = self.futures_manager.create_order(
                                side="BUY",
                                quantity=self.trade_amount
                            )
                            
                            logger.info(f"Ordem de fechamento Futures Testnet executada: ID={order.get('orderId', 'N/A')}")
                            logger.info(f"Detalhes: {order}")
                        except Exception as e:
                            logger.error(f"Erro ao fechar posição vendida Futures Testnet: {e}")
                            logger.error(f"Detalhes: {traceback.format_exc()}")
                    else:
                        logger.info(f"SIMULAÇÃO: Fechando posição vendida Futures Testnet @ {price:.2f}")
                    
                    logger.info(f"POSIÇÃO VENDIDA FUTURES TESTNET FECHADA com {'LUCRO' if reward > 0 else 'PREJUÍZO'} de {reward:.4f}")
                    self.position = 0
                    self.entry_price = 0
                    self.total_trades += 1
                    if reward > 0:
                        self.profitable_trades += 1
                        self.total_gain += reward
                    else:
                        self.losing_trades += 1
                        self.total_loss += abs(reward)
                else:
                    logger.info(f"MANTENDO POSIÇÃO VENDIDA FUTURES TESTNET - Lucro atual abaixo do limiar mínimo")
                    reward = 0
                
                logger.info("=" * 50)
        
        return reward
    
    def run(self):
        """
        Executa o bot de trading em tempo real usando WebSockets para o mercado Futures Testnet.
        """
        logger.info("Iniciando execução do bot de trading Futures Testnet com WebSockets oficiais")
        
        try:
            # Aguardar dados iniciais
            logger.info("Aguardando dados iniciais do WebSocket Futures Testnet...")
            while self.futures_manager.df is None:
                time.sleep(1)
            
            logger.info(f"Dados iniciais Futures Testnet recebidos: {len(self.futures_manager.df)} registros")
            
            # Loop principal
            while True:
                # Obter DataFrame atual
                df = self.futures_manager.get_current_dataframe()
                
                if df is None or len(df) < 10:
                    logger.warning("Dados Futures Testnet insuficientes. Aguardando...")
                    time.sleep(1)
                    continue
                
                # Preparar observação
                try:
                    observation = self._prepare_observation(df)
                except Exception as e:
                    logger.error(f"Erro ao preparar observação Futures Testnet: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(1)
                    continue
                
                # Obter previsão do modelo
                action, _states = self.model.predict(observation, deterministic=False)
                
                # Obter preço atual
                current_price = self.futures_manager.get_current_price()
                
                if current_price is None:
                    logger.warning("Preço atual Futures Testnet não disponível. Aguardando...")
                    time.sleep(1)
                    continue
                
                # Executar trade
                reward = self.execute_trade(int(action), current_price)
                
                # Incrementar contador de passos
                self.current_step += 1
                
                # Exibir estatísticas parciais a cada 50 passos
                if self.current_step % 50 == 0:
                    self._print_stats()
                
                # Aguardar próxima verificação
                time.sleep(TRADE_UPDATE_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Execução interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante execução do bot Futures Testnet: {e}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            # Fechar posições abertas
            self._close_positions()
            
            # Parar WebSocket
            self.futures_manager.stop()
            
            # Exibir estatísticas finais
            self._print_stats(final=True)
    
    def _close_positions(self):
        """
        Fecha todas as posições abertas no mercado Futures Testnet.
        """
        if self.position != 0:
            logger.info("Fechando posições Futures Testnet abertas...")
            
            current_price = self.futures_manager.get_current_price()
            
            if current_price is None:
                logger.warning("Preço atual Futures Testnet não disponível. Usando último preço conhecido.")
                current_price = self.entry_price
            
            if not self.simulation_mode:
                try:
                    if self.position == 1:
                        # Criar ordem via API Futures
                        order = self.futures_manager.create_order(
                            side="SELL",
                            quantity=self.trade_amount
                        )
                        logger.info("Posição comprada Futures Testnet fechada")
                    elif self.position == -1:
                        # Criar ordem via API Futures
                        order = self.futures_manager.create_order(
                            side="BUY",
                            quantity=self.trade_amount
                        )
                        logger.info("Posição vendida Futures Testnet fechada")
                except Exception as e:
                    logger.error(f"Erro ao fechar posições Futures Testnet: {e}")
            else:
                logger.info("SIMULAÇÃO: Fechando posições Futures Testnet abertas")
            
            self.position = 0
            self.entry_price = 0
    
    def _print_stats(self, final=False):
        """
        Exibe estatísticas de trading.
        
        Args:
            final: Se True, exibe estatísticas finais
        """
        if final:
            logger.info("--- ESTATÍSTICAS FINAIS FUTURES TESTNET ---")
        else:
            logger.info(f"--- ESTATÍSTICAS PARCIAIS FUTURES TESTNET (Passo {self.current_step}) ---")
        
        logger.info(f"→ Operações totais: {self.total_trades}")
        
        if self.total_trades > 0:
            win_rate = (self.profitable_trades / self.total_trades) * 100
            loss_rate = (self.losing_trades / self.total_trades) * 100
            
            logger.info(f"✓ Lucros: {self.profitable_trades} ({win_rate:.2f}%)")
            logger.info(f"! Prejuízos: {self.losing_trades} ({loss_rate:.2f}%)")
            logger.info(f"+ Ganho total líquido: {self.total_gain:.4f}")
            logger.info(f"- Perda total líquida: {self.total_loss:.4f}")
            
            if self.total_loss > 0:
                profit_factor = self.total_gain / self.total_loss
                logger.info(f"→ Fator de lucro: {profit_factor:.2f}")
            
            net_profit = self.total_gain - self.total_loss
            logger.info(f"$ Lucro líquido total: {net_profit:.4f}")
        else:
            logger.info("Nenhuma operação realizada ainda.")
        
        logger.info("-" * 30)


def main():
    """
    Função principal para execução do bot.
    """
    logger.info("Iniciando script de execução do bot Futures Testnet com WebSockets oficiais")
    
    try:
        # Verificar se diretórios necessários existem
        for directory in [MODEL_DIR, CONFIG_DIR, LOG_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Diretório criado: {directory}")
        
        # Verificar se arquivos necessários existem
        model_file_path = f"{MODEL_PATH}.zip"
        if not os.path.exists(model_file_path):
            logger.error(f"Modelo não encontrado: {model_file_path}")
            logger.error("Certifique-se de que o modelo treinado (ppo_model.zip) e os arquivos de configuração existem nos diretórios corretos.")
            return
        
        if not os.path.exists(INDICATORS_CONFIG_PATH):
            logger.error(f"Arquivo de indicadores não encontrado: {INDICATORS_CONFIG_PATH}")
            logger.error("Certifique-se de que o arquivo de indicadores (indicators.txt) existe no diretório de configurações.")
            return
        
        # Obter credenciais da API da Binance
        api_key = os.environ.get('BINANCE_API_KEY', BINANCE_API_KEY)
        api_secret = os.environ.get('BINANCE_API_SECRET', BINANCE_API_SECRET)
        
        # Verificar se as credenciais estão disponíveis
        if not api_key or not api_secret:
            logger.warning("Credenciais da API Binance não encontradas.")
            logger.warning("Você precisa configurar as credenciais da Testnet da Binance.")
            logger.warning("Você pode configurá-las como variáveis de ambiente:")
            logger.warning("export BINANCE_API_KEY='sua_api_key_da_testnet'")
            logger.warning("export BINANCE_API_SECRET='sua_api_secret_da_testnet'")
            logger.warning("Ou diretamente no código, nas constantes BINANCE_API_KEY e BINANCE_API_SECRET.")
            return
        
        # Inicializar e executar o bot
        bot = ScalpingBotFuturesOfficial(
            api_key=api_key,
            api_secret=api_secret,
            simulation_mode=False  # FORÇADO para modo real
        )
        
        # Executar o bot
        bot.run()
        
    except Exception as e:
        logger.error(f"Erro durante execução do bot Futures Testnet: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
