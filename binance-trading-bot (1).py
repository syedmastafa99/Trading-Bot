import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import talib as ta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

class BinanceTradingBot:
    def __init__(self, api_key=None, api_secret=None, test_mode=True):
        """
        Initialize the Binance trading bot
        
        Parameters:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        test_mode (bool): If True, run in test mode without making actual trades
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.test_mode = test_mode
        self.client = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Trading parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.pairs_to_analyze = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.timeframes = {'1h': Client.KLINE_INTERVAL_1HOUR, 
                          '4h': Client.KLINE_INTERVAL_4HOUR, 
                          '1d': Client.KLINE_INTERVAL_1DAY}
        
        # Connect to Binance
        self.connect_to_binance()
        
    def connect_to_binance(self):
        """Establish connection with Binance API"""
        try:
            if self.test_mode:
                self.client = Client(self.api_key, self.api_secret, testnet=True)
                logger.info("Connected to Binance Testnet")
            else:
                self.client = Client(self.api_key, self.api_secret)
                logger.info("Connected to Binance")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
            
    def get_account_balance(self, asset='USDT'):
        """Get account balance for a specific asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0
        except BinanceAPIException as e:
            logger.error(f"Error fetching account balance: {e}")
            return 0
            
    def get_historical_data(self, symbol, interval, lookback_days=30):
        """
        Get historical OHLCV data from Binance
        
        Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Candlestick interval
        lookback_days (int): Number of days to look back
        
        Returns:
        pandas.DataFrame: Historical data
        """
        try:
            start_str = str(int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000))
            klines = self.client.get_historical_klines(symbol, interval, start_str)
            
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            
            # Convert columns to numeric values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
            
            data.set_index('timestamp', inplace=True)
            
            return data
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
            
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for trading analysis
        
        Parameters:
        df (pandas.DataFrame): Historical price data
        
        Returns:
        pandas.DataFrame: Price data with technical indicators
        """
        # Make a copy of the dataframe
        data = df.copy()
        
        # Simple Moving Averages
        data['SMA_20'] = ta.SMA(data['close'], timeperiod=20)
        data['SMA_50'] = ta.SMA(data['close'], timeperiod=50)
        data['SMA_200'] = ta.SMA(data['close'], timeperiod=200)
        
        # Exponential Moving Averages
        data['EMA_12'] = ta.EMA(data['close'], timeperiod=12)
        data['EMA_26'] = ta.EMA(data['close'], timeperiod=26)
        
        # MACD
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['close'], 
                                                                      fastperiod=12, 
                                                                      slowperiod=26, 
                                                                      signalperiod=9)
        
        # RSI
        data['RSI'] = ta.RSI(data['close'], timeperiod=14)
        
        # Bollinger Bands
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.BBANDS(data['close'], 
                                                                        timeperiod=20, 
                                                                        nbdevup=2, 
                                                                        nbdevdn=2)
        
        # Stochastic Oscillator
        data['STOCH_K'], data['STOCH_D'] = ta.STOCH(data['high'], data['low'], data['close'], 
                                                  fastk_period=14, 
                                                  slowk_period=3, 
                                                  slowd_period=3)
        
        # Average True Range (ATR)
        data['ATR'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # On-Balance Volume (OBV)
        data['OBV'] = ta.OBV(data['close'], data['volume'])
        
        # Percentage Price Oscillator (PPO)
        data['PPO'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)
        
        # Commodity Channel Index (CCI)
        data['CCI'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Williams %R
        data['WILLR'] = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Average Directional Index (ADX)
        data['ADX'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
        
    def build_lstm_model(self, X_train):
        """
        Build and compile LSTM model for price prediction
        
        Parameters:
        X_train (numpy.ndarray): Training data shape
        
        Returns:
        tensorflow.keras.models.Sequential: Compiled LSTM model
        """
        model = Sequential()
        
        # Add LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
        
    def prepare_data_for_lstm(self, data, feature_columns, target_column='close', sequence_length=60):
        """
        Prepare data for LSTM model
        
        Parameters:
        data (pandas.DataFrame): Data with features
        feature_columns (list): List of feature column names
        target_column (str): Target column name
        sequence_length (int): Sequence length for LSTM
        
        Returns:
        tuple: X_train, y_train, X_test, y_test, scaler
        """
        # Select features
        features = data[feature_columns].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i + sequence_length])
            # Target is the closing price
            target_idx = feature_columns.index(target_column)
            y.append(scaled_features[i + sequence_length, target_idx])
            
        X, y = np.array(X), np.array(y)
        
        # Split into training and testing sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
        
    def train_model(self, symbol, interval='1d', epochs=50, batch_size=32):
        """
        Train LSTM model for price prediction
        
        Parameters:
        symbol (str): Trading pair symbol
        interval (str): Candlestick interval
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
        Returns:
        tensorflow.keras.models.Sequential: Trained model
        """
        # Get historical data
        data = self.get_historical_data(symbol, self.timeframes[interval], lookback_days=365)
        if data is None:
            return None
            
        # Calculate technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Select features for prediction
        feature_columns = ['close', 'volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'ATR', 'CCI']
        
        # Prepare data for LSTM
        X_train, y_train, X_test, y_test = self.prepare_data_for_lstm(data, feature_columns)
        
        # Build model
        self.model = self.build_lstm_model(X_train)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        logger.info(f"Model trained for {symbol} with final loss: {history.history['loss'][-1]}")
        
        return self.model
        
    def predict_next_price(self, symbol, interval='1d', days_to_predict=5):
        """
        Predict future prices using the trained model
        
        Parameters:
        symbol (str): Trading pair symbol
        interval (str): Candlestick interval
        days_to_predict (int): Number of days to predict ahead
        
        Returns:
        list: Predicted prices
        """
        if self.model is None:
            logger.info(f"Training model for {symbol} first...")
            self.train_model(symbol, interval)
            
        # Get latest data
        data = self.get_historical_data(symbol, self.timeframes[interval], lookback_days=100)
        if data is None:
            return None
            
        # Calculate technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Select features for prediction
        feature_columns = ['close', 'volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'ATR', 'CCI']
        
        # Get the scaled data
        scaled_data = self.scaler.transform(data[feature_columns].values)
        
        # Make predictions for future prices
        predicted_prices = []
        last_sequence = scaled_data[-60:].reshape(1, 60, len(feature_columns))
        
        current_price = data['close'].iloc[-1]
        
        for _ in range(days_to_predict):
            # Predict next price
            next_pred = self.model.predict(last_sequence)
            
            # Create a copy of the last row to update with prediction
            next_row = scaled_data[-1].copy()
            # Update closing price with prediction
            target_idx = feature_columns.index('close')
            next_row[target_idx] = next_pred[0][0]
            
            # Update the sequence
            last_sequence = np.append(last_sequence[:, 1:, :], [next_row.reshape(1, len(feature_columns))], axis=1)
            
            # Collect predicted price
            # We need to inverse transform to get the actual price
            pred_row = np.zeros((1, len(feature_columns)))
            pred_row[0, target_idx] = next_pred[0][0]
            actual_pred = self.scaler.inverse_transform(pred_row)[0, target_idx]
            predicted_prices.append(actual_pred)
            
        return current_price, predicted_prices
        
    def analyze_market_sentiment(self, symbol, intervals=None):
        """
        Analyze market sentiment using multiple indicators and timeframes
        
        Parameters:
        symbol (str): Trading pair symbol
        intervals (list): List of timeframe intervals to analyze
        
        Returns:
        dict: Market sentiment analysis
        """
        if intervals is None:
            intervals = ['1h', '4h', '1d']
            
        sentiment = {
            'overall': 0,  # -100 to 100 score
            'timeframes': {},
            'momentum': None,
            'volatility': None,
            'trend': None,
            'support_resistance': {}
        }
        
        for interval in intervals:
            data = self.get_historical_data(symbol, self.timeframes[interval], lookback_days=30)
            if data is None:
                continue
                
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Calculate sentiment for this timeframe
            timeframe_sentiment = 0
            
            # Trend analysis
            if latest['close'] > latest['SMA_50']:
                timeframe_sentiment += 10
            else:
                timeframe_sentiment -= 10
                
            if latest['close'] > latest['SMA_200']:
                timeframe_sentiment += 15
            else:
                timeframe_sentiment -= 15
                
            # MACD analysis
            if latest['MACD'] > latest['MACD_signal']:
                timeframe_sentiment += 10
            else:
                timeframe_sentiment -= 10
                
            # RSI analysis
            if latest['RSI'] < 30:
                timeframe_sentiment += 15  # Oversold
            elif latest['RSI'] > 70:
                timeframe_sentiment -= 15  # Overbought
            else:
                # Neutral RSI
                rsi_sentiment = (latest['RSI'] - 50) * -0.6  # Scale and invert
                timeframe_sentiment += rsi_sentiment
                
            # Bollinger Bands
            bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle']
            
            if latest['close'] < latest['BB_lower']:
                timeframe_sentiment += 15  # Potential bounce
            elif latest['close'] > latest['BB_upper']:
                timeframe_sentiment -= 15  # Potential reversal
                
            # Stochastic
            if latest['STOCH_K'] < 20 and latest['STOCH_D'] < 20:
                timeframe_sentiment += 10  # Oversold
            elif latest['STOCH_K'] > 80 and latest['STOCH_D'] > 80:
                timeframe_sentiment -= 10  # Overbought
                
            # ADX (trend strength)
            trend_strength = min(latest['ADX'] / 5, 10)  # Scale ADX
            if latest['close'] > latest['SMA_50']:
                timeframe_sentiment += trend_strength  # Strong uptrend
            else:
                timeframe_sentiment -= trend_strength  # Strong downtrend
                
            # Store sentiment for this timeframe
            sentiment['timeframes'][interval] = round(timeframe_sentiment, 2)
            
            # Compute volatility
            volatility = latest['ATR'] / latest['close'] * 100
            
            # Find support and resistance levels
            price_history = data['close'].values
            
            # Simple method to identify support/resistance levels
            levels = []
            for i in range(5, len(price_history) - 5):
                if (price_history[i] <= price_history[i-1] and 
                    price_history[i] <= price_history[i-2] and
                    price_history[i] <= price_history[i+1] and 
                    price_history[i] <= price_history[i+2]):
                    # Found a support level
                    levels.append(('support', price_history[i]))
                elif (price_history[i] >= price_history[i-1] and 
                      price_history[i] >= price_history[i-2] and
                      price_history[i] >= price_history[i+1] and 
                      price_history[i] >= price_history[i+2]):
                    # Found a resistance level
                    levels.append(('resistance', price_history[i]))
            
            # Filter levels to the most significant ones
            if levels:
                # Sort by price
                levels.sort(key=lambda x: x[1])
                
                # Group close levels
                grouped_levels = []
                current_group = [levels[0]]
                
                for i in range(1, len(levels)):
                    if abs(levels[i][1] - current_group[-1][1]) / current_group[-1][1] < 0.005:  # 0.5% threshold
                        current_group.append(levels[i])
                    else:
                        # Calculate average price for the group
                        avg_price = sum(x[1] for x in current_group) / len(current_group)
                        # Determine if it's mainly support or resistance
                        support_count = sum(1 for x in current_group if x[0] == 'support')
                        resistance_count = len(current_group) - support_count
                        level_type = 'support' if support_count > resistance_count else 'resistance'
                        
                        grouped_levels.append((level_type, avg_price))
                        current_group = [levels[i]]
                
                # Add the last group
                if current_group:
                    avg_price = sum(x[1] for x in current_group) / len(current_group)
                    support_count = sum(1 for x in current_group if x[0] == 'support')
                    resistance_count = len(current_group) - support_count
                    level_type = 'support' if support_count > resistance_count else 'resistance'
                    grouped_levels.append((level_type, avg_price))
                
                # Store only the closest levels
                current_price = latest['close']
                closest_levels = []
                
                # Find closest support below current price
                supports = [(i, level) for i, (type_, level) in enumerate(grouped_levels) 
                           if type_ == 'support' and level < current_price]
                if supports:
                    closest_support = max(supports, key=lambda x: x[1])
                    closest_levels.append(('support', closest_support[1]))
                
                # Find closest resistance above current price
                resistances = [(i, level) for i, (type_, level) in enumerate(grouped_levels) 
                              if type_ == 'resistance' and level > current_price]
                if resistances:
                    closest_resistance = min(resistances, key=lambda x: x[1])
                    closest_levels.append(('resistance', closest_resistance[1]))
                
                sentiment['support_resistance'][interval] = closest_levels
        
        # Calculate overall sentiment (weighted average of timeframes)
        weights = {'1h': 0.2, '4h': 0.3, '1d': 0.5}
        sentiment['overall'] = sum(sentiment['timeframes'].get(interval, 0) * weights.get(interval, 0) 
                                 for interval in intervals)
        
        # Determine momentum, volatility, and trend
        if sentiment['overall'] > 20:
            sentiment['momentum'] = 'Strong Bullish'
        elif sentiment['overall'] > 5:
            sentiment['momentum'] = 'Bullish'
        elif sentiment['overall'] < -20:
            sentiment['momentum'] = 'Strong Bearish'
        elif sentiment['overall'] < -5:
            sentiment['momentum'] = 'Bearish'
        else:
            sentiment['momentum'] = 'Neutral'
        
        # Average volatility across timeframes
        try:
            avg_vol = volatility  # Just use the last calculated volatility for simplicity
            if avg_vol > 5:
                sentiment['volatility'] = 'High'
            elif avg_vol > 2:
                sentiment['volatility'] = 'Medium'
            else:
                sentiment['volatility'] = 'Low'
        except:
            sentiment['volatility'] = 'Unknown'
        
        # Trend determination based on moving averages in daily timeframe
        daily_data = data if interval == '1d' else self.get_historical_data(symbol, self.timeframes['1d'], lookback_days=100)
        if daily_data is not None:
            daily_data = self.calculate_technical_indicators(daily_data)
            latest = daily_data.iloc[-1]
            
            if (latest['close'] > latest['SMA_50'] and latest['SMA_50'] > latest['SMA_200']):
                sentiment['trend'] = 'Strong Uptrend'
            elif latest['close'] > latest['SMA_50']:
                sentiment['trend'] = 'Uptrend'
            elif (latest['close'] < latest['SMA_50'] and latest['SMA_50'] < latest['SMA_200']):
                sentiment['trend'] = 'Strong Downtrend'
            elif latest['close'] < latest['SMA_50']:
                sentiment['trend'] = 'Downtrend'
            else:
                sentiment['trend'] = 'Sideways'
        
        return sentiment
        
    def calculate_position_size(self, symbol, stop_loss_percent):
        """
        Calculate position size based on account balance and risk per trade
        
        Parameters:
        symbol (str): Trading pair symbol
        stop_loss_percent (float): Stop loss percentage
        
        Returns:
        float: Position size in quote currency
        """
        # Get account balance
        balance = self.get_account_balance()
        
        # Calculate risk amount
        risk_amount = balance * self.risk_per_trade
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_percent / 100)
        
        return position_size
        
    def generate_trading_recommendation(self, symbol):
        """
        Generate a trading recommendation including entry, targets, stop loss, and position size
        
        Parameters:
        symbol (str): Trading pair symbol
        
        Returns:
        dict: Trading recommendation
        """
        # Analyze market sentiment
        sentiment = self.analyze_market_sentiment(symbol)
        
        # Get current price and price prediction
        current_price, predicted_prices = self.predict_next_price(symbol)
        
        if current_price is None or predicted_prices is None:
            logger.error(f"Failed to get price data for {symbol}")
            return None
            
        # Get daily data for ATR calculation
        daily_data = self.get_historical_data(symbol, self.timeframes['1d'], lookback_days=30)
        daily_data = self.calculate_technical_indicators(daily_data)
        
        # Get latest ATR for stop loss calculation
        latest_atr = daily_data['ATR'].iloc[-1]
        
        # Default recommendations
        recommendation = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_prices': predicted_prices,
            'recommendation': 'HOLD',
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': None,
            'position_size': None,
            'position_size_usd': None,
            'trade_duration': '1-2 weeks',
            'sentiment': sentiment,
            'confidence': 'Medium',
            'notes': []
        }
        
        # Determine if it's a buy or sell based on sentiment and predictions
        avg_predicted = sum(predicted_prices) / len(predicted_prices)
        price_change_percent = (avg_predicted - current_price) / current_price * 100
        
        # Adjust the sentiment threshold based on the prediction
        sentiment_threshold = 10 if price_change_percent > 0 else -10
        
        # Determine recommendation
        if sentiment['overall'] > sentiment_threshold and price_change_percent > 2:
            recommendation['recommendation'] = 'BUY'
            
            # Calculate stop loss (2 ATR)
            stop_loss_price = current_price - (2 * latest_atr)
            stop_loss_percent = (current_price - stop_loss_price) / current_price * 100
            
            # Calculate take profit levels (1:2 and 1:3 risk-reward ratios)
            take_profit_1 = current_price + (current_price - stop_loss_price) * 2
            take_profit_2 = current_price + (current_price - stop_loss_price) * 3
            
            # Calculate position size
            position_size_usd = self.calculate_position_size(symbol, stop_loss_percent)
            position_size = position_size_usd / current_price
            
            # Update recommendation
            recommendation['stop_loss'] = stop_loss_price
            recommendation['take_profit'] = [take_profit_1, take_profit_2]
            recommendation['risk_reward_ratio'] = '1:2 and 1:3'
            recommendation['position_size'] = position_size
            recommendation['position_size_usd'] = position_size_usd
            
            # Set confidence level
            if sentiment['overall'] > 30 and price_change_percent > 5:
                recommendation['confidence'] = 'High'
            elif sentiment['overall'] > 20 and price_change_percent > 3:
                recommendation['confidence'] = 'Medium-High'
                
            # Add notes
            recommendation['notes'].append(f"Strong bullish sentiment ({sentiment['overall']:.2f}) and positive price prediction (+{price_change_percent:.2f}%)")
            if 'support_resistance' in sentiment and '1d' in sentiment['support_resistance']:
                for level_type, level in sentiment['support_resistance']['1d']:
                    if level_type == 'support' and level < current_price:
                        recommendation['notes'].append(f"Price is above support level at {level:.2f}")
            
        elif sentiment['overall'] < sentiment_threshold and price_change_percent < -2:
            recommendation['recommendation'] = 'SELL'
            
            # Calculate stop loss (2 ATR)
            stop_loss_price = current_price + (2 * latest_atr)
            stop_loss_percent = (stop_loss_price - current_price) / current_price * 100
            
            # Calculate take profit levels (1:2 and 1:3 risk-reward ratios)
            take_profit_1 = current_price - (stop_loss_price - current_price) * 2
            take_profit_2 = current_price - (stop_loss_price - current_price) * 3
            
            # Calculate position size
            position_size_usd = self.calculate_position_size(symbol, stop_loss_percent)
            position_size = position_size_usd / current_price
            
            # Update recommendation
            recommendation['stop_loss'] = stop_loss_price
            recommendation['take_profit'] = [take_profit_1, take_profit_2]
            recommendation['risk_reward_ratio'] = '1:2 and 1:3'
            recommendation['position_size'] = position_size
            recommendation['position_size_usd'] = position_size_usd
            
            # Set confidence level
            if sentiment['overall'] < -30 and price_change_percent < -5:
                recommendation['confidence'] = 'High'
            elif sentiment['overall'] < -20 and price_change_percent < -3:
                recommendation['confidence'] = 'Medium-High'
                
            # Add notes
            recommendation['notes'].append(f"Strong bearish sentiment ({sentiment['overall']:.2f}) and negative price prediction ({price_change_percent:.2f}%)")
            if 'support_resistance' in sentiment and '1d' in sentiment['support_resistance']:
                for level_type, level in sentiment['support_resistance']['1d']:
                    if level_type == 'resistance' and level > current_price:
                        recommendation['notes'].append(f"Price is below resistance level at {level:.2f}")
        else:
            recommendation['notes'].append(f"Neutral conditions: sentiment ({sentiment['overall']:.2f}) and price prediction ({price_change_percent:.2f}%)")
            if abs(price_change_percent) < 1:
                recommendation['notes'].append("Price is expected to remain stable")
                
        # Add more context about market conditions
        recommendation['notes'].append(f"Market momentum: {sentiment['momentum']}")
        recommendation['notes'].append(f"Volatility: {sentiment['volatility']}")
        recommendation['notes'].append(f"Trend: {sentiment['trend']}")
        
        # Determine appropriate trading duration
        if sentiment['volatility'] == 'High':
            recommendation['trade_duration'] = '3-5 days'
        elif sentiment['trend'] in ['Strong Uptrend', 'Strong Downtrend']:
            recommendation['trade_duration'] = '2-4 weeks'
        
        return recommendation
        
    def plot_prediction(self, symbol, recommendation):
        """
        Plot the price prediction and technical indicators
        
        Parameters:
        symbol (str): Trading pair symbol
        recommendation (dict): Trading recommendation
        
        Returns:
        matplotlib