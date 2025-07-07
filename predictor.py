"""
Universal Stock Prediction AI - Natural Language Processing
Understands natural language prompts and analyzes any stock/index mentioned
Examples: "tell me will the nifty go up or down", "what about apple stock tomorrow", "adani power prediction"
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings('ignore')

class UniversalStockPredictionAI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Universal Stock Prediction AI
        
        Args:
            api_key: API key for news/LLM integration (optional for demo)
        """
        self.api_key = api_key
        self.stock_data = {}
        self.market_analysis = {}
        self.prediction_result = {}
        
        # Stock symbol mappings for Indian and global markets
        self.stock_mappings = {
            # Indian Indices
            'nifty': '^NSEI',
            'nifty50': '^NSEI',
            'sensex': '^BSESN',
            'banknifty': '^NSEBANK',
            'bank nifty': '^NSEBANK',
            'nifty bank': '^NSEBANK',
            
            # Indian Stocks (NSE)
            'reliance': 'RELIANCE.NS',
            'tcs': 'TCS.NS',
            'hdfc': 'HDFCBANK.NS',
            'hdfc bank': 'HDFCBANK.NS',
            'infosys': 'INFY.NS',
            'icici': 'ICICIBANK.NS',
            'icici bank': 'ICICIBANK.NS',
            'wipro': 'WIPRO.NS',
            'adani': 'ADANIPORTS.NS',
            'adani power': 'ADANIPOWER.NS',
            'adani enterprises': 'ADANIENT.NS',
            'adani green': 'ADANIGREEN.NS',
            'adani transmission': 'ADANITRANS.NS',
            'bajaj finance': 'BAJFINANCE.NS',
            'bajaj finserv': 'BAJAJFINSV.NS',
            'asian paints': 'ASIANPAINT.NS',
            'hul': 'HINDUNILVR.NS',
            'hindustan unilever': 'HINDUNILVR.NS',
            'itc': 'ITC.NS',
            'axis bank': 'AXISBANK.NS',
            'kotak mahindra': 'KOTAKBANK.NS',
            'maruti': 'MARUTI.NS',
            'maruti suzuki': 'MARUTI.NS',
            'tata motors': 'TATAMOTORS.NS',
            'tata steel': 'TATASTEEL.NS',
            'bharti airtel': 'BHARTIARTL.NS',
            'airtel': 'BHARTIARTL.NS',
            'ongc': 'ONGC.NS',
            'ntpc': 'NTPC.NS',
            'power grid': 'POWERGRID.NS',
            'sun pharma': 'SUNPHARMA.NS',
            'dr reddy': 'DRREDDY.NS',
            'cipla': 'CIPLA.NS',
            'tech mahindra': 'TECHM.NS',
            'hcl tech': 'HCLTECH.NS',
            'larsen toubro': 'LT.NS',
            'l&t': 'LT.NS',
            'ultratech cement': 'ULTRACEMCO.NS',
            'nestle': 'NESTLEIND.NS',
            'britannia': 'BRITANNIA.NS',
            'bajaj auto': 'BAJAJ-AUTO.NS',
            'hero motocorp': 'HEROMOTOCO.NS',
            'eicher motors': 'EICHERMOT.NS',
            'mahindra': 'M&M.NS',
            'titan': 'TITAN.NS',
            'coal india': 'COALINDIA.NS',
            'ioc': 'IOC.NS',
            'bpcl': 'BPCL.NS',
            'grasim': 'GRASIM.NS',
            'jswsteel': 'JSWSTEEL.NS',
            'hindalco': 'HINDALCO.NS',
            'vedanta': 'VEDL.NS',
            'sbi': 'SBIN.NS',
            'state bank': 'SBIN.NS',
            'divislab': 'DIVISLAB.NS',
            'divis lab': 'DIVISLAB.NS',
            'apollo hospitals': 'APOLLOHOSP.NS',
            'dmart': 'DMART.NS',
            'avenue supermarts': 'DMART.NS',
            
            # Global Stocks (US)
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
            'nvidia': 'NVDA',
            'intel': 'INTC',
            'amd': 'AMD',
            'ibm': 'IBM',
            'oracle': 'ORCL',
            'salesforce': 'CRM',
            'adobe': 'ADBE',
            'paypal': 'PYPL',
            'visa': 'V',
            'mastercard': 'MA',
            'jpmorgan': 'JPM',
            'bank of america': 'BAC',
            'goldman sachs': 'GS',
            'morgan stanley': 'MS',
            'wells fargo': 'WFC',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'mcdonalds': 'MCD',
            'walmart': 'WMT',
            'disney': 'DIS',
            'nike': 'NKE',
            'boeing': 'BA',
            'caterpillar': 'CAT',
            'ge': 'GE',
            'general electric': 'GE',
            'ford': 'F',
            'general motors': 'GM',
            'exxon': 'XOM',
            'chevron': 'CVX',
            'johnson & johnson': 'JNJ',
            'pfizer': 'PFE',
            'merck': 'MRK',
            'abbvie': 'ABBV',
            'bristol myers': 'BMY',
            'unitedhealth': 'UNH',
            'home depot': 'HD',
            'procter gamble': 'PG',
            'verizon': 'VZ',
            'att': 'T',
            
            # Global Indices
            'sp500': '^GSPC',
            's&p 500': '^GSPC',
            'nasdaq': '^IXIC',
            'dow jones': '^DJI',
            'dow': '^DJI',
            'ftse': '^FTSE',
            'dax': '^GDAXI',
            'nikkei': '^N225',
            'hang seng': '^HSI',
            'shanghai': '000001.SS',
            'asx': '^AXJO',
            'tsx': '^GSPTSE',
            'cac40': '^FCHI',
            'ibex': '^IBEX',
            'russell': '^RUT',
            'vix': '^VIX'
        }
        
        # Company name mappings
        self.company_names = {
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK NIFTY',
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'RELIANCE.NS': 'Reliance Industries Ltd.',
            'TCS.NS': 'Tata Consultancy Services Ltd.',
            'HDFCBANK.NS': 'HDFC Bank Ltd.',
            'INFY.NS': 'Infosys Ltd.',
            'ADANIPORTS.NS': 'Adani Ports & SEZ Ltd.',
            'ADANIPOWER.NS': 'Adani Power Ltd.',
            'ADANIENT.NS': 'Adani Enterprises Ltd.'
        }
        
    def parse_user_prompt(self, prompt: str) -> List[Tuple[str, str]]:
        """
        Parse user prompt to extract stock/index names
        
        Args:
            prompt: Natural language prompt from user
            
        Returns:
            List of tuples (symbol, company_name)
        """
        prompt_lower = prompt.lower()
        found_stocks = []
        
        # Remove common words that might interfere
        stop_words = ['the', 'will', 'stock', 'price', 'go', 'up', 'down', 'today', 'tomorrow', 
                     'predict', 'prediction', 'analysis', 'tell', 'me', 'about', 'what', 'how',
                     'is', 'are', 'can', 'you', 'please', 'share', 'market', 'trading']
        
        # Check for exact matches in our mappings
        for stock_name, symbol in self.stock_mappings.items():
            if stock_name in prompt_lower:
                company_name = self.company_names.get(symbol, stock_name.title())
                found_stocks.append((symbol, company_name))
        
        # If no matches found, try to extract potential stock symbols
        if not found_stocks:
            # Look for potential stock symbols (2-5 capital letters)
            symbol_pattern = r'\b[A-Z]{2,5}\b'
            potential_symbols = re.findall(symbol_pattern, prompt)
            
            for symbol in potential_symbols:
                if symbol not in stop_words:
                    found_stocks.append((symbol, f"{symbol} Corp"))
        
        # Remove duplicates while preserving order
        unique_stocks = []
        seen = set()
        for stock in found_stocks:
            if stock[0] not in seen:
                unique_stocks.append(stock)
                seen.add(stock[0])
        
        return unique_stocks if unique_stocks else [("^NSEI", "NIFTY 50")]  # Default to NIFTY
    
    def fetch_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """
        Fetch stock data for any symbol
        
        Args:
            symbol: Stock/index symbol
            period: Time period for data
            
        Returns:
            DataFrame with stock data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                # Try adding .NS for Indian stocks if not found
                if not symbol.endswith('.NS') and not symbol.startswith('^'):
                    symbol_ns = f"{symbol}.NS"
                    stock = yf.Ticker(symbol_ns)
                    data = stock.history(period=period)
            
            if not data.empty:
                # Store market data
                self.stock_data[symbol] = {
                    'current_price': data['Close'].iloc[-1],
                    'previous_close': data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1],
                    'volume': data['Volume'].iloc[-1],
                    'high_52w': data['High'].max(),
                    'low_52w': data['Low'].min(),
                    'data': data,
                    'daily_change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                }
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """
        Calculate comprehensive technical indicators
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of technical indicators
        """
        if symbol not in self.stock_data:
            return {}
            
        data = self.stock_data[symbol]['data'].copy()
        
        # Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data['RSI'] = calculate_rsi(data['Close'])
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Stochastic Oscillator
        data['Stoch_K'] = ((data['Close'] - data['Low'].rolling(window=14).min()) / 
                          (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * 100
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Support and Resistance levels
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        # Store latest values
        latest_indicators = {
            'MA_5': data['MA_5'].iloc[-1] if not pd.isna(data['MA_5'].iloc[-1]) else 0,
            'MA_10': data['MA_10'].iloc[-1] if not pd.isna(data['MA_10'].iloc[-1]) else 0,
            'MA_20': data['MA_20'].iloc[-1] if not pd.isna(data['MA_20'].iloc[-1]) else 0,
            'MA_50': data['MA_50'].iloc[-1] if not pd.isna(data['MA_50'].iloc[-1]) else 0,
            'RSI': data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50,
            'MACD': data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0,
            'MACD_Signal': data['MACD_Signal'].iloc[-1] if not pd.isna(data['MACD_Signal'].iloc[-1]) else 0,
            'MACD_Histogram': data['MACD_Histogram'].iloc[-1] if not pd.isna(data['MACD_Histogram'].iloc[-1]) else 0,
            'BB_Position': ((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / 
                           (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])) if not pd.isna(data['BB_Lower'].iloc[-1]) else 0.5,
            'Stoch_K': data['Stoch_K'].iloc[-1] if not pd.isna(data['Stoch_K'].iloc[-1]) else 50,
            'Stoch_D': data['Stoch_D'].iloc[-1] if not pd.isna(data['Stoch_D'].iloc[-1]) else 50,
            'Volume_Ratio': data['Volume_Ratio'].iloc[-1] if not pd.isna(data['Volume_Ratio'].iloc[-1]) else 1,
            'Recent_High': recent_high,
            'Recent_Low': recent_low,
            'Price_Position': (data['Close'].iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        }
        
        return latest_indicators
    
    def generate_market_sentiment(self, symbol: str) -> Dict:
        """
        Generate market sentiment analysis
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment analysis results
        """
        # Simulate news sentiment based on price movement and volume
        if symbol not in self.stock_data:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        daily_change = self.stock_data[symbol]['daily_change']
        volume_data = self.stock_data[symbol]['data']['Volume'].tail(5)
        avg_volume = volume_data.mean()
        current_volume = volume_data.iloc[-1]
        
        # Sentiment based on price movement and volume
        if daily_change > 2 and current_volume > avg_volume * 1.2:
            sentiment = 'very_bullish'
            confidence = 0.9
        elif daily_change > 1:
            sentiment = 'bullish'
            confidence = 0.7
        elif daily_change < -2 and current_volume > avg_volume * 1.2:
            sentiment = 'very_bearish'
            confidence = 0.9
        elif daily_change < -1:
            sentiment = 'bearish'
            confidence = 0.7
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'daily_change': daily_change,
            'volume_factor': current_volume / avg_volume
        }
    
    def llm_reasoning_engine(self, symbol: str, company_name: str) -> Dict:
        """
        Advanced LLM-style reasoning for stock prediction
        
        Args:
            symbol: Stock symbol
            company_name: Company name
            
        Returns:
            Detailed prediction with reasoning
        """
        # Get all analysis data
        stock_data = self.stock_data.get(symbol, {})
        indicators = self.calculate_technical_indicators(symbol)
        sentiment = self.generate_market_sentiment(symbol)
        
        current_price = stock_data.get('current_price', 0)
        daily_change = stock_data.get('daily_change', 0)
        
        # LLM-style reasoning process
        reasoning_steps = []
        score_factors = []
        
        # 1. Technical Analysis Reasoning
        rsi = indicators.get('RSI', 50)
        if rsi > 70:
            reasoning_steps.append(f"🔴 RSI at {rsi:.1f} indicates overbought conditions - potential selling pressure")
            score_factors.append(-0.3)
        elif rsi < 30:
            reasoning_steps.append(f"🟢 RSI at {rsi:.1f} indicates oversold conditions - potential buying opportunity")
            score_factors.append(0.3)
        else:
            reasoning_steps.append(f"🟡 RSI at {rsi:.1f} shows neutral momentum")
            score_factors.append(0.0)
        
        # 2. Moving Average Analysis
        ma_5 = indicators.get('MA_5', 0)
        ma_20 = indicators.get('MA_20', 0)
        if current_price > ma_5 > ma_20:
            reasoning_steps.append(f"🟢 Price above both 5-day and 20-day moving averages - bullish trend")
            score_factors.append(0.2)
        elif current_price < ma_5 < ma_20:
            reasoning_steps.append(f"🔴 Price below both moving averages - bearish trend")
            score_factors.append(-0.2)
        else:
            reasoning_steps.append(f"🟡 Mixed signals from moving averages - consolidation phase")
            score_factors.append(0.0)
        
        # 3. MACD Analysis
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        macd_histogram = indicators.get('MACD_Histogram', 0)
        
        if macd > macd_signal and macd_histogram > 0:
            reasoning_steps.append(f"🟢 MACD shows bullish momentum - histogram trending up")
            score_factors.append(0.25)
        elif macd < macd_signal and macd_histogram < 0:
            reasoning_steps.append(f"🔴 MACD shows bearish momentum - histogram trending down")
            score_factors.append(-0.25)
        else:
            reasoning_steps.append(f"🟡 MACD signals mixed - waiting for clear direction")
            score_factors.append(0.0)
        
        # 4. Bollinger Bands Analysis
        bb_position = indicators.get('BB_Position', 0.5)
        if bb_position > 0.8:
            reasoning_steps.append(f"🔴 Price near upper Bollinger Band - potential resistance")
            score_factors.append(-0.15)
        elif bb_position < 0.2:
            reasoning_steps.append(f"🟢 Price near lower Bollinger Band - potential support")
            score_factors.append(0.15)
        else:
            reasoning_steps.append(f"🟡 Price in middle of Bollinger Bands - neutral zone")
            score_factors.append(0.0)
        
        # 5. Volume Analysis
        volume_ratio = indicators.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            reasoning_steps.append(f"🟢 High volume ({volume_ratio:.1f}x average) confirms price movement")
            score_factors.append(0.1 if daily_change > 0 else -0.1)
        elif volume_ratio < 0.7:
            reasoning_steps.append(f"🔴 Low volume ({volume_ratio:.1f}x average) suggests weak conviction")
            score_factors.append(-0.05)
        else:
            reasoning_steps.append(f"🟡 Normal volume levels")
            score_factors.append(0.0)
        
        # 6. Price Position Analysis
        price_position = indicators.get('Price_Position', 0.5)
        if price_position > 0.8:
            reasoning_steps.append(f"🔴 Price near 20-day high - potential resistance zone")
            score_factors.append(-0.1)
        elif price_position < 0.2:
            reasoning_steps.append(f"🟢 Price near 20-day low - potential support zone")
            score_factors.append(0.1)
        
        # 7. Sentiment Analysis
        market_sentiment = sentiment.get('sentiment', 'neutral')
        if market_sentiment == 'very_bullish':
            reasoning_steps.append(f"🟢 Very bullish market sentiment with strong volume")
            score_factors.append(0.3)
        elif market_sentiment == 'bullish':
            reasoning_steps.append(f"🟢 Bullish market sentiment")
            score_factors.append(0.2)
        elif market_sentiment == 'very_bearish':
            reasoning_steps.append(f"🔴 Very bearish market sentiment with strong volume")
            score_factors.append(-0.3)
        elif market_sentiment == 'bearish':
            reasoning_steps.append(f"🔴 Bearish market sentiment")
            score_factors.append(-0.2)
        else:
            reasoning_steps.append(f"🟡 Neutral market sentiment")
            score_factors.append(0.0)
        
        # 8. Calculate Overall Score
        overall_score = sum(score_factors) / len(score_factors) if score_factors else 0
        
        # 9. Generate Prediction
        base_volatility = 0.015  # 1.5% base daily volatility
        predicted_change = overall_score * base_volatility * 2  # Amplify for prediction
        predicted_price = current_price * (1 + predicted_change)
        
        # 10. Determine Direction and Confidence
        if predicted_change > 0.01:  # > 1%
            direction = "UP ⬆️"
            direction_simple = "UP"
            confidence = min(0.9, abs(overall_score) + 0.5)
        elif predicted_change < -0.01:  # < -1%
            direction = "DOWN ⬇️"
            direction_simple = "DOWN"
            confidence = min(0.9, abs(overall_score) + 0.5)
        else:
            direction = "SIDEWAYS ↔️"
            direction_simple = "SIDEWAYS"
            confidence = 0.6
        
        # 11. Risk Assessment
        risk_level = "LOW"
        if abs(predicted_change) > 0.03:  # > 3%
            risk_level = "HIGH"
        elif abs(predicted_change) > 0.015:  # > 1.5%
            risk_level = "MEDIUM"
        
        # 12. Final Reasoning Summary
        reasoning_steps.append(f"📊 Overall Technical Score: {overall_score:.2f}")
        reasoning_steps.append(f"🎯 Prediction: {direction} with {confidence*100:.0f}% confidence")
        reasoning_steps.append(f"⚠️ Risk Level: {risk_level}")
        
        return {
            'predicted_price': predicted_price,
            'predicted_change_percent': predicted_change * 100,
            'direction': direction_simple,
            'direction_emoji': direction,
            'confidence': confidence,
            'risk_level': risk_level,
            'overall_score': overall_score,
            'reasoning_steps': reasoning_steps,
            'technical_factors': {
                'rsi_score': score_factors[0] if len(score_factors) > 0 else 0,
                'ma_score': score_factors[1] if len(score_factors) > 1 else 0,
                'macd_score': score_factors[2] if len(score_factors) > 2 else 0,
                'sentiment_score': score_factors[-1] if score_factors else 0
            }
        }
    
    def generate_response(self, symbol: str, company_name: str) -> str:
        """
        Generate natural language response
        
        Args:
            symbol: Stock symbol
            company_name: Company name
            
        Returns:
            Natural language response
        """
        prediction = self.llm_reasoning_engine(symbol, company_name)
        stock_data = self.stock_data.get(symbol, {})
        
        current_price = stock_data.get('current_price', 0)
        daily_change = stock_data.get('daily_change', 0)
        
        # Format price based on market (Indian vs Global)
        if symbol.endswith('.NS') or symbol.startswith('^NSE') or symbol.startswith('^BSE'):
            price_format = f"₹{current_price:.2f}"
            predicted_price_format = f"₹{prediction['predicted_price']:.2f}"
        else:
            price_format = f"${current_price:.2f}"
            predicted_price_format = f"${prediction['predicted_price']:.2f}"
        
        response = f"""
🤖 AI Analysis for {company_name} ({symbol})

📊 CURRENT STATUS:
• Price: {price_format}
• Today's Change: {daily_change:+.2f}%
• Market Status: {'🟢 Bullish' if daily_change > 0 else '🔴 Bearish' if daily_change < 0 else '🟡 Neutral'}

🔮 TOMORROW'S PREDICTION:
• Direction: {prediction['direction_emoji']}
• Predicted Price: {predicted_price_format}
• Expected Change: {prediction['predicted_change_percent']:+.2f}%
• Confidence: {prediction['confidence']*100:.0f}%
• Risk Level: {prediction['risk_level']}

🧠 AI REASONING:
"""
        
        for i, step in enumerate(prediction['reasoning_steps'], 1):
            response += f"{i}. {step}\n"
        
        response += f"""
💡 TRADING SUGGESTION:
"""
        
        if prediction['direction'] == 'UP':
            response += "• Consider LONG positions with proper risk management\n"
            response += "• Set stop-loss below recent support levels\n"
            response += "• Monitor for profit-taking at resistance levels\n"
        elif prediction['direction'] == 'DOWN':
            response += "• Consider SHORT positions or protect existing longs\n"
            response += "• Wait for better entry points on pullbacks\n"
            response += "• Set stop-loss above recent resistance levels\n"
        else:
            response += "• HOLD current positions and wait for clarity\n"
            response += "• Consider range-bound trading strategies\n"
            response += "• Monitor for breakout signals\n"
        
        response += f"""
⚠️ DISCLAIMER: This is an AI-generated prediction based on technical analysis. 
Always do your own research and consider multiple factors before trading.

📈 Overall AI Confidence: {prediction['confidence']*100:.0f}% | Risk: {prediction['risk_level']}
"""
        
        return response
    
    def process_prompt(self, user_prompt: str) -> str:
        """
        Main function to process user prompts and generate predictions
        
        Args:
            user_prompt: Natural language prompt from user
            
        Returns:
            AI response with prediction
        """
        print(f"🤖 Processing prompt: '{user_prompt}'")
        
        # Parse the prompt to extract stock symbols
        stocks_found = self.parse_user_prompt(user_prompt)
        
        if not stocks_found:
            return "❌ I couldn't identify any specific stock or index in your prompt. Please mention a stock name or symbol."
        
        responses = []
        
        for symbol, company_name in stocks_found:
            print(f"📊 Analyzing {symbol} ({company_name})...")
            
            # Fetch stock data
            stock_data = self.fetch_stock_data(symbol)
            
            if stock_data.empty:
                responses.append(f"❌ Sorry, I couldn't fetch data for {symbol}. Please check the symbol or try a different one.")
                continue
            
            # Generate prediction and response
            try:
                response = self.generate_response(symbol, company_name)
                responses.append(response)
            except Exception as e:
                responses.append(f"❌ Error analyzing {symbol}: {str(e)}")
        
        return "\n" + "="*80 + "\n".join(responses) + "="*80

# Interactive AI Chat Interface
class StockPredictionChatBot:
    def __init__(self):
        self.ai = UniversalStockPredictionAI()
        self.chat_history = []
        
    def start_chat(self):
        """Start interactive chat session"""
        print("🚀 Universal Stock Prediction AI - Chat Interface")
        print("="*60)
        print("💬 Ask me about any stock or index!")
        print("📝 Examples:")
        print("   • 'Tell me will the NIFTY go up or down'")
        print("   • 'What about Apple stock tomorrow?'")
        print("   • 'Adani Power prediction'")
        print("   • 'Should I buy Tesla?'")
        print("   • 'Analysis of Reliance stock'")
        print("   • 'Sensex outlook'")
        print("\n💡 Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("🗣️ You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("👋 Thank you for using Stock Prediction AI! Happy trading!")
                    break
                
                if not user_input:
                    print("🤔 Please ask me something about stocks or markets!")
                    continue
                
                # Process the prompt
                print("🤖 AI: Analyzing your request...")
                response = self.ai.process_prompt(user_input)
                
                print("🤖 AI:", response)
                
                # Store in chat history
                self.chat_history.append({
                    'user': user_input,
                    'ai': response,
                    'timestamp': datetime.now()
                })
                
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using Stock Prediction AI!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("🔄 Please try again with a different query.")

# Batch Processing Function
def batch_analyze_stocks(stock_list: List[str]) -> Dict:
    """
    Analyze multiple stocks in batch
    
    Args:
        stock_list: List of stock names or symbols
        
    Returns:
        Dictionary with analysis results
    """
    ai = UniversalStockPredictionAI()
    results = {}
    
    for stock in stock_list:
        try:
            response = ai.process_prompt(f"analyze {stock}")
            results[stock] = response
        except Exception as e:
            results[stock] = f"Error analyzing {stock}: {str(e)}"
    
    return results

# Demo and Testing Functions
def run_demo():
    """Run demonstration of the AI system"""
    ai = UniversalStockPredictionAI()
    
    demo_prompts = [
        "Tell me will the NIFTY go up or down",
        "What about Apple stock tomorrow?",
        "Adani Power prediction",
        "Should I buy Tesla?",
        "Analysis of Reliance stock",
        "Sensex outlook for tomorrow",
        "Microsoft prediction",
        "Bank NIFTY direction",
        "TCS stock analysis",
        "Google stock tomorrow"
    ]
    
    print("🎯 DEMO: Stock Prediction AI - Natural Language Processing")
    print("="*70)
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n📝 Demo Query {i}: '{prompt}'")
        print("🤖 AI Response:")
        try:
            response = ai.process_prompt(prompt)
            print(response)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*70)
        
        # Add small delay for demo effect
        import time
        time.sleep(1)

def test_stock_parsing():
    """Test the stock parsing functionality"""
    ai = UniversalStockPredictionAI()
    
    test_prompts = [
        "NIFTY prediction",
        "Apple stock analysis",
        "What about TSLA?",
        "Reliance Industries outlook",
        "HDFC Bank tomorrow",
        "Sensex and Bank NIFTY",
        "Microsoft and Google stocks",
        "Adani Power and Adani Enterprises",
        "TCS Infosys Wipro analysis",
        "SP500 and NASDAQ prediction"
    ]
    
    print("🧪 Testing Stock Symbol Parsing")
    print("="*50)
    
    for prompt in test_prompts:
        stocks = ai.parse_user_prompt(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Parsed: {stocks}")
        print("-" * 30)

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo()
        elif sys.argv[1] == "test":
            test_stock_parsing()
        elif sys.argv[1] == "batch":
            # Batch analysis example
            stocks = ["NIFTY", "Apple", "Tesla", "Reliance", "TCS"]
            results = batch_analyze_stocks(stocks)
            for stock, result in results.items():
                print(f"\n{stock}:\n{result}")
        else:
            # Single prompt processing
            prompt = " ".join(sys.argv[1:])
            ai = UniversalStockPredictionAI()
            response = ai.process_prompt(prompt)
            print(response)
    else:
        # Interactive chat mode
        chatbot = StockPredictionChatBot()
        chatbot.start_chat()