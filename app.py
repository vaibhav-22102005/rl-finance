from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from qnn_model import HybridSequenceVQC  
app = Flask(__name__)

TICKERS = ['TITAN.NS', 'RELIANCE.NS']

@app.route('/')
def index():
    return render_template('index.html', tickers=TICKERS)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    selected_tickers = data.get('tickers', [])

    period = "1y"  
    
    response_data = []

    qnn = HybridSequenceVQC() 

    for ticker in selected_tickers:
        try:
            print(f"Fetching data for {ticker}...")
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty: continue
            

            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if ticker in df['Close'].columns:
                        prices_series = df['Close'][ticker]
                    else:
                        prices_series = df['Close'].iloc[:, 0]
                else:
                    prices_series = df['Close']
                prices = prices_series.values.flatten()
            except Exception as e:
                print(f"Data error {ticker}: {e}")
                continue

            if len(prices) < 60: continue


            dates = df.index[-30:].strftime('%Y-%m-%d').tolist()
            

            q_probs, c_probs = qnn.get_analysis(prices, lookback_days=30)
            

            graph_len = min(len(q_probs), 30)
            q_probs = q_probs[-graph_len:]
            c_probs = c_probs[-graph_len:]
            dates = dates[-graph_len:]
            
            if len(q_probs) > 0:
                current_q = q_probs[-1]
                current_c = c_probs[-1]
                current_price = float(prices[-1])

                # --- Accuracy Calculation ---
                correct = 0
                for i in range(len(q_probs) - 1):
                    pred_dir = 1 if q_probs[i] > 0.5 else -1
                   
                    idx_in_prices = -len(q_probs) + i
                    actual_ret = prices[idx_in_prices + 1] - prices[idx_in_prices]
                    actual_dir = 1 if actual_ret > 0 else -1
                    if pred_dir == actual_dir: correct += 1
                
                acc_pct = (correct / (len(q_probs)-1) * 100) if len(q_probs) > 1 else 0

                # --- Prediction Logic ---
                daily_vol = pd.Series(prices).pct_change().std()
                if pd.isna(daily_vol): daily_vol = 0.01
                
                direction = 1 if current_q > 0.5 else -1
                pred_price = current_price * (1 + (direction * daily_vol))

                response_data.append({
                    'ticker': ticker,
                    'dates': dates,
                    'history': {'quantum': q_probs, 'classical': c_probs},
                    'latest': {
                        'quantum': round(current_q, 3),
                        'classical': round(current_c, 3),
                        'price': round(current_price, 2),
                        'predicted_price': round(pred_price, 2),
                        'accuracy': round(acc_pct, 1)
                    }
                })
                
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)