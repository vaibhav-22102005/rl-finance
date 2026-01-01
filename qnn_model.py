import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

algorithm_globals.random_seed = 42
np.random.seed(42)

class HybridSequenceVQC:
    def __init__(self):
        
        self.reservoir_dim = 8  
        self.spectral_radius = 0.95
        self.sparsity = 0.2
        
        
        self.W_in = None  
        self.W_res = None 

        
        self.num_qubits = 4 
        
        
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=2, entanglement='linear')
        
        
        self.ansatz = EfficientSU2(num_qubits=self.num_qubits, reps=3, entanglement='full')
        
        
        self.optimizer = L_BFGS_B(maxiter=50)
        
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.pca = PCA(n_components=self.num_qubits)

    def _init_reservoir(self, input_dim):
        """Initializes the Echo State Network (Sequence Model) weights"""
        
        self.W_in = np.random.rand(self.reservoir_dim, input_dim) * 2 - 1
        
        
        W = np.random.rand(self.reservoir_dim, self.reservoir_dim) - 0.5
        
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = W * (self.spectral_radius / radius)

    def run_reservoir(self, X_sequence):
        """
        Passes the time-series data through the Recurrent Reservoir.
        X_sequence shape: (Time_Steps, Features)
        Returns: The final 'Hidden State' vector capturing the sequence history.
        """
        
        input_dim = X_sequence.shape[1]
        if self.W_in is None:
            self._init_reservoir(input_dim)
            
        
        state = np.zeros(self.reservoir_dim)
        
        
        for t in range(len(X_sequence)):
            u = X_sequence[t]

            state = np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, state))
            
        return state

    def prepare_data(self, prices, lookback_days=30, is_training=False):
        df = pd.DataFrame(prices, columns=['Close'])
        
        
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Return'].rolling(window=5).std()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MA_Dist'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
        
        df = df.dropna()
        
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        
        
        if is_training:
            mask = df['Return'].shift(-1).abs() > 0.002
            df = df[mask]

        if len(df) < lookback_days + 5:
            return None, None

        feature_cols = ['Return', 'Volatility', 'RSI', 'MA_Dist']
        data_values = df[feature_cols].values
        labels = df['Target'].values
        
        
        processed_X = []
        valid_y = []
        
        
        for i in range(lookback_days, len(data_values)):
            
            seq_window = data_values[i-lookback_days : i]
            
            
            reservoir_state = self.run_reservoir(seq_window)
            
            processed_X.append(reservoir_state)
            valid_y.append(labels[i]) 
            
        return np.array(processed_X), np.array(valid_y)

    def calculate_rsi(self, data, window=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_analysis(self, prices, lookback_days=30):
        try:
            if len(prices) < 60: return [], []

            
            X_seq, y = self.prepare_data(prices, lookback_days, is_training=True)
            if X_seq is None or len(X_seq) < 20: return [], []
            

            X_pca = self.pca.fit_transform(X_seq)
            
            
            X_scaled = self.scaler.fit_transform(X_pca)
            
            split = int(len(X_scaled) * 0.9)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y[:split], y[split:]
            
            if len(X_test) == 0: 
                X_test = X_train[-10:] 
                y_test = y_train[-10:]

            
            vqc = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer
            )
            
            vqc.fit(X_train, y_train)
            
        
            pred_results = vqc.predict_proba(X_test)
            q_probs = [float(p[1]) for p in pred_results]
            
            
            c_probs = [float(np.mean(y_train[-5:])) for _ in range(len(q_probs))]

            return q_probs, c_probs

        except Exception as e:
            print(f"Hybrid Model Error: {e}")
            import traceback
            traceback.print_exc()
            return [], []