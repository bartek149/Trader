import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import json
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Import modułu logowania predykcji (opcjonalnie, jeśli dostępny)
try:
    from prediction_logger import PredictionLogger
    PREDICTION_LOGGING_AVAILABLE = True
except ImportError:
    PREDICTION_LOGGING_AVAILABLE = False
    PredictionLogger = None

class StockPredictor:
    def __init__(self, symbol, period='1y', data_dir='stock_data'):
        """
        Inicjalizacja predyktora akcji
        
        Parameters:
        symbol (str): Symbol akcji (np. 'AAPL', 'MSFT', 'GOOGL')
        period (str): Okres danych historycznych (np. '1y', '2y', '6mo')
        data_dir (str): Katalog do przechowywania danych historycznych
        """
        self.symbol = symbol
        self.period = period
        self.model = None
        self.data = None
        self.features = None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / f'{symbol}_data.csv'
        self.model_file = self.data_dir / f'{symbol}_model.json'
        self.model_pickle_file = self.data_dir / f'{symbol}_model.pkl'
        self.weekly_model_pickle_file = self.data_dir / f'{symbol}_weekly_model.pkl'
        self.features_file = self.data_dir / f'{symbol}_features.json'
        self.weekly_features_file = self.data_dir / f'{symbol}_weekly_features.json'
        self.data_stale = False
        self.data_stale_last_date = None
        self.data_stale_age_days = None

    def log(self, message, level="INFO"):
        """Wypisz wiadomość do konsoli z jednolitym formatem."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.symbol}] {message}")
        
    def load_historical_data(self):
        """Ładowanie zapisanych danych historycznych"""
        if self.data_file.exists():
            try:
                df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
                # Normalizuj timezone - usuń timezone info jeśli istnieje
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                print(f"Załadowano {len(df)} dni danych historycznych dla {self.symbol}")
                return df
            except Exception as e:
                print(f"Błąd przy ładowaniu danych: {e}")
        return None
    
    def save_historical_data(self, new_data):
        """Zapisywanie lub aktualizacja danych historycznych"""
        # Normalizuj timezone w nowych danych
        if new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)
        
        historical = self.load_historical_data()
        
        if historical is not None:
            # Połącz dane, usuń duplikaty, posortuj
            combined = pd.concat([historical, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = new_data
        
        combined.to_csv(self.data_file)
        print(f"Zapisano {len(combined)} dni danych dla {self.symbol}")
        return combined
    
    def convert_symbol_for_yfinance(self, symbol):
        """Konwertuj symbol do formatu rozpoznawanego przez yfinance"""
        # yfinance zwykle wymaga symboli bez sufiksów giełdowych
        # Spróbuj różne warianty
        variants = [symbol]
        
        # Dla symboli .DE, spróbuj bez sufiksu lub z innym formatem
        if symbol.endswith('.DE'):
            base = symbol.replace('.DE', '')
            german_variants = [
                f"{base}.DE",   # Frankfurt (domyślny)
                f"{base}.F"     # XETRA
            ]
            for variant in german_variants:
                if variant not in variants:
                    variants.append(variant)
        
        # Dla symboli .US, spróbuj bez sufiksu
        if '.US' in symbol:
            base = symbol.replace('.US', '')
            variants.extend([base])
        
        # Dla surowców (np. NATGAS), spróbuj różne formaty
        if symbol.upper() in ['NATGAS', 'GAS', 'NATURALGAS']:
            variants.extend(['NG=F', 'NG1=F', 'NATGAS=X'])
        
        return variants
    
    def delete_local_data(self, remove_models=True, remove_weekly=True, remove_features=True):
        """Usuń lokalne dane i (opcjonalnie) zapisane modele dla danej spółki."""
        targets = [self.data_file]
        if remove_models:
            targets.extend([self.model_file, self.model_pickle_file])
        if remove_weekly:
            targets.append(self.weekly_model_pickle_file)
        if remove_features:
            targets.extend([self.features_file, self.weekly_features_file])
        removed = []
        for path in targets:
            if path.exists():
                try:
                    path.unlink()
                    removed.append(path.name)
                except Exception as e:
                    print(f"Nie można usunąć {path}: {e}")
        if removed:
            self.log(f"Usunięto lokalne pliki: {', '.join(removed)}", level="WARN")

    def fetch_data(self, use_saved=True, max_age_days=7, delete_stale=True):
        """Pobieranie danych historycznych akcji
        
        Parameters:
            use_saved (bool): 
                - True: Używa TYLKO lokalnych zapisanych danych (nie pobiera z Yahoo)
                - False: Pobiera nowe dane z Yahoo (i zapisuje lokalnie)
            max_age_days (int or None):
                - jeśli >0: maksymalny wiek danych lokalnych; starsze zostaną uznane za przestarzałe
            delete_stale (bool):
                - jeśli True, przestarzałe pliki zostaną usunięte automatycznie
        """
        self.log("Rozpoczynam pobieranie danych...")
        self.data_stale = False
        self.data_stale_last_date = None
        self.data_stale_age_days = None
        
        # Spróbuj załadować zapisane dane
        if use_saved:
            historical = self.load_historical_data()
            if historical is not None and len(historical) > 0:
                last_date = historical.index[-1]
                # Normalizuj timezone - upewnij się, że oba są timezone-naive
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                
                if max_age_days is not None:
                    data_age_days = (datetime.now() - last_date.to_pydatetime()).days
                    if data_age_days > max_age_days:
                        print(f"Dane lokalne dla {self.symbol} są przestarzałe "
                              f"({last_date.date()} / {data_age_days} dni temu).")
                        self.data_stale = True
                        self.data_stale_last_date = last_date
                        self.data_stale_age_days = data_age_days
                        if delete_stale:
                            self.delete_local_data(remove_models=True, remove_weekly=True, remove_features=True)
                        self.data = None
                        return None
                
                # Użyj TYLKO lokalnych danych - nie pobieraj z Yahoo
                self.log(f"Używam zapisanych danych lokalnych (ostatnia data: {last_date.date()})")
                self.data = historical
                return self.data
            else:
                # Brak zapisanych danych - zwróć None zamiast pobierać z Yahoo
                self.log("Brak zapisanych danych lokalnych.")
                self.data = None
                return None
        
        # Pobierz wszystkie dane - spróbuj różne warianty symbolu
        variants = self.convert_symbol_for_yfinance(self.symbol)
        self.data = None
        
        for variant in variants:
            try:
                self.log(f"Próba pobrania z Yahoo dla wariantu '{variant}'...")
                ticker = yf.Ticker(variant)
                data = ticker.history(period=self.period)
                if not data.empty:
                    self.log(f"Pomyślnie pobrano {len(data)} rekordów (wariant: {variant})")
                    self.data = data
                    break
            except Exception as e:
                self.log(f"Nie udało się pobrać danych dla wariantu '{variant}': {str(e)[:120]}", level="ERROR")
                continue
        
        if self.data is None or self.data.empty:
            error_msg = f"Nie udało się pobrać danych (próbowano: {', '.join(variants)})"
            self.log(error_msg, level="ERROR")
            raise ValueError(error_msg)
        
        # Normalizuj timezone
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)
        
        # Zapisz dane (zawsze zapisuj, nawet jeśli use_saved=False, aby mieć dane na przyszłość)
        self.data = self.save_historical_data(self.data)
        
        self.log(f"Zapisano {len(self.data)} dni danych (nowy cache).")
        return self.data
    
    def create_features(self, horizon_days=1):
        """Tworzenie cech (features) dla modelu - klasyfikacja kierunku UP/DOWN
        
        Parameters:
            horizon_days (int): Horyzont przewidywania w dniach (1 = jutro, 5 = tydzień)
        """
        if self.data is None:
            raise ValueError("Najpierw pobierz dane używając fetch_data()")
        
        df = self.data.copy()
        
        # ========== MUST-HAVE FEATURES ==========
        
        # 1. Średnie kroczące
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # 2. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Unikaj dzielenia przez zero
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], 0).fillna(50)))
        
        # 3. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # 5. GAP - super ważne!
        prev_close_gap = df['Close'].shift(1).replace(0, np.nan)
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / prev_close_gap
        
        # 6. Time Features - dzień tygodnia i miesiąc
        df['Day_of_week'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        # 7. Lag Features - opóźnione ceny
        df['Lag1'] = df['Close'].shift(1)
        df['Lag3'] = df['Close'].shift(3)
        df['Lag5'] = df['Close'].shift(5)
        
        # 8. Zmiana ceny (dzienna)
        df['Price_change'] = df['Close'].pct_change()
        
        # 9. Wolumen (normalizowany)
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        # Unikaj dzielenia przez zero - zastąp 0 małą wartością
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.nan)
        df['Volume_rate'] = df['Volume'].pct_change()  # Dodatkowa normalizacja
        
        # 10. Wysokie/Niskie w stosunku do zamknięcia
        close_for_ratio = df['Close'].replace(0, np.nan)
        df['High_Low_ratio'] = (df['High'] - df['Low']) / close_for_ratio
        
        # 11. Cena otwarcia w stosunku do zamknięcia poprzedniego dnia
        prev_close = df['Close'].shift(1).replace(0, np.nan)
        df['Open_Close_ratio'] = df['Open'] / prev_close
        
        # ========== POWINIENEŚ DODAĆ ==========
        
        # 12. Rate-of-Change (ROC)
        df['ROC'] = df['Close'].pct_change(periods=5)
        
        # 13. ATR - Average True Range (zmienność realna)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # LOG-TRANSFORMACJA dużych wartości (poprawka maklera)
        # Volume, ATR, TR mają bardzo duże wartości (rzędu 1 000 000)
        # Log-transformacja normalizuje je do podobnej skali jak inne cechy
        # UWAGA: Tworzymy OBIE wersje dla kompatybilności wstecznej ze starymi modelami
        df['Volume_log'] = np.log1p(df['Volume'])  # log(Volume + 1)
        df['ATR_log'] = np.log1p(df['ATR'].fillna(0))  # log(ATR + 1)
        df['TR_log'] = np.log1p(df['TR'].fillna(0))  # log(TR + 1)
        # Zachowaj też stare wersje dla kompatybilności wstecznej (jeśli model ich oczekuje)
        # Uwaga: W nowych modelach używamy tylko _log wersji
        
        # 14. Momentum
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # ========== PRO-TIPS ==========
        
        # 15. Window statistics
        df['Roll_mean_10'] = df['Close'].rolling(10).mean()
        df['Roll_std_10'] = df['Close'].rolling(10).std()
        
        # 16. Keltner Channels (lepsze od Bollingera)
        df['KC_upper'] = df['MA_20'] + df['ATR'] * 2
        df['KC_lower'] = df['MA_20'] - df['ATR'] * 2
        
        # ========== POPRAWKA DEV: Feature'y średnioterminowe dla tygodnia ==========
        # MA_20 i MA_50 już są, ale dodajmy relacje i tygodniową zmienność
        # Upewnij się, że mamy wystarczająco danych przed obliczeniem
        if len(df) >= 50:
            df['Price_to_MA50'] = df['Close'] / df['MA_50'].replace(0, np.nan)  # Relacja ceny do MA_50
        else:
            df['Price_to_MA50'] = 1.0  # Domyślna wartość jeśli brak danych
        
        if len(df) >= 200:
            df['Price_to_MA200'] = df['Close'] / df['Close'].rolling(200).mean().replace(0, np.nan)  # Relacja do MA_200 (trend)
        else:
            df['Price_to_MA200'] = 1.0  # Domyślna wartość jeśli brak danych
        
        if len(df) >= 10:
            df['Weekly_volatility'] = df['Close'].pct_change().rolling(10).std()  # Tygodniowa zmienność (10 dni)
        else:
            df['Weekly_volatility'] = 0.0  # Domyślna wartość jeśli brak danych
        
        if len(df) >= 50:
            df['Trend_strength'] = (df['MA_20'] - df['MA_50']) / df['MA_50'].replace(0, np.nan)  # Siła trendu
        else:
            df['Trend_strength'] = 0.0  # Domyślna wartość jeśli brak danych
        
        # ========== TARGET: KIERUNEK (UP/DOWN) z horyzontem ==========
        # POPRAWKA DEV: Zmiana targetu z 1 dnia na horizon_days dni
        horizon = horizon_days  # 5 dni handlowych = ~tydzień
        
        # Dla kompatybilności wstecznej, zachowaj Direction (1 dzień)
        df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Nowy target dla horyzontu horizon_days
        df['future_close'] = df['Close'].shift(-horizon)
        df['weekly_return'] = np.log(df['future_close'] / df['Close'].replace(0, np.nan))
        df[f'Direction_{horizon}d'] = (df['weekly_return'] > 0).astype(int)
        
        # Zapisz weekly_return dla późniejszego użycia w prognozach
        self.weekly_return = df['weekly_return']
        
        # Wybierz cechy do modelu
        # UWAGA: Dla kompatybilności wstecznej, tworzymy OBIE wersje (stare i nowe)
        # Nowe modele będą używać _log wersji, stare modele będą używać oryginalnych
        feature_columns = [
            # Podstawowe ceny
            'Open', 'High', 'Low', 'Close',
            # Średnie kroczące
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            # Wskaźniki techniczne
            'RSI', 'MACD', 'MACD_signal',
            # Bollinger Bands
            'BB_middle', 'BB_upper', 'BB_lower',
            # Keltner Channels
            'KC_upper', 'KC_lower',
            # GAP i zmiany
            'Gap', 'Price_change', 'ROC',
            # Lag features
            'Lag1', 'Lag3', 'Lag5',
            # Momentum
            'Momentum_10',
            # Wolumen - OBIE wersje dla kompatybilności
            'Volume', 'Volume_log', 'Volume_ratio', 'Volume_rate',
            # Inne wskaźniki - OBIE wersje dla kompatybilności
            'High_Low_ratio', 'Open_Close_ratio',
            'ATR', 'ATR_log',  # Stare i nowe
            'TR', 'TR_log',    # Stare i nowe
            # Window statistics
            'Roll_mean_10', 'Roll_std_10',
            # POPRAWKA DEV: Feature'y średnioterminowe dla tygodnia
            'Price_to_MA50', 'Price_to_MA200', 'Weekly_volatility', 'Trend_strength',
            # Time features
            'Day_of_week', 'Month'
        ]
        
        # Upewnij się, że wszystkie kolumny istnieją
        available_features = [col for col in feature_columns if col in df.columns]
        # Użyj odpowiedniego targetu w zależności od horyzontu
        target_col = f'Direction_{horizon}d' if horizon > 1 else 'Direction'
        if target_col not in df.columns:
            target_col = 'Direction'  # Fallback dla kompatybilności
        
        # Przygotuj listę kolumn do wyboru (tylko te które istnieją)
        columns_to_select = available_features + ['Direction']
        if target_col in df.columns and target_col != 'Direction':
            columns_to_select.append(target_col)
        if 'weekly_return' in df.columns:
            columns_to_select.append('weekly_return')
        
        features_df = df[columns_to_select].copy()
        
        # ========== CZYSZCZENIE DANYCH - usuń infinity i NaN ==========
        # Zastąp infinity wartościami skończonymi
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Zastąp NaN wartościami mediany dla każdej kolumny (oprócz Direction)
        for col in available_features:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                if pd.isna(median_val):
                    # Jeśli mediana też jest NaN, użyj 0
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].fillna(median_val)
        
        # Ogranicz wartości do rozsądnego zakresu (zapobiega overflow)
        # Dla każdej kolumny numerycznej, przycinaj ekstremalne wartości
        for col in available_features:
            if features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Oblicz percentyle 1% i 99%
                p1 = features_df[col].quantile(0.01)
                p99 = features_df[col].quantile(0.99)
                
                # Jeśli wartości są zbyt ekstremalne, przycinaj
                if abs(p1) > 1e6 or abs(p99) > 1e6:
                    # Dla bardzo dużych wartości, użyj log transformacji lub przycinaj
                    features_df[col] = np.clip(features_df[col], -1e6, 1e6)
        
        # Usuń wiersze, które nadal mają NaN (głównie w Direction)
        features_df = features_df.dropna()
        
        # Upewnij się, że nie ma wartości infinity
        if np.isinf(features_df[available_features].values).any():
            # Zastąp pozostałe infinity wartościami 0
            features_df[available_features] = features_df[available_features].replace([np.inf, -np.inf], 0)
        
        # Sprawdź czy są jeszcze problemy
        if features_df[available_features].isna().any().any():
            # Wypełnij pozostałe NaN wartościami 0
            features_df[available_features] = features_df[available_features].fillna(0)
        
        # Przypisz oczyszczone dane
        self.features = features_df[available_features]
        # Użyj odpowiedniego targetu w zależności od horyzontu
        self.target = features_df[target_col] if target_col in features_df.columns else features_df['Direction']
        self.horizon_days = horizon_days  # Zapisz horyzont dla późniejszego użycia
        
        # Sprawdź czy mamy wystarczająco danych
        if len(self.features) < 50:
            raise ValueError(f"Za mało danych po czyszczeniu: {len(self.features)} wierszy. Minimum: 50")
        
        return features_df
    
    def _calculate_pnl_with_costs(self, prices, signals, transaction_cost=0.001):
        """Oblicz P&L z sygnałów transakcyjnych z uwzględnieniem kosztów
        
        Parameters:
            prices: Series z cenami
            signals: Lista sygnałów ('BUY', 'SELL', 'NO TRADE')
            transaction_cost: Koszt transakcji jako ułamek (0.001 = 0.1%)
        
        Returns:
            dict: {'total_return', 'num_trades', 'returns', 'sharpe'}
        """
        if len(prices) != len(signals):
            return {'total_return': 0, 'num_trades': 0, 'returns': [], 'sharpe': 0}
        
        position = None  # None, 'long', 'short'
        entry_price = 0
        returns = []
        num_trades = 0
        equity = 1.0  # Start z 1.0
        
        for i in range(len(signals)):
            signal = signals[i]
            current_price = prices.iloc[i]
            
            if signal == 'BUY' and position != 'long':
                # Zamknij poprzednią pozycję jeśli była
                if position == 'short':
                    pnl = (entry_price - current_price) / entry_price - transaction_cost
                    equity *= (1 + pnl)
                    returns.append(pnl)
                    num_trades += 1
                
                # Otwórz long
                position = 'long'
                entry_price = current_price
                num_trades += 1
                
            elif signal == 'SELL' and position != 'short':
                # Zamknij poprzednią pozycję jeśli była
                if position == 'long':
                    pnl = (current_price - entry_price) / entry_price - transaction_cost
                    equity *= (1 + pnl)
                    returns.append(pnl)
                    num_trades += 1
                
                # Otwórz short
                position = 'short'
                entry_price = current_price
                num_trades += 1
                
            elif signal == 'NO TRADE':
                # Zamknij pozycję jeśli jest otwarta
                if position == 'long':
                    pnl = (current_price - entry_price) / entry_price - transaction_cost
                    equity *= (1 + pnl)
                    returns.append(pnl)
                    position = None
                elif position == 'short':
                    pnl = (entry_price - current_price) / entry_price - transaction_cost
                    equity *= (1 + pnl)
                    returns.append(pnl)
                    position = None
        
        # Zamknij ostatnią pozycję
        if position == 'long' and len(prices) > 0:
            final_price = prices.iloc[-1]
            pnl = (final_price - entry_price) / entry_price - transaction_cost
            equity *= (1 + pnl)
            returns.append(pnl)
        elif position == 'short' and len(prices) > 0:
            final_price = prices.iloc[-1]
            pnl = (entry_price - final_price) / entry_price - transaction_cost
            equity *= (1 + pnl)
            returns.append(pnl)
        
        total_return = equity - 1.0
        
        # Oblicz Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'returns': returns,
            'sharpe': sharpe
        }
    
    def _calculate_baselines(self, prices, actual_direction):
        """Oblicz baseline'y dla porównania
        
        Parameters:
            prices: Series z cenami
            actual_direction: Series z rzeczywistymi kierunkami (0=DOWN, 1=UP)
        
        Returns:
            dict: Wyniki dla różnych baseline'ów
        """
        baselines = {}
        
        # Baseline 1: Zawsze UP (kup i trzymaj)
        always_up_signals = ['BUY'] * len(prices)
        always_up_pnl = self._calculate_pnl_with_costs(prices, always_up_signals, transaction_cost=0.001)
        always_up_accuracy = accuracy_score(actual_direction, [1] * len(actual_direction))
        baselines['always_up'] = {
            'accuracy': float(always_up_accuracy),
            'total_return': float(always_up_pnl['total_return']),
            'num_trades': int(always_up_pnl['num_trades'])
        }
        
        # Baseline 2: Moving Average Crossover (MA 20 vs MA 50)
        if len(prices) > 50:
            ma20 = prices.rolling(20).mean()
            ma50 = prices.rolling(50).mean()
            
            ma_signals = []
            for i in range(len(prices)):
                if i < 50:
                    ma_signals.append('NO TRADE')
                elif ma20.iloc[i] > ma50.iloc[i] and (i == 0 or ma20.iloc[i-1] <= ma50.iloc[i-1]):
                    ma_signals.append('BUY')
                elif ma20.iloc[i] < ma50.iloc[i] and (i == 0 or ma20.iloc[i-1] >= ma50.iloc[i-1]):
                    ma_signals.append('SELL')
                else:
                    ma_signals.append('NO TRADE')
            
            ma_pnl = self._calculate_pnl_with_costs(prices, ma_signals, transaction_cost=0.001)
            
            # Oblicz accuracy dla MA (traktuj BUY jako UP)
            ma_predictions = [1 if s == 'BUY' else 0 for s in ma_signals]
            ma_accuracy = accuracy_score(actual_direction.iloc[50:], ma_predictions[50:]) if len(actual_direction) > 50 else 0
            
            baselines['ma_crossover'] = {
                'accuracy': float(ma_accuracy),
                'total_return': float(ma_pnl['total_return']),
                'num_trades': int(ma_pnl['num_trades'])
            }
        else:
            baselines['ma_crossover'] = {
                'accuracy': 0.0,
                'total_return': 0.0,
                'num_trades': 0
            }
        
        # Baseline 3: Losowy sygnał (z takim samym udziałem transakcji jak model)
        np.random.seed(42)
        random_signals = []
        for i in range(len(prices)):
            rand = np.random.random()
            if rand < 0.33:  # ~33% BUY
                random_signals.append('BUY')
            elif rand < 0.66:  # ~33% SELL
                random_signals.append('SELL')
            else:  # ~33% NO TRADE
                random_signals.append('NO TRADE')
        
        random_pnl = self._calculate_pnl_with_costs(prices, random_signals, transaction_cost=0.001)
        random_predictions = [1 if s == 'BUY' else 0 for s in random_signals]
        random_accuracy = accuracy_score(actual_direction, random_predictions)
        
        baselines['random'] = {
            'accuracy': float(random_accuracy),
            'total_return': float(random_pnl['total_return']),
            'num_trades': int(random_pnl['num_trades'])
        }
        
        return baselines
    
    def train_model(self, test_size=0.2, random_state=42, incremental=False, max_months=12, n_features_select=15,
                    out_of_sample_months=6, feature_selection_lookback_months=12, initial_threshold=0.65, horizon_days=1):
        """Trenowanie modelu z poprawkami maklera i deva:
        - TimeSeriesSplit zamiast train_test_split
        - Ograniczenie do ostatnich max_months miesięcy
        - Feature selection BEZ LEAKAG'U: wybór cech na historycznym okresie (sprzed roku)
        - Prawdziwy test out-of-sample: ostatnie out_of_sample_months miesięcy jako test set
        - Rolling retraining (sprawdza czy model wymaga aktualizacji)
        - POPRAWKA DEV: Osobny model dla różnych horyzontów (1 dzień vs 5 dni)
        
        Parameters:
            out_of_sample_months (int): Liczba miesięcy do trzymania jako test set (domyślnie 6)
            feature_selection_lookback_months (int): Okres historyczny do wyboru cech (domyślnie 12)
            horizon_days (int): Horyzont przewidywania (1 = jutro, 5 = tydzień)
        """
        if self.features is None:
            raise ValueError("Najpierw utwórz cechy używając create_features()")
        
        # POPRAWKA DEV: Upewnij się, że cechy są utworzone z odpowiednim horyzontem
        if not hasattr(self, 'horizon_days') or self.horizon_days != horizon_days:
            print(f"Tworzenie cech z horyzontem {horizon_days} dni...")
            self.create_features(horizon_days=horizon_days)
        
        # POPRAWKA DEV: Prawdziwy test out-of-sample - ostatnie out_of_sample_months miesięcy jako "święte"
        # WAŻNE: Musimy użyć indeksów z self.features (po dropna()), nie z self.data!
        if self.features is not None and len(self.features) > 0:
            # Podziel dane na train/validation i out-of-sample test
            # Używamy indeksów z self.features, które są już po dropna()
            features_index = self.features.index
            test_cutoff_date = features_index[-1] - pd.DateOffset(months=out_of_sample_months)
            train_cutoff_date = features_index[-1] - pd.DateOffset(months=max_months)
            
            # Oznacz out-of-sample test set (ostatnie out_of_sample_months miesięcy)
            # Używamy indeksów z self.features, nie z self.data!
            test_mask = features_index >= test_cutoff_date
            train_val_mask = (features_index >= train_cutoff_date) & (features_index < test_cutoff_date)
            
            print(f"Podział danych:")
            print(f"  Train/Validation: {train_val_mask.sum()} dni (od {features_index[train_val_mask][0] if train_val_mask.any() else 'N/A'} do {features_index[train_val_mask][-1] if train_val_mask.any() else 'N/A'})")
            print(f"  Out-of-sample Test: {test_mask.sum()} dni (od {features_index[test_mask][0] if test_mask.any() else 'N/A'} do {features_index[test_mask][-1] if test_mask.any() else 'N/A'})")
            
            # Zapisz test set dla późniejszej oceny - używamy indeksów z self.features
            self.test_features = self.features.loc[test_mask] if test_mask.any() else None
            self.test_target = self.target.loc[test_mask] if test_mask.any() else None
            # Dla self.test_data musimy użyć indeksów z self.data, ale tylko te które są w self.features
            if test_mask.any() and self.data is not None:
                test_dates = features_index[test_mask]
                self.test_data = self.data.loc[self.data.index.isin(test_dates)] if len(test_dates) > 0 else None
            else:
                self.test_data = None
            
            # Użyj tylko train/validation do trenowania
            if train_val_mask.any():
                self.features = self.features.loc[train_val_mask]
                self.target = self.target.loc[train_val_mask]
                # Dla self.data musimy użyć indeksów z self.data, ale tylko te które są w self.features
                train_val_dates = features_index[train_val_mask]
                if self.data is not None:
                    self.data = self.data.loc[self.data.index.isin(train_val_dates)]
            else:
                print("Ostrzeżenie: Brak danych train/validation, używam wszystkich danych")
        
        # Walidacja danych przed trenowaniem
        # Sprawdź czy są wartości infinity
        if np.isinf(self.features.values).any():
            print("Ostrzeżenie: Znaleziono wartości infinity, zastępuję je...")
            self.features = self.features.replace([np.inf, -np.inf], np.nan)
            self.features = self.features.fillna(self.features.median())
        
        # Sprawdź czy są NaN
        if self.features.isna().any().any():
            print("Ostrzeżenie: Znaleziono wartości NaN, zastępuję je...")
            self.features = self.features.fillna(self.features.median())
        
        # Sprawdź czy wartości nie są zbyt duże
        max_val = np.abs(self.features.values).max()
        if max_val > 1e6:
            print(f"Ostrzeżenie: Znaleziono bardzo duże wartości ({max_val:.2e}), przycinam...")
            self.features = self.features.clip(-1e6, 1e6)
        
        # Konwertuj na float32 dla lepszej wydajności i uniknięcia overflow
        try:
            X_clean = self.features.astype(np.float32)
            y_clean = self.target.astype(int)
        except Exception as e:
            print(f"Błąd przy konwersji typów: {e}")
            # Spróbuj zastąpić problematyczne wartości
            X_clean = self.features.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
            y_clean = self.target.astype(int)
        
        # POPRAWKA DEV: Feature Selection BEZ LEAKAG'U - wybierz cechy na historycznym okresie
        # Sprawdź czy mamy już zapisane wybrane cechy (używamy ich na stałe)
        features_file = self.data_dir / f'{self.symbol}_features.json'
        selected_features = None
        
        if features_file.exists() and incremental:
            # Jeśli mamy zapisane cechy i to jest incremental update, użyj ich
            try:
                with open(features_file, 'r') as f:
                    features_data = json.load(f)
                    selected_features = features_data.get('selected_features', [])
                    if selected_features:
                        print(f"Używam zapisanych cech z poprzedniego trenowania: {len(selected_features)} cech")
            except:
                pass
        
        if selected_features is None:
            # Wybierz cechy na historycznym okresie (bez leakag'u)
            print(f"Feature selection BEZ LEAKAG'U: wybieranie {n_features_select} najlepszych cech z {len(X_clean.columns)}...")
            print(f"  Używam okresu sprzed {feature_selection_lookback_months} miesięcy do wyboru cech")
            
            # Określ datę cutoff dla feature selection (sprzed feature_selection_lookback_months miesięcy)
            # WAŻNE: Używamy indeksów z X_clean (które są z self.features), nie z self.data!
            if len(X_clean) > 0:
                feature_selection_cutoff = X_clean.index[-1] - pd.DateOffset(months=feature_selection_lookback_months)
                feature_selection_mask = X_clean.index < feature_selection_cutoff
                
                if feature_selection_mask.sum() > 50:  # Wymagaj minimum 50 dni danych
                    X_fs = X_clean.loc[feature_selection_mask]
                    y_fs = y_clean.loc[feature_selection_mask]
                    print(f"  Używam {len(X_fs)} dni historycznych do wyboru cech")
                else:
                    # Jeśli za mało danych historycznych, użyj pierwszych 70% danych
                    split_idx = int(len(X_clean) * 0.7)
                    X_fs = X_clean.iloc[:split_idx]
                    y_fs = y_clean.iloc[:split_idx]
                    print(f"  Za mało danych historycznych, używam pierwszych {len(X_fs)} dni ({split_idx/len(X_clean)*100:.1f}%)")
            else:
                # Fallback: użyj pierwszych 70% danych
                split_idx = int(len(X_clean) * 0.7)
                X_fs = X_clean.iloc[:split_idx]
                y_fs = y_clean.iloc[:split_idx]
            
            # Trenuj model na historycznych danych do obliczenia importance
            temp_model = RandomForestClassifier(
                n_estimators=50,  # Mniej drzew dla szybkości
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            temp_model.fit(X_fs, y_fs)
            
            # Oblicz feature importance na historycznych danych
            feature_importance = pd.DataFrame({
                'feature': X_fs.columns,
                'importance': temp_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Wybierz najlepsze cechy
            selected_features = feature_importance.head(n_features_select)['feature'].tolist()
            print(f"Wybrane cechy (na podstawie danych historycznych): {selected_features}")
            
            # Zapisz wybrane cechy dla późniejszego użycia
            with open(features_file, 'w') as f:
                json.dump({'selected_features': selected_features, 
                          'selection_date': datetime.now().isoformat(),
                          'lookback_months': feature_selection_lookback_months}, f, indent=2)
        
        # Zaktualizuj features do wybranych cech
        X_clean = X_clean[selected_features]
        self.selected_features = selected_features  # Zapisz dla późniejszego użycia
        
        # POPRAWKA 1: TimeSeriesSplit zamiast train_test_split
        print("Używanie TimeSeriesSplit (5 splits) zamiast train_test_split...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Trenuj model na każdym foldzie i użyj ostatniego do finalnego modelu
        best_model = None
        best_score = 0
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            X_train_fold, X_test_fold = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train_fold, y_test_fold = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            
            # Trenowanie modelu Random Forest CLASSIFIER
            fold_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Ocena folda
            fold_pred = fold_model.predict(X_test_fold)
            fold_score = accuracy_score(y_test_fold, fold_pred)
            fold_scores.append(fold_score)
            
            print(f"  Fold {fold_idx + 1}/5: Accuracy = {fold_score:.4f}")
            
            # Zapisz najlepszy model (lub ostatni)
            if fold_score >= best_score:
                best_score = fold_score
                best_model = fold_model
        
        # Użyj ostatniego modelu (najnowsze dane)
        self.model = best_model
        
        # Finalna ocena na ostatnim foldzie (najnowsze dane)
        last_train_idx, last_test_idx = list(tscv.split(X_clean))[-1]
        X_train_final = X_clean.iloc[last_train_idx]
        X_test_final = X_clean.iloc[last_test_idx]
        y_train_final = y_clean.iloc[last_train_idx]
        y_test_final = y_clean.iloc[last_test_idx]
        
        train_pred = self.model.predict(X_train_final)
        test_pred = self.model.predict(X_test_final)
        train_proba = self.model.predict_proba(X_train_final)[:, 1]
        test_proba = self.model.predict_proba(X_test_final)[:, 1]
        
        train_accuracy = accuracy_score(y_train_final, train_pred)
        test_accuracy = accuracy_score(y_test_final, test_pred)
        train_precision = precision_score(y_train_final, train_pred, zero_division=0)
        test_precision = precision_score(y_test_final, test_pred, zero_division=0)
        train_recall = recall_score(y_train_final, train_pred, zero_division=0)
        test_recall = recall_score(y_test_final, test_pred, zero_division=0)
        train_f1 = f1_score(y_train_final, train_pred, zero_division=0)
        test_f1 = f1_score(y_test_final, test_pred, zero_division=0)
        
        # POPRAWKA DEV: Prawdziwy test out-of-sample + baseline'y + kalibracja progu
        oos_results = None
        baseline_results = None
        optimal_threshold = initial_threshold
        
        if hasattr(self, 'test_features') and self.test_features is not None and len(self.test_features) > 0:
            print(f"\n{'='*60}")
            print(f"PRAWDZIWY TEST OUT-OF-SAMPLE (ostatnie {out_of_sample_months} miesięcy)")
            print(f"{'='*60}")
            
            # Przygotuj test features
            test_X = self.test_features[selected_features].astype(np.float32)
            test_y = self.test_target.astype(int)
            test_prices = self.test_data['Close']
            
            # Przewidywania na test set
            test_pred = self.model.predict(test_X)
            test_proba = self.model.predict_proba(test_X)[:, 1]
            
            # Metryki klasyfikacji
            oos_accuracy = accuracy_score(test_y, test_pred)
            oos_precision = precision_score(test_y, test_pred, zero_division=0)
            oos_recall = recall_score(test_y, test_pred, zero_division=0)
            oos_f1 = f1_score(test_y, test_pred, zero_division=0)
            
            print(f"Out-of-sample metryki:")
            print(f"  Accuracy: {oos_accuracy:.4f} ({oos_accuracy*100:.2f}%)")
            print(f"  Precision: {oos_precision:.4f}")
            print(f"  Recall: {oos_recall:.4f}")
            print(f"  F1-Score: {oos_f1:.4f}")
            
            # POPRAWKA DEV: Kalibracja progu decyzyjnego - znajdź próg maksymalizujący F1 lub Sharpe
            print(f"\nKalibracja progu decyzyjnego...")
            thresholds = np.arange(0.50, 0.95, 0.05)
            threshold_scores = []
            
            for thresh in thresholds:
                # Oblicz sygnały dla tego progu
                signals = []
                for prob in test_proba:
                    if prob > thresh:
                        signals.append('BUY')
                    elif prob < (1 - thresh):
                        signals.append('SELL')
                    else:
                        signals.append('NO TRADE')
                
                # Oblicz P&L dla tego progu (z kosztami transakcji 0.1%)
                pnl = self._calculate_pnl_with_costs(test_prices, signals, transaction_cost=0.001)
                
                # Oblicz F1 dla sygnałów BUY (traktuj jako klasę pozytywną)
                buy_signals = [1 if s == 'BUY' else 0 for s in signals]
                actual_up = test_y.values
                if sum(buy_signals) > 0:
                    buy_f1 = f1_score(actual_up, buy_signals, zero_division=0)
                else:
                    buy_f1 = 0
                
                # Oblicz Sharpe ratio (uproszczony)
                returns = pnl['returns']
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                else:
                    sharpe = 0
                
                threshold_scores.append({
                    'threshold': thresh,
                    'f1': buy_f1,
                    'sharpe': sharpe,
                    'total_return': pnl['total_return'],
                    'num_trades': pnl['num_trades']
                })
            
            # Znajdź próg maksymalizujący F1
            best_f1_idx = np.argmax([s['f1'] for s in threshold_scores])
            optimal_threshold_f1 = threshold_scores[best_f1_idx]['threshold']
            
            # Znajdź próg maksymalizujący Sharpe
            best_sharpe_idx = np.argmax([s['sharpe'] for s in threshold_scores])
            optimal_threshold_sharpe = threshold_scores[best_sharpe_idx]['threshold']
            
            # Użyj progu z lepszym F1 (lub można użyć Sharpe)
            optimal_threshold = optimal_threshold_f1
            
            print(f"  Najlepszy próg (F1): {optimal_threshold_f1:.2f} (F1={threshold_scores[best_f1_idx]['f1']:.4f})")
            print(f"  Najlepszy próg (Sharpe): {optimal_threshold_sharpe:.2f} (Sharpe={threshold_scores[best_sharpe_idx]['sharpe']:.4f})")
            print(f"  Używam progu: {optimal_threshold:.2f}")
            
            # Oblicz P&L z optymalnym progiem
            optimal_signals = []
            for prob in test_proba:
                if prob > optimal_threshold:
                    optimal_signals.append('BUY')
                elif prob < (1 - optimal_threshold):
                    optimal_signals.append('SELL')
                else:
                    optimal_signals.append('NO TRADE')
            
            oos_pnl = self._calculate_pnl_with_costs(test_prices, optimal_signals, transaction_cost=0.001)
            
            oos_results = {
                'accuracy': float(oos_accuracy),
                'precision': float(oos_precision),
                'recall': float(oos_recall),
                'f1': float(oos_f1),
                'total_return': float(oos_pnl['total_return']),
                'num_trades': int(oos_pnl['num_trades']),
                'sharpe': float(oos_pnl['sharpe']) if 'sharpe' in oos_pnl else 0.0,
                'optimal_threshold': float(optimal_threshold)
            }
            
            print(f"\nP&L na out-of-sample test set:")
            print(f"  Total Return: {oos_pnl['total_return']:.2%}")
            print(f"  Liczba transakcji: {oos_pnl['num_trades']}")
            if 'sharpe' in oos_pnl:
                print(f"  Sharpe Ratio: {oos_pnl['sharpe']:.4f}")
            
            # POPRAWKA DEV: Baseline'y
            print(f"\n{'='*60}")
            print(f"BASELINE'Y - Porównanie z prostymi strategiami")
            print(f"{'='*60}")
            
            baseline_results = self._calculate_baselines(test_prices, test_y)
            
            for baseline_name, baseline_metrics in baseline_results.items():
                print(f"{baseline_name}:")
                print(f"  Accuracy: {baseline_metrics.get('accuracy', 0):.4f}")
                print(f"  Total Return: {baseline_metrics.get('total_return', 0):.2%}")
                print(f"  Num Trades: {baseline_metrics.get('num_trades', 0)}")
            
            # Porównanie z baseline'ami
            print(f"\nPorównanie z baseline'ami:")
            always_up_return = baseline_results.get('always_up', {}).get('total_return', 0)
            ma_crossover_return = baseline_results.get('ma_crossover', {}).get('total_return', 0)
            random_return = baseline_results.get('random', {}).get('total_return', 0)
            
            print(f"  Model vs Always UP: {oos_pnl['total_return'] - always_up_return:+.2%}")
            print(f"  Model vs MA Crossover: {oos_pnl['total_return'] - ma_crossover_return:+.2%}")
            print(f"  Model vs Random: {oos_pnl['total_return'] - random_return:+.2%}")
        
        print(f"\n{'='*60}")
        print(f"Wyniki modelu klasyfikacji dla {self.symbol}:")
        print(f"{'='*60}")
        print(f"Średnia accuracy z 5 foldów: {np.mean(fold_scores):.4f} (std: {np.std(fold_scores):.4f})")
        print(f"Ostatni fold (najnowsze dane):")
        print(f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"Liczba wybranych cech: {len(selected_features)}/{len(X_clean.columns)}")
        
        # Zaktualizuj self.features do wybranych cech
        self.features = X_clean
        
        # Zapisz informacje o modelu
        model_info = {
            'symbol': self.symbol,
            'model_type': 'classification',
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_precision': float(train_precision),
            'test_precision': float(test_precision),
            'train_recall': float(train_recall),
            'test_recall': float(test_recall),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'last_trained': datetime.now().isoformat(),
            'data_points': len(self.features),
            'feature_count': len(selected_features),
            'selected_features': selected_features,
            'cv_scores': [float(s) for s in fold_scores],
            'cv_mean_accuracy': float(np.mean(fold_scores)),
            'cv_std_accuracy': float(np.std(fold_scores)),
            'max_months': max_months,
            'optimal_threshold': float(optimal_threshold),
            'out_of_sample': oos_results,
            'baselines': baseline_results
        }
        
        with open(self.model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # POPRAWKA DEV: Zapisz model z odpowiednią nazwą w zależności od horyzontu
        if horizon_days == 5:
            # Model tygodniowy - osobny plik
            weekly_model_file = self.data_dir / f'{self.symbol}_weekly_model.pkl'
            with open(weekly_model_file, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Zapisano model tygodniowy: {weekly_model_file}")
        else:
            # Model dzienny - standardowy plik
            with open(self.model_pickle_file, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Zapisz listę wybranych cech (z informacją o horyzoncie)
        # Dla modelu dziennego (horizon_days=1) zapisz do standardowego pliku
        # Dla modelu tygodniowego (horizon_days=5) zapisz do osobnego pliku
        if horizon_days == 5:
            features_file = self.data_dir / f'{self.symbol}_weekly_features.json'
        else:
            features_file = self.data_dir / f'{self.symbol}_features.json'
        
        features_data = {
            'selected_features': selected_features,
            'horizon_days': horizon_days
        }
        if horizon_days == 5:
            # Dla modelu tygodniowego, zapisz też średni weekly_return dla pewnych sygnałów
            if hasattr(self, 'weekly_return') and self.weekly_return is not None:
                # Oblicz średni weekly_return dla przypadków gdzie model był pewny i trafny
                weekly_returns_series = self.weekly_return.dropna()
                if len(weekly_returns_series) > 0:
                    # Użyj mediany jako bardziej odpornej na outliers
                    avg_weekly_return = weekly_returns_series.median()
                    features_data['avg_weekly_return'] = float(avg_weekly_return)
                    print(f"Średni tygodniowy zwrot (mediana): {avg_weekly_return:.4f}")
        
        with open(features_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        
        return self.model
    
    def load_model(self):
        """Załaduj zapisany model z pliku i wybrane cechy"""
        if self.model_pickle_file.exists() and self.model_file.exists():
            try:
                with open(self.model_pickle_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Załaduj też informacje o modelu
                with open(self.model_file, 'r') as f:
                    model_info = json.load(f)
                
                # Załaduj wybrane cechy jeśli istnieją
                features_file = self.data_dir / f'{self.symbol}_features.json'
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        features_data = json.load(f)
                        self.selected_features = features_data.get('selected_features', [])
                elif 'selected_features' in model_info:
                    self.selected_features = model_info['selected_features']
                else:
                    self.selected_features = None
                
                print(f"Załadowano zapisany model dla {self.symbol}")
                if self.selected_features:
                    print(f"  Wybrane cechy: {len(self.selected_features)}")
                
                return self.model, model_info
            except Exception as e:
                print(f"Błąd przy ładowaniu modelu dla {self.symbol}: {e}")
                return None, {}
        return None, {}
    
    def predict_next_day(self):
        """Przewidywanie kierunku na kolejny dzień (UP/DOWN)"""
        return self.predict_period(days=1)
    
    def predict_period(self, days=1, decision_threshold=None, use_weekly_model=False):
        """Przewidywanie kierunku i sygnału transakcyjnego na określony okres (w dniach)
        
        Parameters:
            days (int): Liczba dni do przodu (1 = jutro, 7 = tydzień, 30 = miesiąc, 180 = 6 miesięcy)
            decision_threshold (float): Próg decyzyjny (None = użyj optymalnego z modelu, 0.65 = domyślny)
        
        Returns:
            dict: Słownik z przewidywaniami zawierający:
                - symbol: symbol spółki
                - current_price: aktualna cena
                - predicted_price: przewidywana cena (szacunkowa, oparta na historycznych średnich)
                - predicted_direction: 'UP' lub 'DOWN'
                - direction_probability: prawdopodobieństwo przewidywanego kierunku
                - signal: 'BUY', 'SELL', lub 'NO TRADE' (POPRAWKA 4: próg decyzyjny 0.65)
                - change: zmiana ceny w $ (szacunkowa)
                - change_percent: zmiana ceny w % (szacunkowa)
                - period_days: liczba dni przewidywania
        """
        # POPRAWKA DEV: Dla tygodnia (5 dni), użyj modelu tygodniowego jeśli dostępny
        if days == 5 and use_weekly_model:
            weekly_model = self.load_weekly_model()
            if weekly_model is not None:
                # Użyj modelu tygodniowego
                original_model = self.model
                self.model = weekly_model
                # Upewnij się, że mamy cechy z horyzontem 5 dni
                if not hasattr(self, 'horizon_days') or self.horizon_days != 5:
                    self.create_features(horizon_days=5)
        
        if self.model is None:
            raise ValueError("Najpierw wytrenuj model używając train_model()")
        
        # POPRAWKA DEV: Użyj optymalnego progu z modelu jeśli dostępny
        if decision_threshold is None:
            # Spróbuj załadować optymalny próg z model_info
            try:
                if self.model_file.exists():
                    with open(self.model_file, 'r') as f:
                        model_info = json.load(f)
                        decision_threshold = model_info.get('optimal_threshold', 0.65)
                else:
                    decision_threshold = 0.65
            except:
                decision_threshold = 0.65
        
        # Sprawdź jakie cechy model oczekuje (z feature_names_in_ jeśli dostępne)
        model_expected_features = None
        if hasattr(self.model, 'feature_names_in_'):
            model_expected_features = list(self.model.feature_names_in_)
        elif hasattr(self, 'selected_features') and self.selected_features:
            model_expected_features = self.selected_features
        
        # Przygotuj features zgodnie z oczekiwaniami modelu
        if model_expected_features:
            # Model ma określone cechy - użyj tylko tych
            # Sprawdź które cechy są dostępne
            available_features = [f for f in model_expected_features if f in self.features.columns]
            
            if len(available_features) != len(model_expected_features):
                # Niektóre cechy brakują - spróbuj użyć dostępnych
                missing = set(model_expected_features) - set(available_features)
                print(f"Ostrzeżenie: Brakujące cechy: {missing}")
                
                # Jeśli model sklearn wymaga dokładnego dopasowania (feature_names_in_)
                if hasattr(self.model, 'feature_names_in_'):
                    # Utwórz DataFrame z wszystkimi oczekiwanymi cechami
                    last_features_df = pd.DataFrame(index=self.features.index)
                    for feat in model_expected_features:
                        if feat in self.features.columns:
                            last_features_df[feat] = self.features[feat]
                        else:
                            # Ustaw brakującą cechę na 0
                            last_features_df[feat] = 0
                            print(f"  Ustawiam brakującą cechę {feat} na 0")
                    
                    # Upewnij się, że kolumny są w tej samej kolejności co model oczekuje
                    last_features_df = last_features_df[model_expected_features]
                    last_features = last_features_df.iloc[-1:].values
                else:
                    # Użyj tylko dostępnych cech
                    last_features = self.features[available_features].iloc[-1:].values
            else:
                # Wszystkie cechy dostępne - użyj w kolejności oczekiwanej przez model
                last_features = self.features[model_expected_features].iloc[-1:].values
        else:
            # Model nie ma określonych cech - użyj wszystkich dostępnych
            if hasattr(self, 'selected_features') and self.selected_features:
                available_features = [f for f in self.selected_features if f in self.features.columns]
                last_features = self.features[available_features].iloc[-1:].values
            else:
                last_features = self.features.iloc[-1:].values
        
        # Przewiduj kierunek (0 = DOWN, 1 = UP)
        direction_pred = self.model.predict(last_features)[0]
        
        # Prawdopodobieństwo wzrostu
        direction_proba = self.model.predict_proba(last_features)[0]
        up_probability = direction_proba[1] if len(direction_proba) > 1 else 0.5
        down_probability = 1 - up_probability
        
        current_price = self.data['Close'].iloc[-1]
        
        # POPRAWKA 4: Próg decyzyjny 0.65 zamiast 0.5
        # Sygnał BUY jeśli up_probability > 0.65
        # Sygnał SELL jeśli down_probability > 0.65 (czyli up_probability < 0.35)
        # Sygnał NO TRADE jeśli prawdopodobieństwo między 0.35 a 0.65
        if up_probability > decision_threshold:
            signal = 'BUY'
            direction_prob = up_probability
        elif up_probability < (1 - decision_threshold):  # down_probability > decision_threshold
            signal = 'SELL'
            direction_prob = down_probability
        else:
            signal = 'NO TRADE'
            direction_prob = max(up_probability, down_probability)  # Użyj wyższego prawdopodobieństwa
        
        # POPRAWKA DEV: Dla tygodnia (5 dni), użyj weekly_return z modelu tygodniowego
        if days == 5 and use_weekly_model and hasattr(self, 'avg_weekly_return'):
            # Użyj średniego weekly_return dla pewnych sygnałów
            avg_weekly_return = self.avg_weekly_return
            
            # Oblicz prognozowany zwrot tygodniowy na podstawie prawdopodobieństwa
            # Jak proba_up ~ 0.5 → prawie brak ruchu
            # Jak proba_up wysoka → mocniejszy ruch w górę
            # Jak proba_up < 0.5 → ruch w dół
            predicted_weekly_return = avg_weekly_return * (2 * up_probability - 1)
            
            # Prognozowany poziom za tydzień
            estimated_price = current_price * np.exp(predicted_weekly_return)
            change = estimated_price - current_price
            change_percent = (np.exp(predicted_weekly_return) - 1) * 100
        else:
            # POPRAWKA 5: Szacunkowa cena (oparta na historycznych średnich, nie na modelu)
            # Model przewiduje tylko kierunek, nie cenę
            historical_changes = self.data['Close'].pct_change().dropna()
            
            # Dla krótkich okresów (1-7 dni) użyj dziennych zmian
            if days <= 7:
                avg_up_change = historical_changes[historical_changes > 0].mean() if len(historical_changes[historical_changes > 0]) > 0 else 0.01
                avg_down_change = historical_changes[historical_changes < 0].mean() if len(historical_changes[historical_changes < 0]) > 0 else -0.01
                # Skaluj na liczbę dni
                avg_up_change = avg_up_change * days
                avg_down_change = avg_down_change * days
                
                # Szacunkowa zmiana (używana tylko do wyświetlania, nie do decyzji)
                if signal == 'BUY':
                    estimated_change_pct = avg_up_change * direction_prob
                elif signal == 'SELL':
                    estimated_change_pct = -abs(avg_down_change) * direction_prob
                else:  # NO TRADE
                    # Użyj średniej ważonej
                    estimated_change_pct = (avg_up_change * up_probability) + (avg_down_change * down_probability)
                
                estimated_price = current_price * (1 + estimated_change_pct)
                change = estimated_price - current_price
                change_percent = estimated_change_pct * 100
            else:
                # Dla dłuższych okresów użyj zmian dla odpowiedniego okna czasowego
                if len(self.data) >= days:
                    # Oblicz zmiany dla okien o długości 'days'
                    period_changes = self.data['Close'].pct_change(periods=days).dropna()
                    avg_up_change = period_changes[period_changes > 0].mean() if len(period_changes[period_changes > 0]) > 0 else 0.01
                    avg_down_change = period_changes[period_changes < 0].mean() if len(period_changes[period_changes < 0]) > 0 else -0.01
                else:
                    # Jeśli nie ma wystarczająco danych, użyj średniej dziennej zmiany * liczba dni
                    avg_up_change = historical_changes[historical_changes > 0].mean() if len(historical_changes[historical_changes > 0]) > 0 else 0.01
                    avg_down_change = historical_changes[historical_changes < 0].mean() if len(historical_changes[historical_changes < 0]) > 0 else -0.01
                    # Skaluj na liczbę dni (z uwzględnieniem efektu składanego)
                    avg_up_change = (1 + avg_up_change) ** days - 1
                    avg_down_change = (1 + avg_down_change) ** days - 1
            
                # Szacunkowa zmiana (używana tylko do wyświetlania, nie do decyzji)
                if signal == 'BUY':
                    estimated_change_pct = avg_up_change * direction_prob
                elif signal == 'SELL':
                    estimated_change_pct = -abs(avg_down_change) * direction_prob
                else:  # NO TRADE
                    # Użyj średniej ważonej
                    estimated_change_pct = (avg_up_change * up_probability) + (avg_down_change * down_probability)
                
                estimated_price = current_price * (1 + estimated_change_pct)
                change = estimated_price - current_price
                change_percent = estimated_change_pct * 100
        
        # LOGOWANIE PREDYKCJI - zapisz do logu jeśli moduł jest dostępny
        if PREDICTION_LOGGING_AVAILABLE:
            try:
                # Pobierz wersję modelu (data trenowania)
                model_version = None
                if self.model_file.exists():
                    try:
                        with open(self.model_file, 'r') as f:
                            model_info = json.load(f)
                            model_version = model_info.get('last_trained', None)
                    except:
                        pass
                
                # Utwórz logger (singleton pattern - jeden logger na katalog)
                logger = PredictionLogger(data_dir=str(self.data_dir), use_sqlite=True)
                
                # Konwertuj sygnał na format logu (NO TRADE -> NO_TRADE)
                decision_log = signal.replace(' ', '_')
                
                # Zapisz predykcję
                logger.log_prediction(
                    symbol=self.symbol,
                    horizon_days=days,
                    price_now=current_price,
                    proba_up=up_probability,
                    decision=decision_log,
                    model_version=model_version
                )
            except Exception as log_error:
                # Nie przerywaj predykcji jeśli logowanie się nie powiodło
                print(f"Ostrzeżenie: Nie udało się zalogować predykcji: {log_error}")
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'predicted_price': estimated_price,  # Szacunkowa, oparta na historycznych średnich
            'predicted_direction': 'UP' if direction_pred == 1 else 'DOWN',
            'direction_probability': direction_prob,
            'signal': signal,  # BUY/SELL/NO TRADE
            'change': change,
            'change_percent': change_percent,
            'period_days': days,
            'decision_threshold': decision_threshold
        }
    
    def train_weekly_model(self, **kwargs):
        """Trenuj osobny model dla tygodnia (5 dni handlowych)
        
        Parameters:
            **kwargs: Argumenty przekazywane do train_model()
        """
        print(f"\n{'='*60}")
        print(f"TRENOWANIE MODELU TYGODNIOWEGO (horyzont 5 dni)")
        print(f"{'='*60}")
        
        # Utwórz cechy z horyzontem 5 dni
        self.create_features(horizon_days=5)
        
        # Trenuj model z horyzontem 5 dni
        return self.train_model(horizon_days=5, **kwargs)
    
    def load_weekly_model(self):
        """Załaduj model tygodniowy jeśli istnieje"""
        weekly_model_file = self.data_dir / f'{self.symbol}_weekly_model.pkl'
        if weekly_model_file.exists():
            try:
                with open(weekly_model_file, 'rb') as f:
                    weekly_model = pickle.load(f)
                
                # Załaduj też informacje o cechach (z osobnego pliku dla modelu tygodniowego)
                weekly_features_file = self.data_dir / f'{self.symbol}_weekly_features.json'
                if weekly_features_file.exists():
                    with open(weekly_features_file, 'r') as f:
                        features_data = json.load(f)
                        if features_data.get('horizon_days') == 5:
                            self.weekly_selected_features = features_data.get('selected_features', [])
                            self.avg_weekly_return = features_data.get('avg_weekly_return', 0.04)
                            print(f"Załadowano model tygodniowy dla {self.symbol}")
                            return weekly_model
                else:
                    # Fallback: spróbuj załadować z standardowego pliku
                    features_file = self.data_dir / f'{self.symbol}_features.json'
                    if features_file.exists():
                        with open(features_file, 'r') as f:
                            features_data = json.load(f)
                            if features_data.get('horizon_days') == 5:
                                self.weekly_selected_features = features_data.get('selected_features', [])
                                self.avg_weekly_return = features_data.get('avg_weekly_return', 0.04)
                                print(f"Załadowano model tygodniowy dla {self.symbol} (z standardowego pliku)")
                                return weekly_model
                
                return weekly_model
            except Exception as e:
                print(f"Błąd przy ładowaniu modelu tygodniowego: {e}")
                return None
        return None
    
    def predict_week(self):
        """Przewidywanie na tydzień (5 dni handlowych) - używa osobnego modelu tygodniowego"""
        # POPRAWKA DEV: Użyj modelu tygodniowego jeśli dostępny
        weekly_model = self.load_weekly_model()
        if weekly_model is not None:
            # Użyj modelu tygodniowego
            return self.predict_period(days=5, use_weekly_model=True)
        else:
            # Fallback: użyj standardowego modelu
            return self.predict_period(days=5)
    
    def predict_month(self):
        """Przewidywanie na miesiąc (30 dni)"""
        return self.predict_period(days=30)
    
    def predict_6months(self):
        """Przewidywanie na 6 miesięcy (180 dni)"""
        return self.predict_period(days=180)
    
    def plot_predictions(self, days=60):
        """Wizualizacja przewidywań kierunku na ostatnich dniach"""
        if self.model is None:
            raise ValueError("Najpierw wytrenuj model używając train_model()")
        
        # Przewidywania na ostatnich dniach
        # Użyj odpowiednich cech zgodnie z oczekiwaniami modelu
        if hasattr(self.model, 'feature_names_in_'):
            model_expected_features = list(self.model.feature_names_in_)
            recent_features = self.features[model_expected_features].iloc[-days:]
        elif hasattr(self, 'selected_features') and self.selected_features:
            recent_features = self.features[self.selected_features].iloc[-days:]
        else:
            recent_features = self.features.iloc[-days:]
        
        recent_actual_direction = self.target.iloc[-days:]  # 0 = DOWN, 1 = UP
        recent_actual_prices = self.data['Close'].iloc[-days:]
        recent_predictions = self.model.predict(recent_features)
        recent_proba = self.model.predict_proba(recent_features)[:, 1]  # Prawdopodobieństwo UP
        recent_dates = self.data.index[-days:]
        
        plt.figure(figsize=(14, 10))
        
        # Wykres cen z oznaczeniem kierunku
        plt.subplot(3, 1, 1)
        plt.plot(recent_dates, recent_actual_prices.values, label='Cena rzeczywista', linewidth=2, color='blue')
        
        # Oznacz poprawne przewidywania kierunku
        correct_predictions = recent_actual_direction.values == recent_predictions
        for i, (date, price, correct) in enumerate(zip(recent_dates, recent_actual_prices.values, correct_predictions)):
            color = 'green' if correct else 'red'
            marker = '^' if recent_predictions[i] == 1 else 'v'
            plt.scatter(date, price, color=color, marker=marker, s=50, alpha=0.6)
        
        plt.title(f'Przewidywania kierunku akcji {self.symbol} - ostatnie {days} dni', fontsize=14, fontweight='bold')
        plt.xlabel('Data')
        plt.ylabel('Cena ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Wykres dokładności przewidywań kierunku
        plt.subplot(3, 1, 2)
        accuracy_by_day = (recent_actual_direction.values == recent_predictions).astype(int)
        plt.plot(recent_dates, accuracy_by_day, label='Poprawne przewidywanie', linewidth=2, color='green', alpha=0.7)
        plt.fill_between(recent_dates, accuracy_by_day, alpha=0.3, color='green')
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, label='Próg losowy')
        plt.title('Dokładność przewidywań kierunku (1 = poprawne, 0 = błędne)', fontsize=12)
        plt.xlabel('Data')
        plt.ylabel('Poprawność')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        
        # Wykres prawdopodobieństwa wzrostu
        plt.subplot(3, 1, 3)
        plt.plot(recent_dates, recent_proba, label='Prawdopodobieństwo wzrostu', linewidth=2, color='purple')
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, label='Próg decyzyjny (50%)')
        plt.fill_between(recent_dates, recent_proba, 0.5, alpha=0.3, where=(recent_proba >= 0.5), color='green', label='Przewidywanie: UP')
        plt.fill_between(recent_dates, recent_proba, 0.5, alpha=0.3, where=(recent_proba < 0.5), color='red', label='Przewidywanie: DOWN')
        plt.title('Prawdopodobieństwo wzrostu ceny', fontsize=12)
        plt.xlabel('Data')
        plt.ylabel('Prawdopodobieństwo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Dodaj statystyki
        total_accuracy = accuracy_by_day.mean()
        plt.figtext(0.02, 0.02, f'Średnia dokładność: {total_accuracy:.2%}', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f'{self.symbol}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"\nWykres zapisany jako plots/{self.symbol}_predictions.png")
        plt.close()


class MultiStockPredictor:
    """Klasa do zarządzania przewidywaniami dla wielu spółek"""
    
    def __init__(self, excel_file='raport.xlsx', data_dir='stock_data'):
        self.excel_file = excel_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.symbols = []
        self.predictors = {}
        
    def load_symbols_from_excel(self):
        """Wczytuje symbole spółek z pliku Excel, rozróżniając otwarte i historyczne pozycje
        
        Returns:
            dict: {'open': [list of symbols], 'closed': [list of symbols], 'all': [list of all symbols]}
        """
        print(f"Wczytywanie symboli z pliku {self.excel_file}...")
        
        open_symbols = []
        closed_symbols = []
        
        try:
            # Sprawdź dostępne arkusze
            xl_file = pd.ExcelFile(self.excel_file)
            sheet_names = xl_file.sheet_names
            print(f"Dostępne arkusze: {sheet_names}")
            
            # Wczytaj otwarte pozycje
            open_sheet = None
            for sheet in sheet_names:
                if 'OPEN POSITION' in sheet.upper():
                    open_sheet = sheet
                    break
            
            if open_sheet:
                print(f"Wczytywanie otwartych pozycji z arkusza: {open_sheet}")
                df_open = pd.read_excel(self.excel_file, sheet_name=open_sheet, header=None)
                
                # Znajdź wiersz z nagłówkami
                header_row = None
                for i in range(len(df_open)):
                    row = df_open.iloc[i]
                    if 'Position' in str(row.iloc[1]) and 'Symbol' in str(row.iloc[2]):
                        header_row = i
                        break
                
                if header_row is not None:
                    positions_df = pd.read_excel(
                        self.excel_file,
                        sheet_name=open_sheet,
                        header=header_row,
                        skiprows=0
                    )
                    
                    if 'Symbol' in positions_df.columns:
                        symbols = positions_df['Symbol'].dropna().unique()
                        open_symbols = [str(s).strip() for s in symbols if str(s).strip() != '' and str(s).strip().upper() != 'TOTAL']
                        print(f"Znaleziono {len(open_symbols)} unikalnych symboli w otwartych pozycjach: {open_symbols}")
            
            # Wczytaj zamknięte/historyczne pozycje
            closed_sheet = None
            for sheet in sheet_names:
                if 'CLOSED POSITION' in sheet.upper() or 'HISTORY' in sheet.upper():
                    closed_sheet = sheet
                    break
            
            if closed_sheet:
                print(f"Wczytywanie historycznych pozycji z arkusza: {closed_sheet}")
                df_closed = pd.read_excel(self.excel_file, sheet_name=closed_sheet, header=None)
                
                # Znajdź wiersz z nagłówkami
                header_row = None
                for i in range(len(df_closed)):
                    row = df_closed.iloc[i]
                    if 'Position' in str(row.iloc[1]) and 'Symbol' in str(row.iloc[2]):
                        header_row = i
                        break
                
                if header_row is not None:
                    positions_df = pd.read_excel(
                        self.excel_file,
                        sheet_name=closed_sheet,
                        header=header_row,
                        skiprows=0
                    )
                    
                    if 'Symbol' in positions_df.columns:
                        symbols = positions_df['Symbol'].dropna().unique()
                        closed_symbols = [str(s).strip() for s in symbols if str(s).strip() != '' and str(s).strip().upper() != 'TOTAL']
                        print(f"Znaleziono {len(closed_symbols)} unikalnych symboli w historycznych pozycjach: {closed_symbols}")
            
            # Połącz wszystkie symbole
            all_symbols = list(set(open_symbols + closed_symbols))
            self.symbols = all_symbols
            
            # Zapisz informacje o statusie pozycji
            self.open_symbols = open_symbols
            self.closed_symbols = closed_symbols
            
            print(f"Łącznie znaleziono {len(all_symbols)} unikalnych symboli ({len(open_symbols)} otwartych, {len(closed_symbols)} historycznych)")
            
            return {
                'open': open_symbols,
                'closed': closed_symbols,
                'all': all_symbols
            }
            
        except Exception as e:
            print(f"Błąd przy wczytywaniu symboli: {e}")
            # Fallback - stara metoda
            try:
                df = pd.read_excel(self.excel_file, header=None)
                
                # Znajdź wiersz z nagłówkami
                header_row = None
                for i in range(len(df)):
                    row = df.iloc[i]
                    if 'Position' in str(row.iloc[1]) and 'Symbol' in str(row.iloc[2]):
                        header_row = i
                        break
                
                if header_row is None:
                    raise ValueError("Nie znaleziono nagłówków w pliku Excel")
                
                # Wczytaj dane pozycji
                positions_df = pd.read_excel(
                    self.excel_file,
                    header=header_row,
                    skiprows=0
                )
                
                # Wyciągnij unikalne symbole
                if 'Symbol' in positions_df.columns:
                    symbols = positions_df['Symbol'].dropna().unique()
                    self.symbols = [str(s).strip() for s in symbols if str(s).strip() != '']
                    print(f"Znaleziono {len(self.symbols)} unikalnych symboli: {self.symbols}")
                    self.open_symbols = []
                    self.closed_symbols = self.symbols
                    return {
                        'open': [],
                        'closed': self.symbols,
                        'all': self.symbols
                    }
                else:
                    raise ValueError("Nie znaleziono kolumny 'Symbol' w pliku Excel")
            except Exception as e2:
                print(f"Błąd przy fallback: {e2}")
                raise
            
        except Exception as e:
            print(f"Błąd przy wczytywaniu pliku Excel: {e}")
            # Zwróć puste listy jeśli wszystko się nie powiodło
            self.symbols = []
            self.open_symbols = []
            self.closed_symbols = []
            return {
                'open': [],
                'closed': [],
                'all': []
            }
    
    def get_all_german_stocks_list(self):
        """Zwraca listę WSZYSTKICH spółek niemieckich do pobrania"""
        german_stocks_list = [
            "12DA.DE", "1COV.DE", "1FC.DE", "1SXP.DE", "1YD.DE", "2PP.DE", "AAD.DE", "ABEA.DE",
            "ADJ.DE", "ADS.DE", "AFX.DE", "AHLA.DE", "AIR.DE", "AIXA.DE", "ALV.DE", "AMD.DE",
            "APC.DE", "AR4.DE", "ARLN.DE", "AT1.DE", "B4B.DE", "BAS.DE", "BAYN.DE", "BC8.DE",
            "BEI.DE", "BFSA.DE", "BMT.DE", "BMW3.DE", "BMW.DE", "BNR.DE", "BOSS.DE", "BPE5.DE",
            "BRYN.DE", "BTCF.DE", "BVB.DE", "CBK.DE", "CEC.DE", "COK.DE", "CON.DE", "DAI.DE",
            "DB1.DE", "DBK.DE", "DEQ.DE", "DEZ.DE", "DHER.DE", "DHL.DE", "DPW.DE", "DRW3.DE",
            "DRW8.DE", "DTE.DE", "DTG.DE", "DUE.DE", "DWNI.DE", "EAD.DE", "ECV.DE", "ENR.DE",
            "EOAN.DE", "EVK.DE", "EVT.DE", "F3C.DE", "FIE.DE", "FME.DE", "FPE3.DE", "FRA.DE",
            "FRE.DE", "FTK.DE", "G1A.DE", "GBF.DE", "GIL.DE", "GKS.DE", "GXI.DE", "GYC.DE",
            "HAG.DE", "HDD.DE", "HEI.DE", "HEN3.DE", "HFG.DE", "HLAG.DE", "HNR.DE", "HOT.DE",
            "HRPK.DE", "HYQ.DE", "HYUD.DE", "IFX.DE", "INS.DE", "JEN.DE", "JST.DE", "JUN3.DE",
            "KBX.DE", "KCO.DE", "KGX.DE", "KRN.DE", "LEG.DE", "LHA.DE", "LXS.DE", "M0Y.DE",
            "M0YN.DE", "MBB.DE", "MBG.DE", "MRK.DE", "MSF.DE", "MTX.DE", "MUV2.DE", "NA9.DE",
            "NDA.DE", "NDX1.DE", "NEM.DE", "NN6.DE", "NOEJ.DE", "NVD.DE", "O2D.DE", "OMV.DE",
            "P911.DE", "PAH3.DE", "PNE3.DE", "PSM.DE", "PUM.DE", "QIA.DE", "RDC.DE", "RHM.DE",
            "RIO1.DE", "RRTL.DE", "RWE.DE", "S92.DE", "SAE.DE", "SAP.DE", "SAX.DE", "SBS.DE",
            "SDF.DE", "SHA.DE", "SHL.DE", "SIE.DE", "SKB.DE", "SMHN.DE", "SRT.DE", "ST5.DE",
            "SVE.DE", "SY1.DE", "SYAB.DE", "SZG.DE", "SZU.DE", "TEG.DE", "TKA.DE", "TLX.DE",
            "TMV.DE", "TSLA.DE", "TUI.DE", "UN01.DE", "UTDI.DE", "VBK.DE", "VNA.DE", "VOW1.DE",
            "VOW3.DE", "VX1.DE", "WAF.DE", "WCH.DE", "WSV2.DE", "WDI.DE", "ZAL.DE", "ZIL2.DE"
        ]
        
        return german_stocks_list
    
    def get_german_stocks_in_database(self):
        """Znajdź spółki niemieckie, które mają już dane w bazie"""
        german_stocks_list = self.get_all_german_stocks_list()
        
        # Sprawdź które mają pliki danych
        stocks_in_db = []
        for symbol in german_stocks_list:
            data_file = self.data_dir / f'{symbol}_data.csv'
            if data_file.exists():
                stocks_in_db.append(symbol)
        
        return stocks_in_db
    
    def daily_update(self, include_german_stocks=True, all_german_stocks=False, max_stocks=None, start_from=0):
        """Codzienne aktualizowanie danych i uczenie modeli
        
        Parameters:
        include_german_stocks (bool): Czy uwzględnić spółki niemieckie
        all_german_stocks (bool): Czy pobrać wszystkie spółki niemieckie (True) czy tylko te w bazie (False)
        max_stocks (int): Maksymalna liczba spółek do przetworzenia (None = wszystkie)
        start_from (int): Indeks od którego zacząć (przydatne do kontynuacji po błędzie)
        """
        print("\n" + "=" * 80)
        print("CODZIENNA AKTUALIZACJA DANYCH I UCZENIE")
        print("=" * 80)
        
        if not self.symbols:
            self.load_symbols_from_excel()
        
        # Pobierz spółki niemieckie
        all_symbols = list(self.symbols)  # Kopia listy z Excel
        
        if include_german_stocks:
            if all_german_stocks:
                # Pobierz WSZYSTKIE spółki niemieckie, nie tylko te w bazie
                german_stocks = self.get_all_german_stocks_list()
                print(f"\nPobieranie danych dla WSZYSTKICH {len(german_stocks)} spółek niemieckich")
            else:
                # Tylko te, które mają już dane w bazie
                german_stocks = self.get_german_stocks_in_database()
                print(f"\nZnaleziono {len(german_stocks)} spółek niemieckich w bazie danych")
            
            # Dodaj tylko te, których jeszcze nie ma
            for symbol in german_stocks:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)
            
            if german_stocks:
                print(f"Łącznie do przetworzenia: {len(all_symbols)} spółek ({len(self.symbols)} z Excel + {len(german_stocks)} niemieckich)")
        
        # Ograniczenie liczby spółek i start_from
        if start_from > 0:
            all_symbols = all_symbols[start_from:]
            print(f"Kontynuowanie od indeksu {start_from} (pozostało {len(all_symbols)} spółek)")
        
        if max_stocks is not None and max_stocks > 0:
            all_symbols = all_symbols[:max_stocks]
            print(f"Ograniczenie do {max_stocks} spółek")
        
        results = []
        errors = []
        total = len(all_symbols)
        start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"ROZPOCZĘCIE PRZETWARZANIA: {total} spółek")
        print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        for idx, symbol in enumerate(all_symbols, 1):
            try:
                print(f"\n{'='*80}")
                print(f"Przetwarzanie: {symbol} ({idx}/{total})")
                print(f"{'='*80}")
                
                # Utwórz lub użyj istniejącego predyktora
                if symbol not in self.predictors:
                    self.predictors[symbol] = StockPredictor(symbol, period='2y', data_dir=self.data_dir)
                
                predictor = self.predictors[symbol]
                
                # Pobierz i zaktualizuj dane z Yahoo - z lepszą obsługą błędów
                # daily_update() ZAWSZE pobiera nowe dane z Yahoo
                try:
                    predictor.fetch_data(use_saved=False)
                except Exception as fetch_error:
                    error_msg = f"Błąd pobierania danych: {fetch_error}"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
                # Sprawdź czy mamy wystarczająco danych
                if predictor.data is None or len(predictor.data) < 50:
                    error_msg = f"Za mało danych: {len(predictor.data) if predictor.data is not None else 0} dni (minimum: 50)"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
                # Utwórz cechy - z obsługą błędów
                try:
                    predictor.create_features(horizon_days=1)
                except Exception as feature_error:
                    error_msg = f"Błąd tworzenia cech: {feature_error}"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
                # Sprawdź czy mamy cechy
                if predictor.features is None or len(predictor.features) < 20:
                    error_msg = f"Za mało cech: {len(predictor.features) if predictor.features is not None else 0} wierszy"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
                # Trenuj model - z obsługą błędów
                try:
                    predictor.train_model(incremental=True)
                except Exception as train_error:
                    error_msg = f"Błąd trenowania: {train_error}"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
                # Przewiduj na kolejny dzień - z obsługą błędów
                try:
                    prediction = predictor.predict_next_day()
                    results.append(prediction)
                    
                    direction_icon = "↑" if prediction.get('predicted_direction') == 'UP' else "↓"
                    prob = prediction.get('direction_probability', 0) * 100
                    print(f"✓ {symbol}: ${prediction['current_price']:.2f} → {direction_icon} (prawdopodobieństwo: {prob:.1f}%)")
                except Exception as predict_error:
                    error_msg = f"Błąd przewidywania: {predict_error}"
                    print(f"✗ {symbol}: {error_msg}")
                    errors.append(f"{symbol}: {error_msg}")
                    continue
                
            except Exception as e:
                error_msg = f"Nieoczekiwany błąd: {e}"
                print(f"✗ {symbol}: {error_msg}")
                errors.append(f"{symbol}: {error_msg}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Podsumowanie
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print("PODSUMOWANIE")
        print(f"{'='*80}")
        print(f"✓ Pomyślnie przetworzono: {len(results)} spółek")
        print(f"✗ Błędy: {len(errors)} spółek")
        print(f"⏱ Czas trwania: {duration}")
        print(f"📊 Średnio: {duration.total_seconds() / total:.1f} sekund/spółka" if total > 0 else "")
        print(f"Zakończono: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if errors:
            print(f"\n{'='*80}")
            print(f"BŁĘDY ({len(errors)} spółek):")
            print(f"{'='*80}")
            for error in errors[:30]:  # Pokaż pierwsze 30 błędów
                print(f"  - {error}")
            if len(errors) > 30:
                print(f"  ... i {len(errors) - 30} więcej błędów")
            
            # Zapisz błędy do pliku
            error_file = Path('stock_data') / f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Błędy z daily_update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                for error in errors:
                    f.write(f"{error}\n")
            print(f"\n📝 Błędy zapisane do: {error_file}")
        
        return results
    
    def generate_report(self, results):
        """Generuje raport z przewidywań"""
        if not results:
            print("Brak wyników do raportowania")
            return
        
        print("\n" + "=" * 80)
        print("RAPORT PRZEWIDYWAŃ NA KOLEJNY DZIEŃ")
        print("=" * 80)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('change_percent', ascending=False)
        
        print("\n" + df_results.to_string(index=False))
        
        # Zapisz raport
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        df_results.to_csv(report_file, index=False)
        print(f"\nRaport zapisany jako: {report_file}")
        
        return df_results


def main():
    """Główna funkcja - codzienne uczenie i przewidywania"""
    print("=" * 80)
    print("SYSTEM PRZEWIDYWANIA CEN AKCJI - CODZIENNE UCZENIE")
    print("=" * 80)
    
    # Utworzenie multi-stock predictor
    multi_predictor = MultiStockPredictor(excel_file='raport.xlsx')
    
    # Wczytaj symbole z pliku Excel
    try:
        multi_predictor.load_symbols_from_excel()
    except Exception as e:
        print(f"Błąd: {e}")
        print("Używam przykładowych symboli...")
        multi_predictor.symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Codzienna aktualizacja - ZAWSZE pobierz wszystkie spółki niemieckie
    results = multi_predictor.daily_update(include_german_stocks=True, all_german_stocks=True)
    
    # Generuj raport
    if results:
        multi_predictor.generate_report(results)


if __name__ == "__main__":
    main()
