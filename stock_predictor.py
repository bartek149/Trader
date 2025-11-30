import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import json
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

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
        if '.DE' in symbol:
            base = symbol.replace('.DE', '')
            # Dla giełdy niemieckiej, yfinance używa formatu .F (XETR)
            variants.extend([f"{base}.F", base, f"{base}.DE"])
        
        # Dla symboli .US, spróbuj bez sufiksu
        if '.US' in symbol:
            base = symbol.replace('.US', '')
            variants.extend([base])
        
        # Dla surowców (np. NATGAS), spróbuj różne formaty
        if symbol.upper() in ['NATGAS', 'GAS', 'NATURALGAS']:
            variants.extend(['NG=F', 'NG1=F', 'NATGAS=X'])
        
        return variants
    
    def fetch_data(self, use_saved=True):
        """Pobieranie danych historycznych akcji"""
        print(f"Pobieranie danych dla {self.symbol}...")
        
        # Spróbuj załadować zapisane dane
        if use_saved:
            historical = self.load_historical_data()
            if historical is not None and len(historical) > 0:
                last_date = historical.index[-1]
                # Normalizuj timezone - upewnij się, że oba są timezone-naive
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                today = pd.Timestamp.now().normalize()
                if today.tz is not None:
                    today = today.tz_localize(None)
                
                # Jeśli mamy dane z dzisiaj, użyj ich
                if last_date >= today - timedelta(days=1):
                    print(f"Używam zapisanych danych (ostatnia data: {last_date.date()})")
                    self.data = historical
                    return self.data
                
                # Pobierz tylko nowe dane od ostatniej daty
                print(f"Pobieranie nowych danych od {last_date.date()}...")
                variants = self.convert_symbol_for_yfinance(self.symbol)
                new_data = None
                
                for variant in variants:
                    try:
                        ticker = yf.Ticker(variant)
                        new_data = ticker.history(start=last_date + timedelta(days=1))
                        if not new_data.empty:
                            print(f"Pobrano dane używając symbolu: {variant}")
                            break
                    except Exception as e:
                        print(f"  Próba {variant} nie powiodła się: {str(e)[:100]}")
                        continue
                
                if new_data is not None and not new_data.empty:
                    self.data = self.save_historical_data(new_data)
                    return self.data
                else:
                    print("Brak nowych danych, używam zapisanych")
                    self.data = historical
                    return self.data
        
        # Pobierz wszystkie dane - spróbuj różne warianty symbolu
        variants = self.convert_symbol_for_yfinance(self.symbol)
        self.data = None
        
        for variant in variants:
            try:
                ticker = yf.Ticker(variant)
                data = ticker.history(period=self.period)
                if not data.empty:
                    print(f"Pobrano dane używając symbolu: {variant}")
                    self.data = data
                    break
            except Exception as e:
                print(f"  Próba {variant} nie powiodła się: {str(e)[:100]}")
                continue
        
        if self.data is None or self.data.empty:
            raise ValueError(f"Nie udało się pobrać danych dla {self.symbol} (próbowano: {', '.join(variants)})")
        
        # Normalizuj timezone
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)
        
        # Zapisz dane
        if use_saved:
            self.data = self.save_historical_data(self.data)
        
        print(f"Pobrano {len(self.data)} dni danych")
        return self.data
    
    def create_features(self):
        """Tworzenie cech (features) dla modelu - klasyfikacja kierunku UP/DOWN"""
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
        
        # 14. Momentum
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # ========== PRO-TIPS ==========
        
        # 15. Window statistics
        df['Roll_mean_10'] = df['Close'].rolling(10).mean()
        df['Roll_std_10'] = df['Close'].rolling(10).std()
        
        # 16. Keltner Channels (lepsze od Bollingera)
        df['KC_upper'] = df['MA_20'] + df['ATR'] * 2
        df['KC_lower'] = df['MA_20'] - df['ATR'] * 2
        
        # ========== TARGET: KIERUNEK (UP/DOWN) zamiast ceny ==========
        # 1 = UP (cena wzrośnie), 0 = DOWN (cena spadnie)
        df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Wybierz cechy do modelu
        feature_columns = [
            # Podstawowe ceny
            'Open', 'High', 'Low', 'Close', 'Volume',
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
            # Wolumen
            'Volume_ratio', 'Volume_rate',
            # Inne wskaźniki
            'High_Low_ratio', 'Open_Close_ratio',
            'ATR', 'TR',
            # Window statistics
            'Roll_mean_10', 'Roll_std_10',
            # Time features
            'Day_of_week', 'Month'
        ]
        
        # Upewnij się, że wszystkie kolumny istnieją
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features + ['Direction']].copy()
        
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
        self.target = features_df['Direction']
        
        # Sprawdź czy mamy wystarczająco danych
        if len(self.features) < 50:
            raise ValueError(f"Za mało danych po czyszczeniu: {len(self.features)} wierszy. Minimum: 50")
        
        return features_df
    
    def train_model(self, test_size=0.2, random_state=42, incremental=False):
        """Trenowanie modelu"""
        if self.features is None:
            raise ValueError("Najpierw utwórz cechy używając create_features()")
        
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
        
        # Zaktualizuj self.features i self.target oczyszczonymi danymi
        self.features = X_clean
        self.target = y_clean
        
        # Jeśli incremental learning, załaduj poprzedni model
        if incremental and self.model_file.exists():
            try:
                # Dla uproszczenia, zawsze trenujemy od nowa, ale możemy dodać incremental learning później
                print("Uwaga: Incremental learning nie jest jeszcze zaimplementowane, trenuję od nowa")
            except:
                pass
        
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Trenowanie modelu Random Forest CLASSIFIER (nie regressor!)
        print("Trenowanie modelu klasyfikacji kierunku...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Ważne dla niezbalansowanych danych
        )
        
        self.model.fit(X_train, y_train)
        
        # Ocena modelu - metryki klasyfikacji
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_precision = precision_score(y_train, train_pred, zero_division=0)
        test_precision = precision_score(y_test, test_pred, zero_division=0)
        train_recall = recall_score(y_train, train_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        
        print(f"\nWyniki modelu klasyfikacji dla {self.symbol}:")
        print(f"Zbiór treningowy:")
        print(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1-Score: {train_f1:.4f}")
        print(f"Zbiór testowy:")
        print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        
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
            'feature_count': len(self.features.columns)
        }
        
        with open(self.model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Zapisz model do pliku pickle
        with open(self.model_pickle_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        return self.model
    
    def load_model(self):
        """Załaduj zapisany model z pliku"""
        if self.model_pickle_file.exists() and self.model_file.exists():
            try:
                with open(self.model_pickle_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Załaduj też informacje o modelu
                with open(self.model_file, 'r') as f:
                    model_info = json.load(f)
                
                print(f"Załadowano zapisany model dla {self.symbol}")
                return self.model, model_info
            except Exception as e:
                print(f"Błąd przy ładowaniu modelu dla {self.symbol}: {e}")
                return None, {}
        return None, {}
    
    def predict_next_day(self):
        """Przewidywanie kierunku na kolejny dzień (UP/DOWN)"""
        return self.predict_period(days=1)
    
    def predict_period(self, days=1):
        """Przewidywanie kierunku i ceny na określony okres (w dniach)
        
        Parameters:
            days (int): Liczba dni do przodu (1 = jutro, 7 = tydzień, 30 = miesiąc, 180 = 6 miesięcy)
        
        Returns:
            dict: Słownik z przewidywaniami zawierający:
                - symbol: symbol spółki
                - current_price: aktualna cena
                - predicted_price: przewidywana cena
                - predicted_direction: 'UP' lub 'DOWN'
                - direction_probability: prawdopodobieństwo przewidywanego kierunku
                - change: zmiana ceny w $
                - change_percent: zmiana ceny w %
                - period_days: liczba dni przewidywania
        """
        if self.model is None:
            raise ValueError("Najpierw wytrenuj model używając train_model()")
        
        # Ostatni wiersz danych (najnowsze dane)
        last_features = self.features.iloc[-1:].values
        
        # Przewiduj kierunek (0 = DOWN, 1 = UP)
        direction_pred = self.model.predict(last_features)[0]
        
        # Prawdopodobieństwo wzrostu
        direction_proba = self.model.predict_proba(last_features)[0]
        up_probability = direction_proba[1] if len(direction_proba) > 1 else 0.5
        
        current_price = self.data['Close'].iloc[-1]
        
        # Oblicz średnie zmiany historyczne dla różnych okresów
        historical_changes = self.data['Close'].pct_change().dropna()
        
        # Dla krótkich okresów (1-7 dni) użyj dziennych zmian
        if days <= 7:
            avg_up_change = historical_changes[historical_changes > 0].mean() if len(historical_changes[historical_changes > 0]) > 0 else 0.01
            avg_down_change = historical_changes[historical_changes < 0].mean() if len(historical_changes[historical_changes < 0]) > 0 else -0.01
            # Skaluj na liczbę dni
            avg_up_change = avg_up_change * days
            avg_down_change = avg_down_change * days
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
        
        # Zastosuj przewidywany kierunek z wagą prawdopodobieństwa
        if direction_pred == 1:  # UP
            # Mieszanka wzrostu i spadku w zależności od prawdopodobieństwa
            estimated_change_pct = (avg_up_change * up_probability) + (avg_down_change * (1 - up_probability))
        else:  # DOWN
            # Mieszanka spadku i wzrostu w zależności od prawdopodobieństwa
            estimated_change_pct = (avg_down_change * (1 - up_probability)) + (avg_up_change * up_probability)
        
        estimated_price = current_price * (1 + estimated_change_pct)
        change = estimated_price - current_price
        change_percent = estimated_change_pct * 100
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'predicted_price': estimated_price,
            'predicted_direction': 'UP' if direction_pred == 1 else 'DOWN',
            'direction_probability': up_probability if direction_pred == 1 else (1 - up_probability),
            'change': change,
            'change_percent': change_percent,
            'period_days': days
        }
    
    def predict_week(self):
        """Przewidywanie na tydzień (7 dni)"""
        return self.predict_period(days=7)
    
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
            # DAX 40
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'MUV2.DE', 'DBK.DE',
            'BAYN.DE', 'BMW.DE', 'VOW3.DE', 'IFX.DE', 'DTE.DE',
            'MRK.DE', 'RWE.DE', 'ENR.DE', 'AIR.DE', 'MTX.DE',
            'FRE.DE', 'HEN3.DE', 'VNA.DE', 'EOAN.DE', 'BAS.DE',
            'CON.DE', 'SHL.DE', 'ZAL.DE', 'PUM.DE', '1COV.DE',
            'PAH3.DE', 'HEI.DE', 'QIA.DE', 'RHM.DE', 'SY1.DE',
            'ADS.DE', 'HNR1.DE', 'BNR.DE', 'DHL.DE', 'MBG.DE',
            'DHER.DE', 'LIN.DE', 'P911.DE', 'NDX1.DE', 'KCO.DE',
            # MDAX
            'ABEA.DE', 'ADN.DE', 'AFX.DE', 'AOX.DE', 'ARL.DE',
            'AT1.DE', 'BC8.DE', 'BIO3.DE', 'BVB.DE', 'BYW6.DE',
            'CEC.DE', 'CLS.DE', 'COP.DE', 'DIC.DE', 'DRI.DE',
            'DUE.DE', 'EVD.DE', 'EVK.DE', 'FNTN.DE', 'G1A.DE',
            'G24.DE', 'GIL.DE', 'GLJ.DE', 'GXI.DE', 'HLE.DE',
            'HOT.DE', 'JEN.DE', 'KGX.DE', 'KRN.DE', 'LEG.DE',
            'LEO.DE', 'LXS.DE', 'M5Z.DE', 'MOR.DE', 'NDA.DE',
            'NOEJ.DE', 'OSR.DE', 'PFV.DE', 'PSM.DE',
            'RAA.DE', 'RRTL.DE', 'SAX.DE', 'SBS.DE',
            'SDF.DE', 'SGL.DE', 'SIX2.DE', 'SKB.DE', 'SNH.DE',
            # SDAX i TecDAX
            'A1OS.DE', 'ACX.DE', 'ADJ.DE', 'ADL.DE', 'AHC.DE',
            'AIXA.DE', 'ALT.DE', 'AM3D.DE', 'AOF.DE', 'APM.DE',
            'ARZ.DE', 'ASL.DE', 'ATN.DE', 'AUR.DE', 'B5A.DE',
            'B8A.DE', 'B9B.DE', 'BAG.DE', 'BAN.DE', 'BAT.DE',
            'BBZA.DE', 'BCO.DE', 'BDF.DE',
            'BEI.DE', 'BKS.DE', 'BLH.DE',
            'BOS.DE', 'BPE5.DE', 'BRM.DE', 'BSL.DE',
            'BTBB.DE', 'BWO.DE', 'BZR.DE', 'C1V.DE',
            'CAJ.DE', 'CAP.DE', 'CAR.DE', 'CAS.DE', 'CAT1.DE',
            'CBK.DE', 'CEC1.DE', 'CEV.DE', 'CFR.DE'
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
                
                # Pobierz i zaktualizuj dane - z lepszą obsługą błędów
                try:
                    predictor.fetch_data(use_saved=True)
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
                    predictor.create_features()
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
