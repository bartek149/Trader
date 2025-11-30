"""
Moduł do logowania predykcji modelu i śledzenia ich skuteczności
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Optional, Dict, List
import yfinance as yf


class PredictionLogger:
    """Klasa do logowania predykcji i śledzenia ich wyników"""
    
    def __init__(self, data_dir='stock_data', use_sqlite=True):
        """
        Parameters:
            data_dir (str): Katalog do przechowywania logów
            use_sqlite (bool): Jeśli True, używa SQLite, w przeciwnym razie CSV
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_sqlite = use_sqlite
        
        if use_sqlite:
            self.db_path = self.data_dir / 'predictions_log.db'
            self._init_database()
        else:
            self.csv_path = self.data_dir / 'predictions_log.csv'
            self._init_csv()
    
    def _init_database(self):
        """Inicjalizacja bazy danych SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_prediction TEXT NOT NULL,
                symbol TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                price_at_prediction REAL NOT NULL,
                proba_up REAL NOT NULL,
                decision TEXT NOT NULL,
                model_version TEXT,
                outcome_filled INTEGER DEFAULT 0,
                price_at_outcome REAL,
                return_realized REAL,
                return_pct REAL,
                was_correct INTEGER
            )
        ''')
        
        # Indeksy dla szybkiego wyszukiwania
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome_filled ON predictions(outcome_filled)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp_prediction)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_horizon ON predictions(symbol, horizon_days)')
        
        conn.commit()
        conn.close()
    
    def _init_csv(self):
        """Inicjalizacja pliku CSV"""
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=[
                'id', 'timestamp_prediction', 'symbol', 'horizon_days',
                'price_at_prediction', 'proba_up', 'decision', 'model_version',
                'outcome_filled', 'price_at_outcome', 'return_realized', 'was_correct'
            ])
            df.to_csv(self.csv_path, index=False)
    
    def _find_existing_prediction_today(
        self,
        symbol: str,
        horizon_days: int
    ) -> Optional[int]:
        """
        Sprawdza czy już istnieje predykcja dla danego symbolu i horyzontu na dzisiaj
        
        Returns:
            int: ID istniejącej predykcji lub None
        """
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time()).isoformat()
        today_end = datetime.combine(today, datetime.max.time()).isoformat()
        
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM predictions
                WHERE symbol = ? 
                  AND horizon_days = ?
                  AND timestamp_prediction >= ?
                  AND timestamp_prediction <= ?
                ORDER BY timestamp_prediction DESC
                LIMIT 1
            ''', (symbol, horizon_days, today_start, today_end))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
        else:
            # CSV
            if not self.csv_path.exists():
                return None
            
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                return None
            
            # Konwertuj timestamp_prediction na datę
            df['prediction_date'] = pd.to_datetime(df['timestamp_prediction']).dt.date
            
            # Znajdź predykcję dla dzisiaj
            existing = df[
                (df['symbol'] == symbol) &
                (df['horizon_days'] == horizon_days) &
                (df['prediction_date'] == today)
            ]
            
            if len(existing) > 0:
                return int(existing.iloc[0]['id'])
            return None
    
    @staticmethod
    def _add_business_days(start_date: datetime.date, business_days: int) -> datetime.date:
        """Dodaje określoną liczbę dni roboczych do daty."""
        current = start_date
        days_added = 0
        direction = 1 if business_days >= 0 else -1
        business_days = abs(business_days)
        
        while days_added < business_days:
            current += timedelta(days=direction)
            # Monday=0, Sunday=6
            if current.weekday() < 5:
                days_added += 1
        return current

    @staticmethod
    def _find_first_trading_day(symbol: str, target_date: datetime.date, max_lookahead: int = 5) -> Optional[datetime.date]:
        """Znajdź pierwszy dzień handlowy >= target_date."""
        try:
            ticker = yf.Ticker(symbol)
            start_date = target_date
            end_date = target_date + timedelta(days=max_lookahead)
            data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index).date
            valid_dates = [d for d in data.index if d >= target_date]
            return min(valid_dates) if valid_dates else None
        except Exception:
            return None

    def log_prediction(
        self,
        symbol: str,
        horizon_days: int,
        price_now: float,
        proba_up: float,
        decision: str,
        model_version: Optional[str] = None
    ) -> int:
        """
        Loguje predykcję modelu (tylko raz dziennie - jeśli istnieje, nadpisuje)
        
        Parameters:
            symbol: Ticker spółki
            horizon_days: Horyzont predykcji w dniach
            price_now: Cena w momencie predykcji
            proba_up: Prawdopodobieństwo wzrostu (0-1)
            decision: BUY / SELL / NO_TRADE
            model_version: Wersja modelu (np. data trenowania)
        
        Returns:
            int: ID zapisanego/aktualizowanego rekordu
        """
        timestamp = datetime.utcnow().isoformat()
        prediction_date = datetime.fromisoformat(timestamp).date()
        
        # Sprawdź czy już istnieje predykcja na dzisiaj
        existing_id = self._find_existing_prediction_today(symbol, horizon_days)
        
        if existing_id is not None:
            # Aktualizuj istniejącą predykcję (ale zachowaj outcome_filled jeśli już jest wypełnione)
            if self.use_sqlite:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Sprawdź czy outcome_filled jest już 1 (nie nadpisuj jeśli wynik już jest)
                cursor.execute('SELECT outcome_filled FROM predictions WHERE id = ?', (existing_id,))
                result = cursor.fetchone()
                outcome_filled = result[0] if result else 0
                
                if outcome_filled == 0:
                    cursor.execute('''
                        UPDATE predictions
                        SET price_at_prediction = ?,
                            proba_up = ?,
                            decision = ?,
                            model_version = ?
                        WHERE id = ?
                    ''', (price_now, proba_up, decision, model_version, existing_id))
                
                conn.commit()
                conn.close()
                return existing_id
            else:
                # CSV
                df = pd.read_csv(self.csv_path)
                idx = df[df['id'] == existing_id].index
                
                if len(idx) > 0:
                    # Sprawdź czy outcome_filled jest już 1
                    outcome_filled = df.loc[idx[0], 'outcome_filled']
                    
                    if outcome_filled == 0:
                        df.loc[idx[0], 'price_at_prediction'] = price_now
                        df.loc[idx[0], 'proba_up'] = proba_up
                        df.loc[idx[0], 'decision'] = decision
                        df.loc[idx[0], 'model_version'] = model_version
                        df.to_csv(self.csv_path, index=False)
                
                return existing_id
        
        # Nie istnieje - utwórz nową predykcję
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp_prediction, symbol, horizon_days, price_at_prediction,
                 proba_up, decision, model_version, outcome_filled, price_at_outcome, return_realized, return_pct, was_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL)
            ''', (timestamp, symbol, horizon_days, price_now, proba_up, decision, model_version))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return prediction_id
        else:
            # CSV
            df = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
            
            new_row = {
                'id': len(df) + 1 if len(df) > 0 else 1,
                'timestamp_prediction': timestamp,
                'symbol': symbol,
                'horizon_days': horizon_days,
                'price_at_prediction': price_now,
                'proba_up': proba_up,
                'decision': decision,
                'model_version': model_version,
                'outcome_filled': 0,
                'price_at_outcome': None,
                'return_realized': None,
                'was_correct': None
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            return new_row['id']
    
    def get_unfilled_predictions(self) -> pd.DataFrame:
        """Pobiera wszystkie predykcje bez wypełnionych wyników"""
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                'SELECT * FROM predictions WHERE outcome_filled = 0',
                conn
            )
            conn.close()
            return df
        else:
            df = pd.read_csv(self.csv_path)
            return df[df['outcome_filled'] == 0]
    
    def update_outcome(
        self,
        prediction_id: int,
        price_at_outcome: float,
        return_realized: float,
        was_correct: int
    ):
        """Aktualizuje wynik predykcji"""
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT price_at_prediction FROM predictions WHERE id = ?', (prediction_id,))
            row = cursor.fetchone()
            price_at_prediction = row[0] if row else None
            if price_at_prediction is None or price_at_prediction == 0:
                price_at_prediction = 1.0
            
            cursor.execute('''
                UPDATE predictions
                SET outcome_filled = 1,
                    price_at_outcome = ?,
                    return_realized = ?,
                    return_pct = ?,
                    was_correct = ?
                WHERE id = ?
            ''', (price_at_outcome, return_realized, ((price_at_outcome / price_at_prediction) - 1) * 100, was_correct, prediction_id))
            
            conn.commit()
            conn.close()
        else:
            df = pd.read_csv(self.csv_path)
            idx = df[df['id'] == prediction_id].index
            if len(idx) > 0:
                df.loc[idx[0], 'outcome_filled'] = 1
                df.loc[idx[0], 'price_at_outcome'] = price_at_outcome
                df.loc[idx[0], 'return_realized'] = return_realized
                price_at_prediction = df.loc[idx[0], 'price_at_prediction']
                if not price_at_prediction:
                    price_at_prediction = 1.0
                df.loc[idx[0], 'return_pct'] = ((price_at_outcome / price_at_prediction) - 1) * 100
                df.loc[idx[0], 'was_correct'] = was_correct
                df.to_csv(self.csv_path, index=False)
    
    def get_predictions_for_calibration(
        self,
        symbol: Optional[str] = None,
        horizon_days: Optional[int] = None,
        model_version: Optional[str] = None,
        min_predictions: int = 50
    ) -> pd.DataFrame:
        """
        Pobiera predykcje do kalibracji
        
        Parameters:
            symbol: Filtr po symbolu (None = wszystkie)
            horizon_days: Filtr po horyzoncie (None = wszystkie)
            model_version: Filtr po wersji modelu (None = wszystkie)
            min_predictions: Minimalna liczba predykcji
        
        Returns:
            DataFrame z predykcjami
        """
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            
            query = 'SELECT * FROM predictions WHERE outcome_filled = 1'
            params = []
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if horizon_days:
                query += ' AND horizon_days = ?'
                params.append(horizon_days)
            
            if model_version:
                query += ' AND model_version = ?'
                params.append(model_version)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df['outcome_filled'] == 1]
            
            if symbol:
                df = df[df['symbol'] == symbol]
            if horizon_days:
                df = df[df['horizon_days'] == horizon_days]
            if model_version:
                df = df[df['model_version'] == model_version]
        
        if len(df) < min_predictions:
            return pd.DataFrame()
        
        return df
    
    def get_statistics(
        self,
        symbol: Optional[str] = None,
        horizon_days: Optional[int] = None,
        model_version: Optional[str] = None
    ) -> Dict:
        """
        Oblicza statystyki skuteczności predykcji
        
        Returns:
            Dict ze statystykami
        """
        df = self.get_predictions_for_calibration(
            symbol=symbol,
            horizon_days=horizon_days,
            model_version=model_version,
            min_predictions=1
        )
        
        if len(df) == 0:
            return {
                'total_predictions': 0,
                'hit_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'num_trades': 0
            }
        
        hit_rate = df['was_correct'].mean() if 'was_correct' in df.columns else 0.0
        avg_return = df['return_realized'].mean() if 'return_realized' in df.columns else 0.0
        total_return = df['return_realized'].sum() if 'return_realized' in df.columns else 0.0
        
        # Liczba transakcji (BUY lub SELL, nie NO_TRADE)
        num_trades = len(df[df['decision'].isin(['BUY', 'SELL'])])
        
        return {
            'total_predictions': len(df),
            'hit_rate': float(hit_rate),
            'avg_return': float(avg_return),
            'total_return': float(total_return),
            'num_trades': int(num_trades)
        }


def update_prediction_outcomes(logger: PredictionLogger, max_days_lookback: int = 30):
    """
    Uzupełnia wyniki dla predykcji, które już powinny mieć wyniki
    
    Parameters:
        logger: Instancja PredictionLogger
        max_days_lookback: Maksymalna liczba dni wstecz do sprawdzenia
    """
    print("Aktualizowanie wyników predykcji...")
    
    unfilled = logger.get_unfilled_predictions()
    
    if len(unfilled) == 0:
        print("Brak predykcji do aktualizacji")
        return
    
    print(f"Znaleziono {len(unfilled)} predykcji bez wyników")
    
    today = datetime.now().date()
    updated_count = 0
    error_count = 0
    
    for idx, row in unfilled.iterrows():
        try:
            prediction_id = row['id']
            timestamp_pred = datetime.fromisoformat(row['timestamp_prediction'])
            symbol = row['symbol']
            horizon_days = int(row['horizon_days'])
            price_at_pred = float(row['price_at_prediction'])
            
            # Oblicz datę wyniku (dni robocze)
            # Dodaj tylko dni robocze
            initial_outcome = PredictionLogger._add_business_days(timestamp_pred.date(), horizon_days)
            outcome_date = PredictionLogger._find_first_trading_day(symbol, initial_outcome)
            if outcome_date is None:
                continue
            
            # Jeśli wynik jest w przyszłości, pomiń
            if outcome_date > today:
                continue
            
            # Pobierz cenę dla outcome_date (z zapasem +/- 2 dni)
            try:
                ticker = yf.Ticker(symbol)
                start_date = (outcome_date - timedelta(days=2)).strftime('%Y-%m-%d')
                end_date = (outcome_date + timedelta(days=2)).strftime('%Y-%m-%d')
                
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    # Spróbuj pobrać ostatnią dostępną cenę
                    data = ticker.history(period='5d')
                
                if data.empty:
                    print(f"  ⚠ Nie można pobrać danych dla {symbol} na {outcome_date}")
                    error_count += 1
                    continue
                
                data.index = pd.to_datetime(data.index).date
                future_or_equal = [d for d in data.index if d >= outcome_date]
                if future_or_equal:
                    chosen_date = min(future_or_equal)
                else:
                    chosen_date = max(data.index)
                price_at_outcome = float(data.loc[chosen_date, 'Close'])
                
                # Oblicz zwrot
                return_realized = np.log(price_at_outcome / price_at_pred)
                
                # Sprawdź czy kierunek się zgadzał
                proba_up = float(row['proba_up'])
                expected_up = proba_up >= 0.5
                actual_up = return_realized > 0
                was_correct = 1 if (expected_up == actual_up) else 0
                
                # Zaktualizuj wynik
                logger.update_outcome(prediction_id, price_at_outcome, return_realized, was_correct)
                updated_count += 1
                
                if updated_count % 10 == 0:
                    print(f"  Zaktualizowano {updated_count} predykcji...")
                    
            except Exception as e:
                print(f"  ✗ Błąd dla {symbol} (ID {prediction_id}): {e}")
                error_count += 1
                continue
                
        except Exception as e:
            print(f"  ✗ Błąd przetwarzania wiersza {idx}: {e}")
            error_count += 1
            continue
    
    print(f"\n✓ Zaktualizowano {updated_count} predykcji")
    if error_count > 0:
        print(f"✗ Błędy: {error_count}")

