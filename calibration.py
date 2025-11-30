"""
Moduł do kalibracji prawdopodobieństw i strojenia progów decyzyjnych
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from prediction_logger import PredictionLogger


def train_calibrator(
    logger: PredictionLogger,
    symbol: Optional[str] = None,
    horizon_days: Optional[int] = None,
    model_version: Optional[str] = None,
    calibrator_type: str = 'isotonic',
    save_path: Optional[Path] = None
) -> Optional[object]:
    """
    Trenuje kalibrator prawdopodobieństw na podstawie logu predykcji
    
    Parameters:
        logger: Instancja PredictionLogger
        symbol: Filtr po symbolu (None = wszystkie)
        horizon_days: Filtr po horyzoncie (None = wszystkie)
        model_version: Filtr po wersji modelu (None = wszystkie)
        calibrator_type: 'isotonic' lub 'logistic'
        save_path: Ścieżka do zapisania kalibratora
    
    Returns:
        Wytrenowany kalibrator lub None
    """
    # Pobierz dane do kalibracji
    df = logger.get_predictions_for_calibration(
        symbol=symbol,
        horizon_days=horizon_days,
        model_version=model_version,
        min_predictions=50
    )
    
    if len(df) == 0:
        print(f"⚠ Za mało danych do kalibracji (wymagane: 50, znaleziono: 0)")
        return None
    
    # Przygotuj dane
    X = df['proba_up'].values.reshape(-1, 1)
    y = (df['return_realized'] > 0).astype(int).values  # 1 jeśli zwrot > 0
    
    # Trenuj kalibrator
    if calibrator_type == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif calibrator_type == 'logistic':
        calibrator = LogisticRegression()
    else:
        raise ValueError(f"Nieznany typ kalibratora: {calibrator_type}")
    
    calibrator.fit(X, y)
    
    print(f"✓ Wytrenowano kalibrator ({calibrator_type}) na {len(df)} predykcjach")
    
    # Zapisz kalibrator
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(calibrator, f)
        print(f"  Zapisano do: {save_path}")
    
    return calibrator


def load_calibrator(calibrator_path: Path) -> Optional[object]:
    """Ładuje zapisany kalibrator"""
    if not calibrator_path.exists():
        return None
    
    try:
        with open(calibrator_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Błąd przy ładowaniu kalibratora: {e}")
        return None


def calibrate_probability(calibrator: object, proba_up: float) -> float:
    """
    Kalibruje prawdopodobieństwo używając kalibratora
    
    Parameters:
        calibrator: Wytrenowany kalibrator
        proba_up: Surowe prawdopodobieństwo (0-1)
    
    Returns:
        Sklorygowane prawdopodobieństwo (0-1)
    """
    if calibrator is None:
        return proba_up
    
    try:
        calibrated = calibrator.predict(np.array([[proba_up]]))[0]
        # Upewnij się, że wynik jest w zakresie [0, 1]
        return np.clip(calibrated, 0.0, 1.0)
    except Exception as e:
        print(f"Błąd przy kalibracji: {e}")
        return proba_up


def find_best_threshold(
    logger: PredictionLogger,
    symbol: Optional[str] = None,
    horizon_days: Optional[int] = None,
    model_version: Optional[str] = None,
    strategy: str = 'long_only',
    transaction_cost: float = 0.001,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Znajduje optymalny próg decyzyjny maksymalizujący zwrot
    
    Parameters:
        logger: Instancja PredictionLogger
        symbol: Filtr po symbolu (None = wszystkie)
        horizon_days: Filtr po horyzoncie (None = wszystkie)
        model_version: Filtr po wersji modelu (None = wszystkie)
        strategy: 'long_only' (tylko BUY) lub 'long_short' (BUY i SELL)
        transaction_cost: Koszt transakcji (0.001 = 0.1%)
        save_path: Ścieżka do zapisania wyników (JSON)
    
    Returns:
        Dict z optymalnym progiem i metrykami
    """
    # Pobierz dane
    df = logger.get_predictions_for_calibration(
        symbol=symbol,
        horizon_days=horizon_days,
        model_version=model_version,
        min_predictions=50
    )
    
    if len(df) == 0:
        print(f"⚠ Za mało danych do strojenia progu (wymagane: 50, znaleziono: 0)")
        return {'best_threshold': 0.65, 'best_metric': 0.0}
    
    # Siatka progów
    thresholds = np.arange(0.50, 0.85, 0.01)
    results = []
    
    for threshold in thresholds:
        # Generuj sygnały
        if strategy == 'long_only':
            signals = df['proba_up'].apply(
                lambda p: 'BUY' if p > threshold else 'NO_TRADE'
            )
        else:  # long_short
            signals = df['proba_up'].apply(
                lambda p: 'BUY' if p > threshold else ('SELL' if p < (1 - threshold) else 'NO_TRADE')
            )
        
        # Filtruj tylko transakcje (nie NO_TRADE)
        trades = df[signals.isin(['BUY', 'SELL'])].copy()
        trades_signals = signals[signals.isin(['BUY', 'SELL'])]
        
        if len(trades) == 0:
            results.append({
                'threshold': threshold,
                'num_trades': 0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'sharpe': 0.0,
                'hit_rate': 0.0
            })
            continue
        
        # Oblicz zwroty z uwzględnieniem kosztów
        returns = []
        for idx, (_, row) in enumerate(trades.iterrows()):
            signal = trades_signals.iloc[idx]
            return_val = row['return_realized']
            
            if signal == 'BUY':
                # Long: zwrot jak jest
                returns.append(return_val - transaction_cost)
            else:  # SELL
                # Short: odwrotny zwrot
                returns.append(-return_val - transaction_cost)
        
        returns = np.array(returns)
        
        # Metryki
        num_trades = len(returns)
        avg_return = np.mean(returns)
        total_return = np.sum(returns)
        
        # Sharpe ratio (uproszczony)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        results.append({
            'threshold': threshold,
            'num_trades': num_trades,
            'avg_return': float(avg_return),
            'total_return': float(total_return),
            'sharpe': float(sharpe),
            'hit_rate': float(hit_rate)
        })
    
    # Znajdź najlepszy próg (maksymalizujący Sharpe lub avg_return)
    results_df = pd.DataFrame(results)
    
    # Filtruj progi z wystarczającą liczbą transakcji (min 10)
    results_df = results_df[results_df['num_trades'] >= 10]
    
    if len(results_df) == 0:
        print("⚠ Brak progów z wystarczającą liczbą transakcji")
        return {'best_threshold': 0.65, 'best_metric': 0.0}
    
    # Wybierz próg maksymalizujący Sharpe (lub avg_return jeśli Sharpe <= 0)
    best_idx_sharpe = results_df['sharpe'].idxmax()
    best_idx_return = results_df['avg_return'].idxmax()
    
    if results_df.loc[best_idx_sharpe, 'sharpe'] > 0:
        best_idx = best_idx_sharpe
        best_metric = results_df.loc[best_idx, 'sharpe']
        metric_name = 'sharpe'
    else:
        best_idx = best_idx_return
        best_metric = results_df.loc[best_idx, 'avg_return']
        metric_name = 'avg_return'
    
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_result = results_df.loc[best_idx].to_dict()
    
    print(f"✓ Najlepszy próg: {best_threshold:.2f} ({metric_name}={best_metric:.4f})")
    print(f"  Liczba transakcji: {best_result['num_trades']}")
    print(f"  Średni zwrot: {best_result['avg_return']:.4f}")
    print(f"  Hit rate: {best_result['hit_rate']:.2%}")
    
    result = {
        'best_threshold': float(best_threshold),
        'best_metric': float(best_metric),
        'metric_name': metric_name,
        'best_result': best_result,
        'all_results': results
    }
    
    # Zapisz wyniki
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Zapisano do: {save_path}")
    
    return result


def load_best_threshold(threshold_path: Path) -> Optional[float]:
    """Ładuje zapisany optymalny próg"""
    if not threshold_path.exists():
        return None
    
    try:
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            return data.get('best_threshold', 0.65)
    except Exception as e:
        print(f"Błąd przy ładowaniu progu: {e}")
        return None

