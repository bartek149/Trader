import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import yfinance as yf
import threading
import subprocess
import sys
import time
import streamlit.components.v1 as components
from stock_predictor import MultiStockPredictor, StockPredictor

# Import modułów logowania i kalibracji (opcjonalnie)
try:
    from prediction_logger import PredictionLogger, update_prediction_outcomes
    from calibration import train_calibrator, find_best_threshold, load_calibrator, load_best_threshold
    PREDICTION_LOGGING_AVAILABLE = True
except ImportError:
    PREDICTION_LOGGING_AVAILABLE = False
    PredictionLogger = None

def clean_dataframe_for_streamlit(df):
    """Czyści DataFrame z wartości inf, -inf i NaN aby był kompatybilny z Arrow/Streamlit"""
    df = df.copy()
    
    # Zastąp inf i -inf przez NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Dla kolumn numerycznych, zastąp NaN przez 0 lub odpowiednią wartość
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'object':
            # Dla kolumn tekstowych, zastąp NaN przez pusty string
            df[col] = df[col].fillna('')
    
    # Upewnij się, że wszystkie kolumny mają odpowiednie typy
    for col in df.columns:
        if df[col].dtype == 'object':
            # Spróbuj przekonwertować na string jeśli to możliwe
            try:
                df[col] = df[col].astype(str)
            except:
                pass
    
    return df

# Konfiguracja strony
st.set_page_config(
    page_title="Przewidywanie Cen Akcji",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_symbols():
    """Wczytaj symbole z pliku Excel i dodaj własne tickery
    
    Returns:
        tuple: (symbols_dict, multi_predictor) gdzie symbols_dict zawiera 'open', 'closed', 'all'
    """
    try:
        multi_predictor = MultiStockPredictor(excel_file='raport.xlsx')
        symbols_dict = multi_predictor.load_symbols_from_excel()
        
        # Dodaj własne tickery do listy symboli
        if 'custom_tickers' in st.session_state:
            for ticker in st.session_state.custom_tickers:
                if ticker not in symbols_dict['all']:
                    symbols_dict['all'].append(ticker)
                    symbols_dict['open'].append(ticker)  # Własne tickery traktujemy jako otwarte
        
        return symbols_dict, multi_predictor
    except Exception as e:
        st.error(f"Błąd przy wczytywaniu symboli: {e}")
        return {'open': [], 'closed': [], 'all': []}, None

def get_all_symbols_with_data(data_dir='stock_data'):
    """Znajdź wszystkie symbole, które mają pliki danych w katalogu"""
    data_path = Path(data_dir)
    symbols = set()
    
    if data_path.exists():
        # Znajdź wszystkie pliki _data.csv
        for file in data_path.glob('*_data.csv'):
            # Wyciągnij symbol z nazwy pliku (np. "AAPL_data.csv" -> "AAPL")
            symbol = file.stem.replace('_data', '')
            # Pomiń plik DAX jeśli nie jest to symbol spółki
            if symbol != 'DAX':
                symbols.add(symbol)
    
    return sorted(list(symbols))

@st.cache_data(ttl=300)  # Cache na 5 minut (krótszy czas, aby szybciej widzieć nowe dane)
def get_predictions_data(symbols=None, include_all_from_db=True, use_saved_models=True, cache_version=0):
    """Pobierz dane przewidywań dla wszystkich spółek
    
    Parameters:
    symbols: Lista symboli z Excel (opcjonalna)
    include_all_from_db: Jeśli True, pobiera dane dla WSZYSTKICH spółek z bazy, nie tylko z Excel
    use_saved_models: Jeśli True, używa zapisanych modeli zamiast trenować na nowo (szybsze, nie blokuje UI)
    """
    try:
        # Jeśli include_all_from_db=True, pobierz wszystkie symbole z bazy
        if include_all_from_db:
            all_symbols = get_all_symbols_with_data()
            # Dodaj symbole z Excel jeśli nie ma ich w bazie
            if symbols:
                for symbol in symbols:
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)
            symbols_to_process = all_symbols
        else:
            # Tylko symbole z Excel
            symbols_to_process = symbols if symbols else []
        
        if not symbols_to_process:
            return []
        
        results = []
        errors = []
        
        # Nie pokazuj progress bara jeśli jest dużo spółek (może być wolne)
        show_progress = len(symbols_to_process) <= 50
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, symbol in enumerate(symbols_to_process):
            if show_progress:
                status_text.text(f"Ładowanie {symbol} ({idx+1}/{len(symbols_to_process)})...")
                progress_bar.progress((idx + 1) / len(symbols_to_process))
            
            try:
                predictor = StockPredictor(symbol, data_dir='stock_data')
                # Użyj TYLKO lokalnych danych - nie pobieraj z Yahoo
                predictor.fetch_data(use_saved=True)
                
                # Jeśli nie ma lokalnych danych, pomiń tę spółkę
                if predictor.data is None or len(predictor.data) == 0:
                    if getattr(predictor, 'data_stale', False):
                        last_date = predictor.data_stale_last_date.date() if predictor.data_stale_last_date is not None else 'N/D'
                        errors.append(f"{symbol}: Lokalny cache został usunięty, bo ostatnie notowanie ({last_date}) jest starsze niż 7 dni. Uruchom '🚀 Uruchom Daily Update' aby pobrać świeże dane.")
                    else:
                        errors.append(f"{symbol}: Brak zapisanych danych lokalnych. Uruchom '🚀 Uruchom Daily Update' lub użyj przycisku '📥 Pobierz dane' aby pobrać dane z Yahoo.")
                    continue
                
                if predictor.data is not None and len(predictor.data) > 0:
                    # Jeśli mamy zapisany model i chcemy go użyć, załaduj go
                    if use_saved_models:
                        loaded_model, model_info = predictor.load_model()
                        if loaded_model is not None:
                            # Model załadowany, użyj go do przewidywań
                            predictor.model = loaded_model
                            # Upewnij się, że mamy cechy (z domyślnym horyzontem 1 dzień)
                            try:
                                predictor.create_features(horizon_days=1)
                            except Exception as e:
                                import traceback
                                error_details = traceback.format_exc()
                                errors.append(f"{symbol}: Błąd tworzenia cech: {e}\n{error_details[:500]}")
                                continue
                        else:
                            # Brak zapisanego modelu - nie trenuj automatycznie, tylko zwróć błąd
                            errors.append(f"{symbol}: Brak zapisanego modelu. Uruchom '🚀 Uruchom Daily Update' aby wytrenować model.")
                            continue
                    else:
                        # use_saved_models=False - nie trenuj automatycznie, tylko zwróć błąd
                        errors.append(f"{symbol}: Brak zapisanego modelu. Uruchom '🚀 Uruchom Daily Update' aby wytrenować model.")
                        continue
                    
                    # Sprawdź czy model istnieje przed przewidywaniami
                    if predictor.model is None:
                        errors.append(f"{symbol}: Brak modelu po trenowaniu")
                        continue
                    
                    # Przewidywania na różne okresy - z obsługą błędów
                    try:
                        prediction_day = predictor.predict_next_day()
                    except Exception as e:
                        errors.append(f"{symbol}: Błąd przewidywania na 1 dzień: {e}")
                        continue
                    
                    # Przewidywania na tydzień - obsługa błędów
                    try:
                        prediction_week = predictor.predict_week()
                    except Exception as e:
                        # Jeśli nie ma modelu tygodniowego, użyj standardowego przewidywania na 5 dni
                        try:
                            prediction_week = predictor.predict_period(days=5)
                        except Exception as e2:
                            # Jeśli to też nie działa, użyj domyślnych wartości
                            prediction_week = {
                                'predicted_direction': prediction_day.get('predicted_direction', 'N/A'),
                                'direction_probability': prediction_day.get('direction_probability', 0.5),
                                'predicted_price': prediction_day.get('predicted_price', prediction_day.get('current_price', 0)),
                                'change': 0,
                                'change_percent': 0
                            }
                    
                    # Przewidywania na miesiąc i 6 miesięcy
                    try:
                        prediction_month = predictor.predict_month()
                    except Exception as e:
                        try:
                            prediction_month = predictor.predict_period(days=30)
                        except:
                            prediction_month = {
                                'predicted_direction': prediction_day.get('predicted_direction', 'N/A'),
                                'direction_probability': prediction_day.get('direction_probability', 0.5),
                                'predicted_price': prediction_day.get('predicted_price', prediction_day.get('current_price', 0)),
                                'change': 0,
                                'change_percent': 0
                            }
                    
                    try:
                        prediction_6months = predictor.predict_6months()
                    except Exception as e:
                        try:
                            prediction_6months = predictor.predict_period(days=180)
                        except:
                            prediction_6months = {
                                'predicted_direction': prediction_day.get('predicted_direction', 'N/A'),
                                'direction_probability': prediction_day.get('direction_probability', 0.5),
                                'predicted_price': prediction_day.get('predicted_price', prediction_day.get('current_price', 0)),
                                'change': 0,
                                'change_percent': 0
                            }
                    
                    results.append({
                        **prediction_day,
                        'prediction_week': prediction_week,
                        'prediction_month': prediction_month,
                        'prediction_6months': prediction_6months,
                        'data_points': model_info.get('data_points', len(predictor.data)),
                        'test_accuracy': model_info.get('test_accuracy', 0),
                        'test_f1': model_info.get('test_f1', 0),
                        'test_precision': model_info.get('test_precision', 0),
                        'test_recall': model_info.get('test_recall', 0),
                        'last_trained': model_info.get('last_trained', '')
                    })
                else:
                    errors.append(f"{symbol}: Brak danych")
            except Exception as e:
                error_msg = str(e)
                errors.append(f"{symbol}: {error_msg}")
                continue
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        # Pokaż błędy jeśli są
        if errors:
            # Zawsze pokaż liczbę błędów
            if len(errors) > 50:
                # Jeśli jest dużo błędów, pokaż tylko pierwsze 50 i podsumowanie
                with st.expander(f"⚠️ Błędy ({len(errors)} spółek) - pokazano pierwsze 50", expanded=True):
                    for error in errors[:50]:
                        st.text(error)
                    st.warning(f"... i {len(errors) - 50} więcej błędów")
            else:
                with st.expander(f"⚠️ Błędy ({len(errors)} spółek)", expanded=True):
                    for error in errors:
                        st.text(error)
        
        # Jeśli nie ma wyników, pokaż informację
        if not results and errors:
            st.warning(f"⚠️ Nie udało się załadować danych dla żadnej spółki. Wszystkie {len(errors)} spółek miały błędy. Sprawdź szczegóły powyżej.")
        
        return results
    except Exception as e:
        st.error(f"Błąd przy pobieraniu danych: {e}")
        return []

def get_cached_predictions(cache_name, symbols, include_all_from_db=False):
    """Zwraca dane predykcji z lokalnego cache w session_state dla wskazanego kontekstu."""
    cache_version = st.session_state.get('cache_version', 0)
    cache_key = f'{cache_name}_predictions_cache'
    version_key = f'{cache_name}_predictions_cache_version'
    
    if cache_key not in st.session_state or st.session_state.get(version_key) != cache_version:
        predictions = get_predictions_data(
            symbols,
            include_all_from_db=include_all_from_db,
            use_saved_models=True,
            cache_version=cache_version
        )
        st.session_state[cache_key] = predictions
        st.session_state[version_key] = cache_version
    
    return st.session_state.get(cache_key, [])

def get_dashboard_predictions(symbols, include_all_from_db=False):
    return get_cached_predictions('dashboard', symbols, include_all_from_db)

def get_dax_predictions(symbols=None, include_all_from_db=True):
    return get_cached_predictions('dax', symbols, include_all_from_db)

def update_prediction_cache(cache_name, updated_prediction):
    cache_key = f'{cache_name}_predictions_cache'
    cache_list = st.session_state.get(cache_key)
    if not cache_list:
        return
    
    replaced = False
    for idx, item in enumerate(cache_list):
        if item.get('symbol') == updated_prediction.get('symbol'):
            cache_list[idx] = updated_prediction
            replaced = True
            break
    
    if not replaced:
        cache_list.append(updated_prediction)
    
    st.session_state[cache_key] = list(cache_list)

def update_all_prediction_caches(updated_prediction):
    update_prediction_cache('dashboard', updated_prediction)
    update_prediction_cache('dax', updated_prediction)

def navigate_to_details(symbol):
    """Ustawia spółkę i przełącza użytkownika na zakładkę szczegółów."""
    st.session_state.selected_symbol = symbol
    st.session_state.switch_tab_to = "📈 Szczegóły spółki"
    st.rerun()

def trigger_tab_switch(tab_label):
    """Wymusza przełączenie zakładki po stronie frontendu (hack JS)."""
    tab_text = json.dumps(tab_label)
    switch_js = f"""
    <script>
    const targetLabel = {tab_text};
    const trySwitch = () => {{
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        for (const btn of tabs) {{
            const label = (btn.innerText || btn.textContent || '').trim();
            if (label === targetLabel) {{
                btn.click();
                return;
            }}
        }}
        setTimeout(trySwitch, 100);
    }};
    setTimeout(trySwitch, 0);
    </script>
    """
    components.html(switch_js, height=0, width=0)

def get_predictor_for_symbol(symbol, use_saved_model=True):
    """Pobierz predictor dla wybranej spółki (nie cache'owany)
    
    Parameters:
    use_saved_model: Jeśli True, używa zapisanego modelu zamiast trenować na nowo
    """
    try:
        predictor = StockPredictor(symbol, data_dir='stock_data')
        predictor.fetch_data(use_saved=True)
        if predictor.data is not None and len(predictor.data) > 0:
            predictor.create_features(horizon_days=1)
            
            # Spróbuj załadować zapisany model
            if use_saved_model:
                loaded_model, _ = predictor.load_model()
                if loaded_model is not None:
                    predictor.model = loaded_model
                else:
                    # Brak zapisanego modelu - nie trenuj automatycznie
                    st.warning(f"Brak zapisanego modelu dla {symbol}. Uruchom '🚀 Uruchom Daily Update' aby wytrenować model.")
                    return None
            else:
                # use_saved_model=False - nie trenuj automatycznie
                st.warning(f"Brak zapisanego modelu dla {symbol}. Uruchom '🚀 Uruchom Daily Update' aby wytrenować model.")
                return None
            
            return predictor
    except Exception as e:
        st.error(f"Błąd przy ładowaniu {symbol}: {e}")
    return None

def run_single_symbol_update(symbol, retrain_weekly=True):
    """Pobierz dane, wytrenuj modele i odśwież cache dla pojedynczej spółki."""
    try:
        print(f"\n{'='*60}")
        print(f"[SINGLE STOCK UPDATE] Start dla {symbol}")
        print(f"{'='*60}\n")

        predictor = StockPredictor(symbol, data_dir='stock_data')
        predictor.fetch_data(use_saved=False)

        if predictor.data is None or len(predictor.data) == 0:
            return False, f"Nie udało się pobrać danych dla {symbol}"

        predictor.create_features(horizon_days=1)
        predictor.train_model(horizon_days=1)

        weekly_warning = None
        if retrain_weekly:
            try:
                predictor.train_weekly_model()
            except Exception as weekly_error:
                weekly_warning = f"Nie udało się wytrenować modelu tygodniowego: {weekly_error}"
                print(f"[SINGLE STOCK UPDATE] Ostrzeżenie: {weekly_warning}")

        single_prediction = get_predictions_data(
            symbols=[symbol],
            include_all_from_db=False,
            use_saved_models=True,
            cache_version=time.time()
        )
        if single_prediction:
            update_all_prediction_caches(single_prediction[0])

        return True, weekly_warning
    except Exception as e:
        print(f"[SINGLE STOCK UPDATE] Błąd dla {symbol}: {e}")
        return False, str(e)

def run_full_daily_update(include_german_stocks=True, all_german_stocks=True):
    """Uruchom daily_update i zwróć listę wyników."""
    update_predictor = MultiStockPredictor(excel_file='raport.xlsx')
    update_predictor.load_symbols_from_excel()
    
    if 'custom_tickers' in st.session_state:
        for ticker in st.session_state.custom_tickers:
            if ticker not in update_predictor.symbols:
                update_predictor.symbols.append(ticker)
    
    results = update_predictor.daily_update(
        include_german_stocks=include_german_stocks,
        all_german_stocks=all_german_stocks
    )
    return results

def merge_prediction_lists(primary, secondary):
    """Zwraca listę predykcji z priorytetem wartości z primary przy duplikatach."""
    merged = {}
    secondary = secondary or []
    for item in secondary:
        merged[item.get('symbol')] = item
    for item in primary or []:
        merged[item.get('symbol')] = item
    return [p for p in merged.values() if p]

GERMAN_INDEX_TICKERS = [
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

@st.cache_data(ttl=3600)
def get_dax_data():
    """Pobierz dane i przewidywania dla indeksu DAX"""
    try:
        # Symbol DAX w yfinance
        dax_symbols = ['^GDAXI', 'DAX.F']  # Próbuj różne warianty
        
        dax_data = None
        used_symbol = None
        
        for symbol in dax_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2y')
                if not data.empty:
                    dax_data = data
                    used_symbol = symbol
                    break
            except:
                continue
        
        if dax_data is None or dax_data.empty:
            return None
        
        # Normalizuj timezone
        if dax_data.index.tz is not None:
            dax_data.index = dax_data.index.tz_localize(None)
        
        # Utwórz predictor dla DAX
        predictor = StockPredictor('DAX', period='2y', data_dir='stock_data')
        predictor.data = dax_data
        predictor.create_features(horizon_days=1)
        
        # Spróbuj załadować zapisany model zamiast trenować
        loaded_model, model_info_dict = predictor.load_model()
        if loaded_model is not None:
            predictor.model = loaded_model
            # Załaduj informacje o modelu
            model_info = model_info_dict if model_info_dict else {}
        else:
            # Brak zapisanego modelu - wytrenuj go od podstaw
            predictor.train_model(horizon_days=1)
            loaded_model, model_info_dict = predictor.load_model()
            predictor.model = loaded_model
            model_info = model_info_dict if model_info_dict else {}
        
        prediction = predictor.predict_next_day()
        
        # Jeśli model_info jest puste, spróbuj załadować z pliku
        if not model_info and predictor.model_file.exists():
            with open(predictor.model_file, 'r') as f:
                model_info = json.load(f)
        
        return {
            'predictor': predictor,
            'prediction': prediction,
            'model_info': model_info,
            'data': dax_data
        }
    except Exception as e:
        st.error(f"Błąd przy pobieraniu danych DAX: {e}")
        return None

def get_dax_components():
    """Lista głównych składników DAX"""
    # Top 10 według najczęściej śledzonych spółek
    return [
        "SAP.DE", "SIE.DE", "ALV.DE", "MUV2.DE", "BAYN.DE",
        "BMW.DE", "IFX.DE", "DTE.DE", "VOW3.DE", "RWE.DE"
    ]

@st.cache_data(ttl=86400)  # Cache na 24 godziny
def get_all_german_stocks():
    """Lista wszystkich głównych spółek niemieckich"""
    all_stocks = sorted(set(GERMAN_INDEX_TICKERS))
    dax_top = get_dax_components()
    return {
        'all': all_stocks,
        'dax': dax_top,
        'mdax': all_stocks,   # Uproszczenie: wykorzystujemy pełną listę do filtrowania
        'sdax': all_stocks,
        'tecdax': all_stocks
    }

def init_favorites():
    """Inicjalizuj listę ulubionych w session_state"""
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    # Załaduj z pliku jeśli istnieje
    favorites_file = Path('favorites.json')
    if favorites_file.exists():
        try:
            with open(favorites_file, 'r') as f:
                saved_favorites = json.load(f)
                if isinstance(saved_favorites, list):
                    st.session_state.favorites = saved_favorites
        except:
            pass

def save_favorites():
    """Zapisz ulubione do pliku"""
    favorites_file = Path('favorites.json')
    try:
        with open(favorites_file, 'w') as f:
            json.dump(st.session_state.favorites, f, indent=2)
    except:
        pass

def toggle_favorite(symbol):
    """Przełącz ulubione dla symbolu"""
    if symbol in st.session_state.favorites:
        st.session_state.favorites.remove(symbol)
    else:
        st.session_state.favorites.append(symbol)
    save_favorites()

def init_custom_tickers():
    """Inicjalizuj listę własnych tickerów w session_state"""
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = []
    
    # Załaduj z pliku jeśli istnieje
    custom_tickers_file = Path('custom_tickers.json')
    if custom_tickers_file.exists():
        try:
            with open(custom_tickers_file, 'r') as f:
                saved_tickers = json.load(f)
                if isinstance(saved_tickers, list):
                    st.session_state.custom_tickers = saved_tickers
        except:
            pass

def save_custom_tickers():
    """Zapisz własne tickery do pliku"""
    custom_tickers_file = Path('custom_tickers.json')
    try:
        with open(custom_tickers_file, 'w') as f:
            json.dump(st.session_state.custom_tickers, f, indent=2)
    except:
        pass

def add_custom_ticker(symbol):
    """Dodaj własny ticker"""
    symbol = symbol.strip().upper()
    if symbol and symbol not in st.session_state.custom_tickers:
        st.session_state.custom_tickers.append(symbol)
        save_custom_tickers()
        return True
    return False

def remove_custom_ticker(symbol):
    """Usuń własny ticker"""
    if symbol in st.session_state.custom_tickers:
        st.session_state.custom_tickers.remove(symbol)
        save_custom_tickers()
        # Usuń też nazwę spółki z cache
        if 'stock_names' in st.session_state and symbol in st.session_state.stock_names:
            del st.session_state.stock_names[symbol]
            save_stock_names()
        return True
    return False

def get_stock_name(symbol):
    """Pobierz nazwę spółki dla symbolu używając yfinance"""
    # Załaduj cache nazw spółek
    if 'stock_names' not in st.session_state:
        init_stock_names()
    
    # Sprawdź czy mamy już nazwę w cache
    if symbol in st.session_state.stock_names:
        return st.session_state.stock_names[symbol]
    
    # Spróbuj pobrać nazwę z yfinance
    try:
        # Konwertuj symbol dla yfinance (np. .DE -> .F)
        variants = []
        if symbol.endswith('.DE'):
            variants = [symbol, symbol.replace('.DE', '.F')]
        elif symbol.endswith('.US'):
            variants = [symbol, symbol.replace('.US', '')]
        else:
            variants = [symbol]
        
        # Dodaj też warianty dla specjalnych symboli
        if symbol == 'NATGAS':
            variants.append('NG=F')
        
        name = None
        for variant in variants:
            try:
                ticker = yf.Ticker(variant)
                info = ticker.info
                if info:
                    # Spróbuj różne pola z nazwą
                    name = info.get('longName') or info.get('shortName') or info.get('name') or info.get('symbol')
                    if name and name != variant:
                        break
            except:
                continue
        
        # Jeśli nie znaleziono, użyj symbolu
        if not name:
            name = symbol
        
        # Zapisz do cache
        st.session_state.stock_names[symbol] = name
        save_stock_names()
        
        return name
    except Exception as e:
        # W przypadku błędu, zwróć symbol
        st.session_state.stock_names[symbol] = symbol
        save_stock_names()
        return symbol

def init_stock_names():
    """Inicjalizuj cache nazw spółek"""
    if 'stock_names' not in st.session_state:
        st.session_state.stock_names = {}
    
    # Załaduj z pliku jeśli istnieje
    names_file = Path('stock_names.json')
    if names_file.exists():
        try:
            with open(names_file, 'r', encoding='utf-8') as f:
                saved_names = json.load(f)
                if isinstance(saved_names, dict):
                    st.session_state.stock_names = saved_names
        except:
            pass

def save_stock_names():
    """Zapisz cache nazw spółek do pliku"""
    names_file = Path('stock_names.json')
    try:
        with open(names_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.stock_names, f, indent=2, ensure_ascii=False)
    except:
        pass

def get_stock_logo(symbol, debug=False, force_refresh=False):
    """Pobierz URL logo spółki dla symbolu używając yfinance
    
    Parameters:
    symbol: Symbol tickera
    debug: Czy wyświetlać debug info
    force_refresh: Czy wymusić ponowne pobranie (ignorować cache)
    """
    # Załaduj cache logo
    if 'stock_logos' not in st.session_state:
        init_stock_logos()
    
    # Sprawdź czy mamy już logo w cache (chyba że force_refresh)
    if not force_refresh and symbol in st.session_state.stock_logos:
        logo_url = st.session_state.stock_logos[symbol]
        if debug:
            st.write(f"DEBUG {symbol}: Logo z cache: {logo_url}")
        # Jeśli w cache jest None lub 'None', spróbuj ponownie (może yfinance teraz zwraca logo)
        if logo_url and logo_url != 'None':
            return logo_url
        elif debug:
            st.write(f"DEBUG {symbol}: Cache ma None, próbuję ponownie pobrać...")
    
    # Spróbuj pobrać logo z yfinance
    if debug:
        st.write(f"DEBUG {symbol}: Próbuję pobrać logo z yfinance...")
    
    try:
        # Konwertuj symbol dla yfinance (np. .DE -> .F)
        variants = []
        if symbol.endswith('.DE'):
            variants = [symbol, symbol.replace('.DE', '.F')]
        elif symbol.endswith('.US'):
            variants = [symbol, symbol.replace('.US', '')]
        else:
            variants = [symbol]
        
        # Dodaj też warianty dla specjalnych symboli
        if symbol == 'NATGAS':
            variants.append('NG=F')
        
        if debug:
            st.write(f"DEBUG {symbol}: Warianty do sprawdzenia: {variants}")
        
        logo_url = None
        for variant in variants:
            try:
                if debug:
                    st.write(f"DEBUG {symbol}: Sprawdzam wariant: {variant}")
                ticker = yf.Ticker(variant)
                info = ticker.info
                if info:
                    if debug:
                        # Pokaż wszystkie klucze związane z logo
                        logo_keys = [k for k in info.keys() if 'logo' in k.lower() or 'image' in k.lower()]
                        st.write(f"DEBUG {symbol}: Klucze związane z logo: {logo_keys}")
                        if logo_keys:
                            for key in logo_keys:
                                st.write(f"DEBUG {symbol}: {key} = {info.get(key)}")
                    
                    # Spróbuj różne pola z logo
                    logo_url = (info.get('logo_url') or 
                               info.get('logo') or 
                               info.get('image') or
                               info.get('companyLogo') or
                               info.get('companyLogoUrl'))
                    
                    if debug:
                        st.write(f"DEBUG {symbol}: Info dla {variant}:")
                        st.write(f"  - logo_url: {info.get('logo_url')}")
                        st.write(f"  - logo: {info.get('logo')}")
                        st.write(f"  - image: {info.get('image')}")
                        st.write(f"  - companyLogo: {info.get('companyLogo')}")
                        st.write(f"  - companyLogoUrl: {info.get('companyLogoUrl')}")
                        st.write(f"  - Wybrany logo_url: {logo_url}")
                    
                    if logo_url:
                        if debug:
                            st.write(f"DEBUG {symbol}: Znaleziono logo: {logo_url}")
                        break
                    elif debug:
                        # Pokaż przykładowe klucze z info, aby zobaczyć co jest dostępne
                        sample_keys = list(info.keys())[:20]
                        st.write(f"DEBUG {symbol}: Przykładowe klucze z info: {sample_keys}")
            except Exception as e:
                if debug:
                    st.write(f"DEBUG {symbol}: Błąd dla {variant}: {str(e)}")
                    import traceback
                    st.write(f"DEBUG {symbol}: Traceback:")
                    st.code(traceback.format_exc())
                continue
        
        # Zapisz do cache (nawet jeśli None)
        st.session_state.stock_logos[symbol] = logo_url if logo_url else 'None'
        save_stock_logos()
        
        if debug:
            st.write(f"DEBUG {symbol}: Finalny logo_url: {logo_url}")
        
        return logo_url if logo_url else None
    except Exception as e:
        # W przypadku błędu, zapisz None
        if debug:
            st.write(f"DEBUG {symbol}: Wyjątek: {str(e)}")
        st.session_state.stock_logos[symbol] = 'None'
        save_stock_logos()
        return None

def init_stock_logos():
    """Inicjalizuj cache logo spółek"""
    if 'stock_logos' not in st.session_state:
        st.session_state.stock_logos = {}
    
    # Załaduj z pliku jeśli istnieje
    logos_file = Path('stock_logos.json')
    if logos_file.exists():
        try:
            with open(logos_file, 'r', encoding='utf-8') as f:
                saved_logos = json.load(f)
                if isinstance(saved_logos, dict):
                    st.session_state.stock_logos = saved_logos
        except:
            pass

def save_stock_logos():
    """Zapisz cache logo spółek do pliku"""
    logos_file = Path('stock_logos.json')
    try:
        with open(logos_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.stock_logos, f, indent=2, ensure_ascii=False)
    except:
        pass

def plot_stock_price(predictor, days=60):
    """Wykres ceny akcji z przewidywaniami kierunku"""
    if predictor.features is None or predictor.model is None:
        return None
    
    # Sprawdź jakie cechy model oczekuje (kompatybilność wsteczna)
    model_expected_features = None
    if hasattr(predictor.model, 'feature_names_in_'):
        model_expected_features = list(predictor.model.feature_names_in_)
    elif hasattr(predictor, 'selected_features') and predictor.selected_features:
        model_expected_features = predictor.selected_features
    
    # Przygotuj features zgodnie z oczekiwaniami modelu
    if model_expected_features:
        # Model ma określone cechy - użyj tylko tych
        available_features = [f for f in model_expected_features if f in predictor.features.columns]
        
        if len(available_features) != len(model_expected_features):
            # Niektóre cechy brakują - spróbuj użyć dostępnych
            if hasattr(predictor.model, 'feature_names_in_'):
                # Model sklearn wymaga dokładnego dopasowania
                recent_features_df = pd.DataFrame(index=predictor.features.index)
                for feat in model_expected_features:
                    if feat in predictor.features.columns:
                        recent_features_df[feat] = predictor.features[feat]
                    else:
                        recent_features_df[feat] = 0
                recent_features_df = recent_features_df[model_expected_features]
                recent_features = recent_features_df.iloc[-days:]
            else:
                recent_features = predictor.features[available_features].iloc[-days:]
        else:
            recent_features = predictor.features[model_expected_features].iloc[-days:]
    else:
        # Model nie ma określonych cech - użyj wszystkich dostępnych
        recent_features = predictor.features.iloc[-days:]
    
    recent_actual_direction = predictor.target.iloc[-days:]  # 0 = DOWN, 1 = UP
    recent_actual_prices = predictor.data['Close'].iloc[-days:]
    recent_predictions = predictor.model.predict(recent_features)
    recent_proba = predictor.model.predict_proba(recent_features)[:, 1]
    recent_dates = predictor.data.index[-days:]
    
    fig = go.Figure()
    
    # Cena rzeczywista
    fig.add_trace(go.Scatter(
        x=recent_dates,
        y=recent_actual_prices.values,
        mode='lines',
        name='Cena rzeczywista',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Oznacz przewidywania kierunku
    correct_mask = recent_actual_direction.values == recent_predictions
    for i, (date, price, correct, direction) in enumerate(zip(recent_dates, recent_actual_prices.values, correct_mask, recent_predictions)):
        color = 'green' if correct else 'red'
        marker = 'triangle-up' if direction == 1 else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[date],
            y=[price],
            mode='markers',
            marker=dict(symbol=marker, size=10, color=color, opacity=0.7),
            name='UP' if direction == 1 else 'DOWN',
            showlegend=False,
            hovertemplate=f"Data: {date}<br>Cena: ${price:.2f}<br>Kierunek: {'UP' if direction == 1 else 'DOWN'}<br>Poprawne: {'Tak' if correct else 'Nie'}<extra></extra>"
        ))
    
    # Przewidywanie na jutro
    last_date = recent_dates[-1]
    next_date = last_date + timedelta(days=1)
    current_price = predictor.data['Close'].iloc[-1]
    
    # Użyj odpowiednich cech dla przewidywania
    if model_expected_features:
        available_features = [f for f in model_expected_features if f in predictor.features.columns]
        if len(available_features) != len(model_expected_features) and hasattr(predictor.model, 'feature_names_in_'):
            last_features_df = pd.DataFrame(index=predictor.features.index[-1:])
            for feat in model_expected_features:
                if feat in predictor.features.columns:
                    last_features_df[feat] = predictor.features[feat].iloc[-1:]
                else:
                    last_features_df[feat] = 0
            last_features_df = last_features_df[model_expected_features]
            last_features = last_features_df.values
        else:
            last_features = predictor.features[model_expected_features].iloc[-1:].values
    else:
        last_features = predictor.features.iloc[-1:].values
    
    next_direction = predictor.model.predict(last_features)[0]
    next_proba = predictor.model.predict_proba(last_features)[0, 1]
    
    # Oszacuj cenę na podstawie średniej zmiany historycznej
    historical_changes = predictor.data['Close'].pct_change().dropna()
    if next_direction == 1:
        avg_change = historical_changes[historical_changes > 0].mean() if len(historical_changes[historical_changes > 0]) > 0 else 0.01
        estimated_price = current_price * (1 + avg_change * next_proba)
    else:
        avg_change = historical_changes[historical_changes < 0].mean() if len(historical_changes[historical_changes < 0]) > 0 else -0.01
        estimated_price = current_price * (1 + avg_change * (1 - next_proba))
    
    fig.add_trace(go.Scatter(
        x=[last_date, next_date],
        y=[current_price, estimated_price],
        mode='lines+markers',
        name=f'Przewidywanie (jutro): {"UP" if next_direction == 1 else "DOWN"} ({next_proba*100:.1f}%)',
        line=dict(color='#2ca02c' if next_direction == 1 else '#d62728', width=3),
        marker=dict(size=12, symbol='triangle-up' if next_direction == 1 else 'triangle-down')
    ))
    
    fig.update_layout(
        title=f'Przewidywania kierunku akcji {predictor.symbol}',
        xaxis_title='Data',
        yaxis_title='Cena ($)',
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_technical_indicators(predictor, days=60):
    """Wykres wskaźników technicznych"""
    if predictor.data is None:
        return None
    
    df = predictor.data.iloc[-days:].copy()
    
    # Oblicz wskaźniki
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    
    # Cena i średnie kroczące
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Cena', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='MA 20', line=dict(color='orange', dash='dash')))
    if len(df) > 50:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name='MA 50', line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title='Wskaźniki techniczne',
        xaxis_title='Data',
        yaxis_title='Cena ($)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_rsi(predictor, days=60):
    """Wykres RSI"""
    if predictor.data is None:
        return None
    
    df = predictor.data.iloc[-days:].copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Przewykupienie (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Przeprodanie (30)")
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        xaxis_title='Data',
        yaxis_title='RSI',
        yaxis_range=[0, 100],
        height=300,
        hovermode='x unified'
    )
    
    return fig

@st.cache_data(ttl=300)
def get_intraday_data(symbol, days=7, interval=None):
    """Pobierz dane intraday dla ostatnich dni (automatycznie dobiera interwał).
    
    Yahoo Finance udostępnia:
    - 1m: maks. 7 dni
    - 5m: maks. 60 dni
    """
    try:
        if interval is None:
            interval = '1m' if days <= 7 else '5m'
        
        max_days = 7 if interval == '1m' else 60
        period_days = min(days, max_days)
        
        variants = [symbol]
        if '.DE' in symbol:
            base = symbol.replace('.DE', '')
            variants.extend([f"{base}.F", base])
        if '.US' in symbol:
            base = symbol.replace('.US', '')
            variants.extend([base])
        
        intraday_data = None
        for variant in variants:
            try:
                ticker = yf.Ticker(variant)
                data = ticker.history(period=f"{period_days}d", interval=interval)
                if not data.empty:
                    intraday_data = data
                    break
            except:
                continue
        
        if intraday_data is None or intraday_data.empty:
            return None
        
        # Normalizuj timezone
        if intraday_data.index.tz is not None:
            intraday_data.index = intraday_data.index.tz_localize(None)
        
        return intraday_data
    except Exception as e:
        return None

def simulate_future_week(predictor, days=7):
    """
    Symuluje przyszły tydzień na podstawie modelu ML i historycznych wzorców
    
    Args:
        predictor: StockPredictor z wytrenowanym modelem
        days: Liczba dni do symulacji (domyślnie 7)
    
    Returns:
        DataFrame z symulowanymi danymi intraday (co minutę) podobny do get_intraday_data
    """
    if predictor.model is None or predictor.data is None or len(predictor.data) == 0:
        return None
    
    try:
        # Pobierz historyczne wzorce intraday (jeśli dostępne)
        historical_intraday = get_intraday_data(predictor.symbol, days=7)
        historical_patterns = None
        historical_template = None  # Szablon z rzeczywistych danych
        
        if historical_intraday is not None and len(historical_intraday) > 0:
            historical_patterns = analyze_intraday_patterns(historical_intraday)
            
            # Użyj ostatniego dnia jako szablonu (jeśli dostępny)
            historical_filtered = filter_trading_hours(historical_intraday)
            if historical_filtered is not None and len(historical_filtered) > 0:
                # Weź ostatni dzień jako szablon
                last_day = historical_filtered.index[-1].date()
                last_day_data = historical_filtered[historical_filtered.index.date == last_day]
                if len(last_day_data) > 10:  # Wystarczająco danych
                    historical_template = last_day_data.copy()
        
        # Oblicz średnie zmiany historyczne
        historical_changes = predictor.data['Close'].pct_change().dropna()
        avg_up_change = historical_changes[historical_changes > 0].mean() if len(historical_changes[historical_changes > 0]) > 0 else 0.01
        avg_down_change = historical_changes[historical_changes < 0].mean() if len(historical_changes[historical_changes < 0]) > 0 else -0.01
        
        # Oblicz średnią zmienność (volatility)
        volatility = historical_changes.std()
        
        # Obecna cena
        current_price = predictor.data['Close'].iloc[-1]
        
        # Generuj daty dla przyszłego tygodnia (tylko dni robocze)
        from datetime import datetime, timedelta
        simulated_data = []
        current_date = datetime.now().date()
        
        # Generuj minuty dla każdego dnia (tylko godziny handlu: 9:00-17:30)
        trading_hours = list(range(9, 18))  # 9:00-17:59
        trading_minutes = [0, 15, 30, 45]  # Co 15 minut dla szybszej symulacji (można zmienić na co minutę)
        
        day_count = 0
        while day_count < days:
            # Sprawdź czy to dzień roboczy (poniedziałek=0, niedziela=6)
            if current_date.weekday() < 5:  # Tylko pon-pt
                # Przewiduj kierunek na ten dzień używając modelu
                try:
                    # Użyj ostatnich cech do przewidywania - z pełnym dopasowaniem cech
                    # Sprawdź jakie cechy model oczekuje (kompatybilność wsteczna)
                    model_expected_features = None
                    if hasattr(predictor.model, 'feature_names_in_'):
                        model_expected_features = list(predictor.model.feature_names_in_)
                    elif hasattr(predictor, 'selected_features') and predictor.selected_features:
                        model_expected_features = predictor.selected_features
                    
                    # Przygotuj features zgodnie z oczekiwaniami modelu (jak w plot_stock_price)
                    if model_expected_features:
                        # Model ma określone cechy - użyj tylko tych
                        available_features = [f for f in model_expected_features if f in predictor.features.columns]
                        
                        if len(available_features) != len(model_expected_features):
                            # Niektóre cechy brakują - użyj mapowania i wypełnij brakujące
                            last_features_df = pd.DataFrame(index=predictor.features.index[-1:])
                            feature_mapping = {
                                'ATR': 'ATR_log',
                                'TR': 'TR_log',
                                'Volume': 'Volume_log'
                            }
                            
                            for feat in model_expected_features:
                                if feat in predictor.features.columns:
                                    last_features_df[feat] = predictor.features[feat].iloc[-1:]
                                elif feat in feature_mapping:
                                    mapped_feat = feature_mapping[feat]
                                    if mapped_feat in predictor.features.columns:
                                        last_features_df[feat] = predictor.features[mapped_feat].iloc[-1:]
                                    else:
                                        last_features_df[feat] = 0
                                else:
                                    last_features_df[feat] = 0
                            
                            last_features_df = last_features_df[model_expected_features]
                            last_features = last_features_df.values
                        else:
                            last_features = predictor.features[model_expected_features].iloc[-1:].values
                    else:
                        # Model nie ma określonych cech - użyj wszystkich dostępnych
                        last_features = predictor.features.iloc[-1:].values
                    
                    # Przewiduj kierunek (użyj prawdopodobieństwa zamiast tylko 0/1)
                    direction_proba = predictor.model.predict_proba(last_features)[0]
                    up_probability = direction_proba[1] if len(direction_proba) > 1 else 0.5
                    
                    # Losuj kierunek na podstawie prawdopodobieństwa
                    daily_direction = 1 if np.random.random() < up_probability else 0
                    
                    # Oblicz oczekiwaną zmianę dzienną
                    if daily_direction == 1:
                        expected_daily_change = avg_up_change * up_probability
                    else:
                        expected_daily_change = avg_down_change * (1 - up_probability)
                    
                    # Ogranicz zmienność dzienną do realistycznych wartości (max 5% dziennie)
                    max_daily_change = 0.05
                    daily_change = expected_daily_change + np.random.normal(0, min(volatility * 0.3, 0.02))
                    daily_change = np.clip(daily_change, -max_daily_change, max_daily_change)
                    
                    # Cena otwarcia (może być mały gap, max 1%)
                    gap_probability = 0.2  # 20% szansy na gap
                    max_gap = 0.01  # Max 1% gap
                    if np.random.random() < gap_probability:
                        gap_size = np.random.normal(0, abs(expected_daily_change) * 0.3)
                        gap_size = np.clip(gap_size, -max_gap, max_gap)
                        day_open = current_price * (1 + gap_size)
                    else:
                        day_open = current_price
                    
                    # Symuluj ruch w ciągu dnia (użyj historycznych wzorców jeśli dostępne)
                    day_close = day_open * (1 + daily_change)
                    
                    # Ogranicz zakres dzienny (High-Low) do max 3% od otwarcia
                    max_intraday_range = 0.03
                    day_range = abs(day_close - day_open)
                    if day_range / day_open > max_intraday_range:
                        # Skaluj zmianę aby zmieścić się w zakresie
                        scale = (day_open * max_intraday_range) / day_range
                        day_close = day_open + (day_close - day_open) * scale
                    
                    day_low = min(day_open, day_close) * (1 - max_intraday_range * 0.5)
                    day_high = max(day_open, day_close) * (1 + max_intraday_range * 0.5)
                    
                    # Generuj dane co 15 minut (lub co minutę jeśli potrzeba)
                    # Jeśli mamy szablon z rzeczywistych danych, użyj go
                    day_data_points = []
                    
                    if historical_template is not None and len(historical_template) > 0:
                        # Użyj szablonu z rzeczywistych danych - bardziej realistyczne
                        template_prices = historical_template['Close'].values
                        template_high = historical_template['High'].values
                        template_low = historical_template['Low'].values
                        
                        # Skaluj szablon do przewidywanego zakresu dnia
                        template_start_price = template_prices[0]
                        template_end_price = template_prices[-1]
                        template_range = template_end_price - template_start_price
                        
                        # Oblicz skalę dla nowego dnia
                        if abs(template_range) > 0.0001:  # Unikaj dzielenia przez zero
                            scale = (day_close - day_open) / template_range
                        else:
                            scale = 1.0
                        
                        # Ogranicz skalę do realistycznych wartości
                        scale = np.clip(scale, 0.5, 2.0)
                        
                        # Generuj dane na podstawie szablonu
                        template_idx = 0
                        for hour in trading_hours:
                            for minute in trading_minutes:
                                timestamp = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute))
                                
                                if template_idx < len(template_prices):
                                    # Użyj ceny z szablonu, przeskalowanej
                                    template_price = template_prices[template_idx]
                                    template_rel_change = (template_price - template_start_price) / template_start_price if template_start_price > 0 else 0
                                    
                                    # Zastosuj skalę do zmiany względnej
                                    price = day_open * (1 + template_rel_change * scale)
                                    
                                    # Dodaj mały szum (max 0.1%)
                                    price_noise = np.random.normal(0, price * 0.001)
                                    price_noise = np.clip(price_noise, -price * 0.002, price * 0.002)
                                    price = price + price_noise
                                    
                                    # High i Low z szablonu
                                    template_high_val = template_high[template_idx]
                                    template_low_val = template_low[template_idx]
                                    template_high_rel = (template_high_val - template_start_price) / template_start_price if template_start_price > 0 else 0
                                    template_low_rel = (template_low_val - template_start_price) / template_start_price if template_start_price > 0 else 0
                                    
                                    price_high = day_open * (1 + template_high_rel * scale) + abs(price_noise) * 0.5
                                    price_low = day_open * (1 + template_low_rel * scale) - abs(price_noise) * 0.5
                                    
                                    template_idx += 1
                                else:
                                    # Jeśli szablon się skończył, użyj interpolacji
                                    day_progress = ((hour - 9) * 60 + minute) / (8 * 60)
                                    price = day_open + (day_close - day_open) * day_progress
                                    price_spread = price * 0.001
                                    price_high = price + price_spread * 0.5
                                    price_low = price - price_spread * 0.5
                                
                                # Upewnij się, że ceny są w realistycznym zakresie
                                price = np.clip(price, day_low, day_high)
                                price_high = max(price, price_high)
                                price_low = min(price, price_low)
                                
                                day_data_points.append({
                                    'timestamp': timestamp,
                                    'Open': price,
                                    'High': price_high,
                                    'Low': price_low,
                                    'Close': price,
                                    'Volume': np.random.lognormal(10, 0.5)
                                })
                    else:
                        # Standardowa symulacja (bez szablonu)
                        for hour in trading_hours:
                            for minute in trading_minutes:
                                timestamp = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute))
                                
                                # Oblicz postęp dnia (0.0 = otwarcie, 1.0 = zamknięcie)
                                day_progress = ((hour - 9) * 60 + minute) / (8 * 60)  # 8 godzin handlu
                                day_progress = min(1.0, max(0.0, day_progress))
                                
                                # Symuluj cenę w ciągu dnia (użyj bardziej realistycznego wzorca)
                                # Wzorzec: często dołek rano, szczyt po południu, ale z mniejszymi wahaniami
                                if historical_patterns and historical_patterns['daily_patterns']:
                                    # Użyj średniego czasu dołku/szczytu z historii
                                    try:
                                        avg_low_hour = int(historical_patterns['avg_low_time'].split(':')[0])
                                        avg_high_hour = int(historical_patterns['avg_high_time'].split(':')[0])
                                        
                                        # Oblicz odległość od typowego dołku/szczytu (bardziej subtelny efekt)
                                        low_distance = abs(hour - avg_low_hour) / 8.0
                                        high_distance = abs(hour - avg_high_hour) / 8.0
                                        
                                        # Cena bliżej dołku rano, szczytu po południu (mniejszy efekt - max 1%)
                                        if hour < 12:
                                            price_factor = 1.0 - low_distance * 0.01  # Max 1% różnica
                                        else:
                                            price_factor = 1.0 + high_distance * 0.01  # Max 1% różnica
                                    except:
                                        price_factor = 1.0
                                else:
                                    # Domyślny wzorzec: dołek rano, szczyt po południu (bardzo subtelny)
                                    if hour < 12:
                                        price_factor = 0.995 + (hour - 9) / 3 * 0.005  # Max 0.5% różnica
                                    else:
                                        price_factor = 1.0 + (hour - 12) / 5 * 0.005  # Max 0.5% różnica
                                
                                # Interpoluj cenę między otwarciem a zamknięciem z uwzględnieniem wzorca
                                base_price = day_open + (day_close - day_open) * day_progress
                                
                                # Dodaj bardzo małe wahania w ciągu dnia (symuluj High/Low)
                                # Użyj znacznie mniejszej zmienności - max 0.1% na minutę
                                intraday_volatility = min(volatility * 0.1, 0.001)  # Max 0.1% zmienność
                                price_noise = np.random.normal(0, intraday_volatility)
                                price_noise = np.clip(price_noise, -0.002, 0.002)  # Max 0.2% szum
                                
                                # Oblicz cenę z uwzględnieniem wzorca i losowości
                                price = base_price * price_factor * (1 + price_noise)
                                
                                # Upewnij się, że cena jest w realistycznym zakresie
                                price = np.clip(price, day_low, day_high)
                                
                                # Aktualizuj Low i High
                                day_low = min(day_low, price)
                                day_high = max(day_high, price)
                                
                                # Symuluj Open, High, Low, Close dla każdej minuty (bardzo małe różnice)
                                # High i Low powinny być bardzo blisko Close (max 0.1% różnica)
                                price_spread = price * 0.001  # Max 0.1% spread
                                price_high = price + abs(np.random.normal(0, price_spread * 0.5))
                                price_low = price - abs(np.random.normal(0, price_spread * 0.5))
                                
                                # Upewnij się, że High >= Close >= Low
                                price_high = max(price, price_high)
                                price_low = min(price, price_low)
                                
                                day_data_points.append({
                                    'timestamp': timestamp,
                                    'Open': price,
                                    'High': price_high,
                                    'Low': price_low,
                                    'Close': price,
                                    'Volume': np.random.lognormal(10, 0.5)  # Symuluj wolumen (mniejsza zmienność)
                                })
                    
                    # Upewnij się, że Low i High są poprawne dla całego dnia
                    # Ale nie zmieniaj zbyt drastycznie - tylko delikatnie dostosuj
                    for point in day_data_points:
                        # Delikatnie dostosuj, ale nie więcej niż 0.5%
                        if point['Low'] < day_low:
                            point['Low'] = min(point['Low'] * 1.005, day_low)
                        if point['High'] > day_high:
                            point['High'] = max(point['High'] * 0.995, day_high)
                    
                    # Zaktualizuj cenę zamknięcia ostatniego punktu dnia
                    if day_data_points:
                        day_data_points[-1]['Close'] = day_close
                    
                    simulated_data.extend(day_data_points)
                    
                    # Zaktualizuj cenę dla następnego dnia
                    current_price = day_close
                    day_count += 1
                    
                except Exception as e:
                    # Jeśli błąd, użyj prostszej symulacji (ale też z ograniczeniami)
                    daily_change = np.random.normal(0, min(volatility, 0.02))
                    daily_change = np.clip(daily_change, -0.05, 0.05)  # Max 5% dziennie
                    day_open = current_price
                    day_close = current_price * (1 + daily_change)
                    
                    day_data_points = []
                    for hour in trading_hours:
                        for minute in trading_minutes:
                            timestamp = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute))
                            day_progress = ((hour - 9) * 60 + minute) / (8 * 60)
                            price = day_open + (day_close - day_open) * day_progress
                            
                            # Bardzo małe wahania
                            price_spread = price * 0.001
                            day_data_points.append({
                                'timestamp': timestamp,
                                'Open': price,
                                'High': price + price_spread * 0.5,
                                'Low': price - price_spread * 0.5,
                                'Close': price,
                                'Volume': np.random.lognormal(10, 0.5)
                            })
                    
                    simulated_data.extend(day_data_points)
                    current_price = day_close
                    day_count += 1
            
            # Przejdź do następnego dnia
            current_date += timedelta(days=1)
        
        if not simulated_data:
            return None
        
        # Utwórz DataFrame z datami - timestamps są już w danych
        if simulated_data and 'timestamp' in simulated_data[0]:
            timestamps = [point['timestamp'] for point in simulated_data]
            data_dict = {key: [point[key] for point in simulated_data] for key in ['Open', 'High', 'Low', 'Close', 'Volume']}
            df_simulated = pd.DataFrame(data_dict, index=timestamps)
        else:
            # Fallback - użyj prostszego podejścia
            df_simulated = pd.DataFrame(simulated_data)
            df_simulated.index = pd.date_range(start=datetime.now(), periods=len(simulated_data), freq='15min')
        
        return df_simulated
        
    except Exception as e:
        print(f"Błąd przy symulacji przyszłego tygodnia: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def filter_trading_hours(intraday_data):
    """Filtruj dane, aby pokazywać tylko godziny handlu (pomiń zamknięcie giełdy)"""
    if intraday_data is None or len(intraday_data) == 0:
        return None
    
    intraday_filtered = intraday_data.copy()
    
    # Sprawdź rzeczywiste godziny w danych (giełda jest otwarta tylko w określonych godzinach)
    hours = intraday_filtered.index.hour
    minutes = intraday_filtered.index.minute
    
    # Znajdź zakres godzin z danymi (prawdopodobnie godziny handlu)
    # Giełda amerykańska: 9:30-16:00 (14:30-21:00 UTC)
    # Giełda niemiecka: 9:00-17:30 (8:00-16:30 UTC)
    # Użyjemy bardziej uniwersalnego podejścia - znajdź główny zakres godzin
    
    unique_hours = sorted(hours.unique())
    if len(unique_hours) > 0:
        # Znajdź najczęstsze godziny (prawdopodobnie godziny handlu)
        hour_counts = hours.value_counts()
        main_hours = hour_counts[hour_counts > len(intraday_filtered) * 0.1].index.tolist()  # Godziny z >10% danych
        
        if main_hours:
            min_hour = max(8, min(main_hours))
            max_hour = min(18, max(main_hours))
            
            # Filtruj dane - tylko godziny handlu
            mask = (hours >= min_hour) & (hours <= max_hour)
            intraday_filtered = intraday_filtered[mask]
    
    return intraday_filtered

def analyze_intraday_patterns(intraday_data):
    """Analizuj wzorce intraday - znajdź najlepsze i najgorsze momenty w ciągu dnia"""
    if intraday_data is None or len(intraday_data) == 0:
        return None
    
    # Filtruj tylko godziny handlu
    intraday_data = filter_trading_hours(intraday_data)
    if intraday_data is None or len(intraday_data) == 0:
        return None
    
    # Grupuj po dniach i godzinach
    intraday_data = intraday_data.copy()
    intraday_data['Date'] = intraday_data.index.date
    intraday_data['Hour'] = intraday_data.index.hour
    intraday_data['Minute'] = intraday_data.index.minute
    intraday_data['Time'] = intraday_data.index.time
    
    # Oblicz zmiany cen w ciągu dnia
    daily_patterns = []
    
    for date in intraday_data['Date'].unique():
        day_data = intraday_data[intraday_data['Date'] == date]
        if len(day_data) < 10:  # Za mało danych dla tego dnia
            continue
        
        # Znajdź najniższą i najwyższą cenę w ciągu dnia
        day_low = day_data['Low'].min()
        day_high = day_data['High'].max()
        day_open = day_data['Open'].iloc[0]
        day_close = day_data['Close'].iloc[-1]
        
        # Znajdź momenty najniższej i najwyższej ceny
        low_time = day_data.loc[day_data['Low'].idxmin(), 'Time']
        high_time = day_data.loc[day_data['High'].idxmax(), 'Time']
        
        daily_patterns.append({
            'date': date,
            'low': day_low,
            'high': day_high,
            'open': day_open,
            'close': day_close,
            'low_time': low_time,
            'high_time': high_time,
            'change_pct': ((day_close - day_open) / day_open * 100) if day_open > 0 else 0
        })
    
    if not daily_patterns:
        return None
    
    # Analizuj wzorce czasowe
    low_times = [p['low_time'] for p in daily_patterns]
    high_times = [p['high_time'] for p in daily_patterns]
    
    # Znajdź najczęstsze godziny dla dołków i szczytów
    low_hours = [t.hour for t in low_times]
    high_hours = [t.hour for t in high_times]
    
    # Oblicz średnie godziny
    avg_low_hour = np.mean(low_hours) if low_hours else 0
    avg_high_hour = np.mean(high_hours) if high_hours else 0
    
    # Znajdź najczęstsze minuty
    low_minutes = [t.minute for t in low_times]
    high_minutes = [t.minute for t in high_times]
    
    avg_low_minute = int(np.mean(low_minutes)) if low_minutes else 0
    avg_high_minute = int(np.mean(high_minutes)) if high_minutes else 0
    
    return {
        'daily_patterns': daily_patterns,
        'avg_low_time': f"{int(avg_low_hour):02d}:{avg_low_minute:02d}",
        'avg_high_time': f"{int(avg_high_hour):02d}:{avg_high_minute:02d}",
        'low_hours_dist': low_hours,
        'high_hours_dist': high_hours
    }

def find_trading_edges(intraday_data, min_profit_pct=0.1, lookback_window=30):
    """
    Edge Finder - znajdź optymalne momenty do zakupu i sprzedaży
    
    Args:
        intraday_data: DataFrame z danymi intraday
        min_profit_pct: Minimalny zysk w % aby uznać transakcję za opłacalną
        lookback_window: Okno w minutach do szukania lokalnych minimów/maksimów
    
    Returns:
        Lista słowników z parami transakcji: {'buy_time', 'buy_price', 'sell_time', 'sell_price', 'profit_pct', 'profit'}
    """
    if intraday_data is None or len(intraday_data) < lookback_window * 2:
        return []
    
    # Filtruj tylko godziny handlu
    intraday_filtered = filter_trading_hours(intraday_data)
    if intraday_filtered is None or len(intraday_filtered) == 0:
        return []
    
    data = intraday_filtered.copy()
    data['Low'] = data['Low'].fillna(method='ffill')
    data['High'] = data['High'].fillna(method='ffill')
    data['Close'] = data['Close'].fillna(method='ffill')
    
    # Znajdź lokalne minima (miejsca do zakupu)
    local_minima = []
    local_maxima = []
    
    for i in range(lookback_window, len(data) - lookback_window):
        # Sprawdź czy to lokalne minimum (najniższa cena w oknie)
        window_low = data['Low'].iloc[i-lookback_window:i+lookback_window+1]
        if data['Low'].iloc[i] == window_low.min() and data['Low'].iloc[i] == window_low.iloc[lookback_window]:
            local_minima.append({
                'index': i,
                'time': data.index[i],
                'price': data['Low'].iloc[i]
            })
        
        # Sprawdź czy to lokalne maximum (najwyższa cena w oknie)
        window_high = data['High'].iloc[i-lookback_window:i+lookback_window+1]
        if data['High'].iloc[i] == window_high.max() and data['High'].iloc[i] == window_high.iloc[lookback_window]:
            local_maxima.append({
                'index': i,
                'time': data.index[i],
                'price': data['High'].iloc[i]
            })
    
    # Znajdź pary: kupno (minimum) -> sprzedaż (maksimum)
    trading_edges = []
    
    for buy in local_minima:
        # Znajdź najbliższe maksimum po tym minimum
        potential_sells = [sell for sell in local_maxima if sell['index'] > buy['index']]
        
        if potential_sells:
            # Weź pierwsze maksimum które daje zysk powyżej minimum
            for sell in potential_sells:
                profit_pct = ((sell['price'] - buy['price']) / buy['price']) * 100
                
                if profit_pct >= min_profit_pct:
                    trading_edges.append({
                        'buy_time': buy['time'],
                        'buy_price': buy['price'],
                        'buy_index': buy['index'],
                        'sell_time': sell['time'],
                        'sell_price': sell['price'],
                        'sell_index': sell['index'],
                        'profit': sell['price'] - buy['price'],
                        'profit_pct': profit_pct,
                        'duration_minutes': (sell['time'] - buy['time']).total_seconds() / 60
                    })
                    break  # Weź tylko pierwszą dobrą okazję po każdym minimum
    
    # Sortuj według zysku (najlepsze na górze)
    trading_edges.sort(key=lambda x: x['profit_pct'], reverse=True)
    
    return trading_edges

def plot_candlestick_intraday(intraday_data, patterns=None, show_edges=True):
    """Wykres świecowy z danymi intraday (co minutę) - tylko godziny handlu, ciągły wykres"""
    if intraday_data is None or len(intraday_data) == 0:
        return None
    
    # Filtruj tylko godziny handlu (użyj funkcji pomocniczej)
    intraday_filtered = filter_trading_hours(intraday_data)
    
    if intraday_filtered is None or len(intraday_filtered) == 0:
        return None
    
    # Ogranicz do ostatnich 2 dni dla czytelności
    last_2_days = intraday_filtered.tail(1000)  # Ostatnie ~1000 minut
    
    # Utwórz ciągłą numerację dla osi X (bez przerw między dniami)
    # Użyjemy indeksu numerycznego zamiast daty, aby uniknąć przerw
    # KONWERTUJ range() na listę - Plotly wymaga listy/tablicy, nie obiektu range
    x_continuous = list(range(len(last_2_days)))
    x_labels = [ts.strftime('%Y-%m-%d %H:%M') for ts in last_2_days.index]
    
    # Znajdź optymalne pary transakcji (Edge Finder)
    trading_edges = []
    if show_edges:
        trading_edges = find_trading_edges(last_2_days, min_profit_pct=0.05, lookback_window=20)
    
    fig = go.Figure()
    
    # Wykres świecowy z ciągłą osią X
    fig.add_trace(go.Candlestick(
        x=x_continuous,
        open=last_2_days['Open'].values,
        high=last_2_days['High'].values,
        low=last_2_days['Low'].values,
        close=last_2_days['Close'].values,
        name='Cena',
        customdata=x_labels,
        hovertemplate='<b>%{customdata}</b><br>' +
                      'Otwarcie: $%{open:.2f}<br>' +
                      'Najwyższa: $%{high:.2f}<br>' +
                      'Najniższa: $%{low:.2f}<br>' +
                      'Zamknięcie: $%{close:.2f}<extra></extra>'
    ))
    
    # Edge Finder - zaznacz optymalne pary transakcji
    if show_edges and trading_edges:
        # Pokaż tylko najlepsze 5 okazji
        best_edges = trading_edges[:5]
        
        for idx, edge in enumerate(best_edges):
            # Znajdź indeksy w ciągłej numeracji
            try:
                buy_idx = list(last_2_days.index).index(edge['buy_time'])
                sell_idx = list(last_2_days.index).index(edge['sell_time'])
                
                # Strzałka od zakupu do sprzedaży
                fig.add_trace(go.Scatter(
                    x=[buy_idx, sell_idx],
                    y=[edge['buy_price'], edge['sell_price']],
                    mode='lines+markers',
                    name=f'💰 Okazja #{idx+1}',
                    line=dict(
                        color='#00FF00' if edge['profit_pct'] > 0.5 else '#90EE90',
                        width=3,
                        dash='dash'
                    ),
                    marker=dict(size=12),
                    showlegend=True,
                    hovertemplate=f"<b>Okazja #{idx+1}</b><br>" +
                                f"Kup: {edge['buy_time'].strftime('%Y-%m-%d %H:%M')} @ ${edge['buy_price']:.2f}<br>" +
                                f"Sprzedaj: {edge['sell_time'].strftime('%Y-%m-%d %H:%M')} @ ${edge['sell_price']:.2f}<br>" +
                                f"Zysk: ${edge['profit']:.2f} ({edge['profit_pct']:.2f}%)<br>" +
                                f"Czas: {edge['duration_minutes']:.0f} min<extra></extra>"
                ))
                
                # Zaznacz punkt zakupu (zielony diament)
                fig.add_trace(go.Scatter(
                    x=[buy_idx],
                    y=[edge['buy_price']],
                    mode='markers+text',
                    name=f'🛒 Kup #{idx+1}',
                    marker=dict(size=18, color='green', symbol='diamond'),
                    text=[f"${edge['buy_price']:.2f}"],
                    textposition='bottom center',
                    showlegend=False,
                    hovertemplate=f"<b>KUP - {edge['buy_time'].strftime('%Y-%m-%d %H:%M')}</b><br>Cena: ${edge['buy_price']:.2f}<extra></extra>"
                ))
                
                # Zaznacz punkt sprzedaży (czerwona strzałka w górę)
                fig.add_trace(go.Scatter(
                    x=[sell_idx],
                    y=[edge['sell_price']],
                    mode='markers+text',
                    name=f'💵 Sprzedaj #{idx+1}',
                    marker=dict(size=18, color='red', symbol='triangle-up'),
                    text=[f"${edge['sell_price']:.2f}"],
                    textposition='top center',
                    showlegend=False,
                    hovertemplate=f"<b>SPRZEDAŻ - {edge['sell_time'].strftime('%Y-%m-%d %H:%M')}</b><br>Cena: ${edge['sell_price']:.2f}<br>Zysk: ${edge['profit']:.2f} ({edge['profit_pct']:.2f}%)<extra></extra>"
                ))
            except (ValueError, KeyError):
                continue  # Pomiń jeśli nie można znaleźć indeksu
    
    # Zaznacz najlepsze i najgorsze momenty jeśli mamy wzorce (tylko jeśli nie pokazujemy edges)
    if patterns and not show_edges:
        # Znajdź dołki i szczyty w ostatnich 2 dniach
        for pattern in patterns['daily_patterns'][-2:]:  # Ostatnie 2 dni
            date = pattern['date']
            day_data = last_2_days[last_2_days.index.date == date]
            
            if len(day_data) > 0:
                # Znajdź indeksy w ciągłej numeracji
                day_indices = [i for i, ts in enumerate(last_2_days.index) if ts.date() == date]
                
                if day_indices:
                    # Zaznacz dołek (najlepszy moment do zakupu)
                    low_idx_in_day = day_data['Low'].idxmin()
                    low_idx_continuous = list(last_2_days.index).index(low_idx_in_day)
                    
                    fig.add_trace(go.Scatter(
                        x=[low_idx_continuous],
                        y=[day_data.loc[low_idx_in_day, 'Low']],
                        mode='markers+text',
                        name='💎 Dołek (kup)',
                        marker=dict(size=15, color='green', symbol='diamond'),
                        text=[f"${day_data.loc[low_idx_in_day, 'Low']:.2f}"],
                        textposition='bottom center',
                        showlegend=False,
                        hovertemplate=f"<b>Dołek - {low_idx_in_day.strftime('%Y-%m-%d %H:%M')}</b><br>Cena: ${day_data.loc[low_idx_in_day, 'Low']:.2f}<extra></extra>"
                    ))
                    
                    # Zaznacz szczyt (najlepszy moment do sprzedaży)
                    high_idx_in_day = day_data['High'].idxmax()
                    high_idx_continuous = list(last_2_days.index).index(high_idx_in_day)
                    
                    fig.add_trace(go.Scatter(
                        x=[high_idx_continuous],
                        y=[day_data.loc[high_idx_in_day, 'High']],
                        mode='markers+text',
                        name='📈 Szczyt (sprzedaj)',
                        marker=dict(size=15, color='red', symbol='triangle-up'),
                        text=[f"${day_data.loc[high_idx_in_day, 'High']:.2f}"],
                        textposition='top center',
                        showlegend=False,
                        hovertemplate=f"<b>Szczyt - {high_idx_in_day.strftime('%Y-%m-%d %H:%M')}</b><br>Cena: ${day_data.loc[high_idx_in_day, 'High']:.2f}<extra></extra>"
                    ))
    
    # Utwórz etykiety dla osi X (co N minut, aby nie było za gęsto)
    tick_interval = max(1, len(x_continuous) // 20)  # ~20 etykiet
    tick_positions = x_continuous[::tick_interval]
    tick_labels = [x_labels[i] for i in tick_positions]
    
    fig.update_layout(
        title='Wykres świecowy - Dane intraday (co minutę, tylko godziny handlu)',
        xaxis_title='Czas (ciągły, bez przerw)',
        yaxis_title='Cena ($)',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=45
        )
    )
    
    return fig

def main():
    # Inicjalizuj ulubione, własne tickery, nazwy spółek i logo
    init_favorites()
    init_custom_tickers()
    init_stock_names()
    init_stock_logos()
    
    # Inicjalizuj flagi dla DAX - zawsze False przy starcie, aby uniknąć automatycznego ładowania
    if 'dax_main_data_loaded' not in st.session_state:
        st.session_state.dax_main_data_loaded = False
    if 'dax_data_loaded' not in st.session_state:
        st.session_state.dax_data_loaded = False
    if 'cache_version' not in st.session_state:
        st.session_state.cache_version = 0
    
    # Nagłówek
    st.markdown('<h1 class="main-header">📈 Przewidywanie Cen Akcji</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        
        # Przycisk do uruchomienia daily_update w tle
        if st.button("🚀 Uruchom Daily Update", type="primary"):
            if 'update_running' not in st.session_state:
                st.session_state.update_running = False
            
            if not st.session_state.update_running:
                # Uruchom aktualizację w wątku (nie blokuje UI)
                st.session_state.update_running = True
                st.session_state.update_start_time = datetime.now()
                st.session_state.update_results = None
                st.session_state.update_error = None
                st.session_state.update_progress = "Rozpoczynanie aktualizacji..."
                
                def run_update_in_thread():
                    try:
                        print("[DAILY UPDATE][THREAD] Start")
                        results = run_full_daily_update(include_german_stocks=True, all_german_stocks=True)
                        print("[DAILY UPDATE][THREAD] Zakończono")
                        st.session_state.update_results = results
                        st.session_state.update_running = False
                        st.session_state.update_progress = f"✓ Zakończono! Zaktualizowano {len(results)} spółek"
                        st.session_state.update_completed = True  # Flaga do powiadomienia
                        # Zwiększ wersję cache, aby wymusić odświeżenie danych po zakończeniu Daily Update
                        if 'cache_version' not in st.session_state:
                            st.session_state.cache_version = 0
                        st.session_state.cache_version += 1
                    except Exception as e:
                        st.session_state.update_error = str(e)
                        st.session_state.update_running = False
                        st.session_state.update_progress = f"✗ Błąd: {e}"
                        st.session_state.update_completed = True
                
                thread = threading.Thread(target=run_update_in_thread, daemon=True)
                thread.start()
                st.success("✓ Aktualizacja uruchomiona w tle!")
                st.rerun()
        
        # Status aktualizacji (jeśli trwa)
        if st.session_state.get('update_running', False):
            start_time = st.session_state.get('update_start_time', datetime.now())
            elapsed = datetime.now() - start_time
            st.info(f"🔄 Aktualizacja w toku... (czas: {elapsed.seconds}s)")
            st.caption("Możesz kontynuować korzystanie z aplikacji - używa ostatnich zapisanych danych")
        elif st.button("📥 Pobierz dane z Yahoo teraz (blokuje UI)"):
            with st.spinner("Pobieranie danych i trenowanie modeli..."):
                try:
                    results = run_full_daily_update(include_german_stocks=True, all_german_stocks=True)
                    st.session_state.update_results = results
                    st.session_state.update_completed = True
                    st.session_state.update_running = False
                    st.session_state.update_progress = f"✓ Zakończono! Zaktualizowano {len(results)} spółek"
                    st.session_state.update_error = None
                    st.session_state.cache_version = st.session_state.get('cache_version', 0) + 1
                    st.success(f"✅ Aktualizacja zakończona! ({len(results)} spółek)")
                    st.rerun()
                except Exception as e:
                    st.error(f"✗ Błąd podczas ręcznej aktualizacji: {e}")
                    st.session_state.update_error = str(e)
                    st.session_state.update_running = False
            
        # Powiadomienie o zakończeniu aktualizacji
        if st.session_state.get('update_completed', False) and st.session_state.get('update_results') is not None:
            results = st.session_state.update_results
            if results:
                st.success(f"✅ Aktualizacja zakończona! Zaktualizowano {len(results)} spółek. Dane zostały automatycznie odświeżone.")
                with st.expander(f"📊 Wyniki aktualizacji ({len(results)} spółek)", expanded=False):
                    df_summary = pd.DataFrame([
                        {
                            'Symbol': r['symbol'],
                            'Nazwa spółki': get_stock_name(r['symbol']),
                            'Kierunek': r.get('predicted_direction', 'N/A'),
                            'Prawdopod.': f"{r.get('direction_probability', 0)*100:.1f}%",
                            'Cena': f"${r['current_price']:.2f}",
                            'Zmiana %': f"{r['change_percent']:.2f}%"
                        }
                        for r in results[:20]  # Pokaż pierwsze 20
                    ])
                    df_summary = clean_dataframe_for_streamlit(df_summary)
                    st.dataframe(df_summary, width='stretch', hide_index=True)
                    if len(results) > 20:
                        st.caption(f"... i {len(results) - 20} więcej spółek")
                
                # Przycisk do zamknięcia powiadomienia
                if st.button("✖️ Zamknij powiadomienie", key="close_notification"):
                    st.session_state.update_results = None
                    st.session_state.update_completed = False
                    st.rerun()
        
        st.divider()
        
        # Wczytaj symbole
        symbols_dict, multi_predictor = load_symbols()
        
        if not symbols_dict or not symbols_dict['all']:
            st.error("Nie znaleziono symboli w pliku raport.xlsx")
            return
        
        symbols = symbols_dict['all']
        open_symbols = symbols_dict.get('open', [])
        closed_symbols = symbols_dict.get('closed', [])
        
        st.success(f"Znaleziono {len(symbols)} spółek ({len(open_symbols)} otwartych, {len(closed_symbols)} historycznych)")
        
        # Wybór spółki do szczegółowej analizy
        st.subheader("📊 Szczegółowa analiza")
        
        # Przygotuj opcje z nazwami spółek
        symbol_options = []
        for symbol in symbols:
            stock_name = get_stock_name(symbol)
            if stock_name != symbol:
                symbol_options.append(f"{symbol} - {stock_name}")
            else:
                symbol_options.append(symbol)
        
        selected_option = st.selectbox(
            "Wybierz spółkę:",
            options=symbol_options if symbol_options else ["Brak spółek"],
            index=0
        )
        
        # Wyciągnij symbol z wybranej opcji
        if " - " in selected_option:
            selected_symbol = selected_option.split(" - ")[0]
        else:
            selected_symbol = selected_option
        
        st.divider()
        
        # Zarządzanie własnymi tickerami
        st.subheader("➕ Własne tickery")
        
        # Formularz do dodawania tickera
        with st.form("add_ticker_form"):
            new_ticker = st.text_input(
                "Dodaj nowy ticker:",
                placeholder="np. AAPL, MSFT, TSLA",
                help="Wpisz symbol tickera (np. AAPL dla Apple, TSLA dla Tesla)"
            )
            submitted = st.form_submit_button("➕ Dodaj", width='stretch')
        
        if submitted and new_ticker:
            ticker_upper = new_ticker.strip().upper()
            if ticker_upper:
                if add_custom_ticker(ticker_upper):
                    # Pobierz nazwę spółki
                    with st.spinner(f"Pobieranie nazwy dla {ticker_upper}..."):
                        stock_name = get_stock_name(ticker_upper)
                    st.success(f"✓ Dodano ticker: {ticker_upper} ({stock_name})")
                    st.info("💡 Uruchom '🚀 Uruchom Daily Update' aby pobrać dane dla nowego tickera.")
                    st.rerun()
                else:
                    st.warning(f"Ticker {ticker_upper} już istnieje lub jest nieprawidłowy")
        
        # Lista własnych tickerów
        if st.session_state.custom_tickers:
            st.caption(f"Własne tickery ({len(st.session_state.custom_tickers)}):")
            for ticker in st.session_state.custom_tickers:
                col1, col2 = st.columns([3, 1])
                with col1:
                    stock_name = get_stock_name(ticker)
                    if stock_name != ticker:
                        st.text(f"{ticker} - {stock_name}")
                    else:
                        st.text(ticker)
                with col2:
                    if st.button("🗑️", key=f"remove_{ticker}", help=f"Usuń {ticker}"):
                        remove_custom_ticker(ticker)
                        st.rerun()
            
            if st.button("🗑️ Usuń wszystkie"):
                st.session_state.custom_tickers = []
                save_custom_tickers()
                st.rerun()
        else:
            st.caption("Brak własnych tickerów")
        
        st.divider()
        
        # Informacje
        st.info("""
        **Funkcje:**
        - Przewidywania na kolejny dzień
        - Wykresy cen i wskaźników technicznych
        - Statystyki modeli ML
        - Porównanie wszystkich spółek
        """)
    
    # Główna zawartość
    if multi_predictor is None:
        st.error("Nie można załadować danych. Sprawdź plik raport.xlsx")
        return
    
    # Pobierz dane przewidywań - używaj zapisanych modeli (szybkie, nie blokuje UI)
    predictions_data = get_dashboard_predictions(symbols, include_all_from_db=False)
    
    if not predictions_data:
        st.warning("Brak danych do wyświetlenia. Kliknij '🚀 Uruchom Daily Update' aby pobrać dane.")
        return
    
    # Rozdziel dane na otwarte i historyczne pozycje
    open_predictions = [p for p in predictions_data if p['symbol'] in open_symbols]
    closed_predictions = [p for p in predictions_data if p['symbol'] in closed_symbols]
    
    # Tabs
    tab_labels = ["📊 Dashboard", "📈 Szczegóły spółki", "📋 Wszystkie spółki", "📉 Statystyki", "🇩🇪 Indeks DAX", "📊 Skuteczność predykcji"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)
    
    if st.session_state.get('switch_tab_to'):
        tab_to_switch = st.session_state.pop('switch_tab_to')
        trigger_tab_switch(tab_to_switch)
    
    with tab1:
        st.header("Dashboard - Przegląd wszystkich spółek")
        
        # Metryki ogólne
        col1, col2, col3, col4 = st.columns(4)
        
        total_stocks = len(predictions_data)
        avg_change = np.mean([p['change_percent'] for p in predictions_data])
        up_predictions = sum(1 for p in predictions_data if p.get('predicted_direction') == 'UP')
        avg_accuracy = np.mean([p['test_accuracy'] for p in predictions_data if p.get('test_accuracy', 0) > 0])
        
        with col1:
            st.metric("Liczba spółek", total_stocks)
        with col2:
            st.metric("Średnia zmiana (%)", f"{avg_change:.2f}%")
        with col3:
            st.metric("Przewidywania UP", f"{up_predictions}/{total_stocks}")
        with col4:
            st.metric("Średnia dokładność", f"{avg_accuracy:.2%}" if avg_accuracy > 0 else "N/A")
        
        st.divider()
        
        favorites_list = list(st.session_state.get('favorites', []))
        favorite_predictions = []
        favorites_without_data = []
        
        if favorites_list:
            st.subheader(f"⭐ Ulubione ({len(favorites_list)})")
            
            predictions_map = {p['symbol']: p for p in predictions_data}
            missing_favorites = []
            
            for symbol in favorites_list:
                prediction = predictions_map.get(symbol)
                if prediction:
                    favorite_predictions.append(prediction)
                else:
                    missing_favorites.append(symbol)
            
            remaining_missing = []
            if missing_favorites:
                try:
                    cache_version = st.session_state.get('cache_version', 0)
                    missing_predictions = get_predictions_data(
                        symbols=missing_favorites,
                        include_all_from_db=False,
                        use_saved_models=True,
                        cache_version=cache_version
                    )
                    missing_map = {p['symbol']: p for p in missing_predictions}
                    for symbol in missing_favorites:
                        if symbol in missing_map:
                            favorite_predictions.append(missing_map[symbol])
                        else:
                            remaining_missing.append(symbol)
                except Exception:
                    remaining_missing = list(missing_favorites)
            else:
                remaining_missing = []
            
            if remaining_missing:
                try:
                    dax_predictions = get_dax_predictions(include_all_from_db=True)
                    dax_map = {p['symbol']: p for p in dax_predictions}
                    resolved = []
                    for symbol in remaining_missing:
                        prediction = dax_map.get(symbol)
                        if prediction:
                            favorite_predictions.append(prediction)
                            resolved.append(symbol)
                    remaining_missing = [s for s in remaining_missing if s not in resolved]
                except Exception:
                    pass
            
            favorites_without_data = remaining_missing
            
            if favorite_predictions:
                period_option_fav = st.radio(
                    "Wybierz okres przewidywania:",
                    ["1 dzień", "1 tydzień", "1 miesiąc", "6 miesięcy"],
                    horizontal=True,
                    key="favorites_period"
                )
                
                period_map_fav = {
                    "1 dzień": None,
                    "1 tydzień": "prediction_week",
                    "1 miesiąc": "prediction_month",
                    "6 miesięcy": "prediction_6months"
                }
                selected_period_fav = period_map_fav[period_option_fav]
                
                df_fav_data = []
                change_values_fav = []
                for p in favorite_predictions:
                    source_pred = p.get(selected_period_fav) if (selected_period_fav and selected_period_fav in p) else p
                    change_val = source_pred.get('change_percent', p.get('change_percent', 0))
                    change_values_fav.append(change_val)
                    df_fav_data.append({
                        'Symbol': p['symbol'],
                        'Nazwa spółki': get_stock_name(p['symbol']),
                        'Kierunek': source_pred.get('predicted_direction', 'N/A'),
                        'Prawdopod.': f"{source_pred.get('direction_probability', 0)*100:.1f}%",
                        'Aktualna cena': source_pred.get('current_price', p.get('current_price', 0)),
                        'Przewidywana cena': source_pred.get('predicted_price', source_pred.get('current_price', p.get('current_price', 0))),
                        'Zmiana': source_pred.get('change', p.get('change', 0)),
                        'Zmiana %': change_val,
                        'Dokładność': p.get('test_accuracy', 0)
                    })
                
                df_fav = pd.DataFrame(df_fav_data)
                df_fav['Zmiana_num'] = change_values_fav
                df_fav = df_fav.sort_values('Zmiana_num', ascending=False)
                
                table_cols = [
                    'Symbol', 'Nazwa spółki', 'Kierunek', 'Prawdopod.',
                    'Aktualna cena', 'Przewidywana cena', 'Zmiana',
                    'Zmiana %', 'Dokładność'
                ]
                df_fav_display = df_fav[table_cols].copy()
                df_fav_display['Aktualna cena'] = df_fav_display['Aktualna cena'].apply(lambda x: f"${x:.2f}")
                df_fav_display['Przewidywana cena'] = df_fav_display['Przewidywana cena'].apply(lambda x: f"${x:.2f}")
                df_fav_display['Zmiana'] = df_fav_display['Zmiana'].apply(lambda x: f"${x:.2f}")
                df_fav_display['Zmiana %'] = df_fav_display['Zmiana %'].apply(lambda x: f"{x:.2f}%")
                df_fav_display['Dokładność'] = df_fav_display['Dokładność'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
                df_fav_display = clean_dataframe_for_streamlit(df_fav_display)
                
                config = {
                    'Symbol': st.column_config.Column(width='small'),
                    'Nazwa spółki': st.column_config.Column(width='medium'),
                    'Kierunek': st.column_config.Column(width='small'),
                    'Prawdopod.': st.column_config.Column(width='small'),
                    'Aktualna cena': st.column_config.Column(width='small'),
                    'Przewidywana cena': st.column_config.Column(width='small'),
                    'Zmiana': st.column_config.Column(width='small'),
                    'Zmiana %': st.column_config.Column(width='small'),
                    'Dokładność': st.column_config.Column(width='small')
                }
                
                st.dataframe(
                    df_fav_display,
                    hide_index=True,
                    use_container_width=True,
                    height=min(400, 60 + 28 * len(df_fav_display)),
                    column_config=config
                )
                
                action_cols = st.columns(2)
                with action_cols[0]:
                    selected_fav_symbol = st.selectbox(
                        "Szybkie akcje dla ulubionej spółki:",
                        df_fav_display['Symbol'],
                        key="fav_quick_select"
                    )
                    selected_fav_name = get_stock_name(selected_fav_symbol)
                with action_cols[1]:
                    st.markdown(f"**{selected_fav_name}**")
                
                quick_cols = st.columns(2)
                with quick_cols[0]:
                    if st.button("📈 Szczegóły (ulubione)", key="fav_quick_details"):
                        navigate_to_details(selected_fav_symbol)
                with quick_cols[1]:
                    if st.button("🗑️ Usuń z ulubionych", key="fav_quick_remove"):
                        toggle_favorite(selected_fav_symbol)
                        st.rerun()
            else:
                st.info("Brak danych dla ulubionych spółek. Uruchom najpierw Daily Update.")
            
            if favorites_without_data:
                missing_str = ", ".join(sorted(favorites_without_data))
                st.caption(f"Brak zapisanych danych dla: {missing_str}. Uruchom aktualizację dla tych spółek.")
        else:
            st.info("Brak ulubionych spółek. Dodaj je w sekcjach poniżej.")

        try:
            all_saved_predictions = get_dax_predictions(include_all_from_db=True)
        except Exception:
            all_saved_predictions = []
        combined_predictions = merge_prediction_lists(predictions_data, all_saved_predictions)

        top_panels = st.columns(1)
        
        with top_panels[0]:
            st.subheader("📊 Metryki ogólne")
            col_a, col_b, col_c = st.columns(3)
            total_stocks = len(predictions_data)
            avg_change = np.mean([p['change_percent'] for p in predictions_data]) if predictions_data else 0
            up_predictions = sum(1 for p in predictions_data if p.get('predicted_direction') == 'UP')
            
            col_a.metric("Liczba spółek", total_stocks)
            col_b.metric("Średnia zmiana (%)", f"{avg_change:.2f}%")
            if total_stocks > 0:
                col_c.metric("Przewidywania UP", f"{up_predictions}/{total_stocks}")
            else:
                col_c.metric("Przewidywania UP", "0/0")
            
            if favorite_predictions:
                st.caption(f"Ulubione ({len(favorite_predictions)})")
                fav_preview = pd.DataFrame([
                    {
                        'Symbol': p['symbol'],
                        'Kierunek': p.get('predicted_direction', 'N/A'),
                        'Prawdopod.': f"{p.get('direction_probability', 0)*100:.1f}%",
                        'Zmiana %': f"{p['change_percent']:.2f}%"
                    }
                    for p in favorite_predictions[:5]
                ])
                fav_preview = clean_dataframe_for_streamlit(fav_preview)
                st.dataframe(fav_preview, hide_index=True, use_container_width=True)
            else:
                st.caption("Brak ulubionych spółek")
        
        st.subheader("📈 Trendy i pewność")
        trend_cols = st.columns(2)

        with trend_cols[0]:
            st.caption("Top 10 tygodniowych zmian")
            top_week_rows = []
            for p in combined_predictions:
                week_pred = p.get('prediction_week')
                if week_pred:
                    top_week_rows.append({
                        'Symbol': p['symbol'],
                        'Kierunek': week_pred.get('predicted_direction', 'N/A'),
                        'Prawdopod.': week_pred.get('direction_probability', 0),
                        'Zmiana % (7d)': week_pred['change_percent']
                    })
            if top_week_rows:
                top_week_df = pd.DataFrame(top_week_rows)
                top_week_df = top_week_df.sort_values('Zmiana % (7d)', ascending=False).head(10)
                top_week_df['Prawdopod.'] = top_week_df['Prawdopod.'].apply(lambda x: f"{x*100:.1f}%")
                top_week_df['Zmiana % (7d)'] = top_week_df['Zmiana % (7d)'].apply(lambda x: f"{x:.2f}%")
                top_week_df = clean_dataframe_for_streamlit(top_week_df)
                st.dataframe(top_week_df, hide_index=True, use_container_width=True)
            else:
                st.caption("Brak predykcji tygodniowych do wyświetlenia.")

        with trend_cols[1]:
            st.caption("Top 20 najwyższej pewności (tylko dodatnie zmiany)")
            confidence_rows = []
            for p in combined_predictions:
                change_pct = p.get('change_percent', 0)
                if change_pct is None or change_pct <= 0:
                    continue
                confidence_rows.append({
                    'Symbol': p['symbol'],
                    'Kierunek': p.get('predicted_direction', 'N/A'),
                    'Prawdopod.': p.get('direction_probability', 0),
                    'Zmiana % (1d)': change_pct
                })
            if confidence_rows:
                conf_df = pd.DataFrame(confidence_rows)
                conf_df = conf_df.sort_values('Prawdopod.', ascending=False).head(20)
                conf_df['Prawdopod.'] = conf_df['Prawdopod.'].apply(lambda x: f"{x*100:.1f}%")
                conf_df['Zmiana % (1d)'] = conf_df['Zmiana % (1d)'].apply(lambda x: f"{x:.2f}%")
                conf_df = clean_dataframe_for_streamlit(conf_df)
                st.dataframe(conf_df, hide_index=True, use_container_width=True)
            else:
                st.caption("Brak spółek z dodatnimi zmianami i wysoką pewnością.")
        
        st.divider()
        
        open_col, closed_col = st.columns(2)
        
        with open_col:
            if open_predictions:
                st.subheader("🟢 Otwarte pozycje")
                
                period_option = st.radio(
                    "Wybierz okres przewidywania:",
                    ["1 dzień", "1 tydzień", "1 miesiąc", "6 miesięcy"],
                    horizontal=True,
                    key="open_period"
                )
                
                period_map = {
                    "1 dzień": None,
                    "1 tydzień": "prediction_week",
                    "1 miesiąc": "prediction_month",
                    "6 miesięcy": "prediction_6months"
                }
                selected_period = period_map[period_option]
                
                df_open_data = []
                for p in open_predictions:
                    source_pred = p.get(selected_period) if (selected_period and selected_period in p) else p
                    df_open_data.append({
                        'Symbol': p['symbol'],
                        'Nazwa spółki': get_stock_name(p['symbol']),
                        'Kierunek': source_pred.get('predicted_direction', 'N/A'),
                        'Prawdopod.': f"{source_pred.get('direction_probability', 0)*100:.1f}%",
                        'Aktualna cena': f"${p['current_price']:.2f}",
                        'Przewidywana cena': f"${source_pred.get('predicted_price', p['predicted_price']):.2f}",
                        'Zmiana': f"${source_pred.get('change', p['change']):.2f}",
                        'Zmiana %': source_pred.get('change_percent', p['change_percent']),
                        'Dokładność': f"{p.get('test_accuracy', 0):.2%}" if p.get('test_accuracy', 0) > 0 else "N/A"
                    })
                
                df_open = pd.DataFrame(df_open_data)
                change_values = [row['Zmiana %'] for row in df_open_data]
                df_open['Zmiana_num'] = change_values
                df_open = df_open.sort_values('Zmiana_num', ascending=False)
                df_open = df_open.drop('Zmiana_num', axis=1)
                df_open_display = df_open.copy()
                df_open_display['Zmiana %'] = df_open_display['Zmiana %'].apply(lambda x: f"{x:.2f}%")
                df_open_display = clean_dataframe_for_streamlit(df_open_display)
                
                table_height = min(420, 120 + len(df_open_display) * 28)
                st.dataframe(df_open_display, hide_index=True, use_container_width=True, height=table_height)
                
                selected_open_symbol = st.selectbox("Wybierz spółkę do szczegółów:", df_open['Symbol'], key="open_details_select")
                if st.button("📈 Przejdź do szczegółów", key="open_details_btn"):
                    navigate_to_details(selected_open_symbol)
                
                fav_cols = st.columns([3, 1])
                with fav_cols[0]:
                    open_fav_symbol = st.selectbox("Dodaj do ulubionych:", df_open['Symbol'], key="open_fav_select")
                with fav_cols[1]:
                    if st.button("⭐ Dodaj", key="open_fav_btn"):
                        toggle_favorite(open_fav_symbol)
                        st.success(f"Dodano {open_fav_symbol} do ulubionych")
                        st.rerun()
            else:
                st.subheader("🟢 Otwarte pozycje")
                st.info("Brak otwartych pozycji")
            
        with closed_col:
            if closed_predictions:
                st.subheader("🔴 Historyczne pozycje")
                
                period_option_closed = st.radio(
                    "Wybierz okres przewidywania:",
                    ["1 dzień", "1 tydzień", "1 miesiąc", "6 miesięcy"],
                    horizontal=True,
                    key="closed_period"
                )
                
                period_map = {
                    "1 dzień": None,
                    "1 tydzień": "prediction_week",
                    "1 miesiąc": "prediction_month",
                    "6 miesięcy": "prediction_6months"
                }
                selected_period = period_map[period_option_closed]
                
                df_closed_data = []
                for p in closed_predictions:
                    source_pred = p.get(selected_period) if (selected_period and selected_period in p) else p
                    df_closed_data.append({
                        'Symbol': p['symbol'],
                        'Nazwa spółki': get_stock_name(p['symbol']),
                        'Kierunek': source_pred.get('predicted_direction', 'N/A'),
                        'Prawdopod.': f"{source_pred.get('direction_probability', 0)*100:.1f}%",
                        'Aktualna cena': f"${p['current_price']:.2f}",
                        'Przewidywana cena': f"${source_pred.get('predicted_price', p['predicted_price']):.2f}",
                        'Zmiana': f"${source_pred.get('change', p['change']):.2f}",
                        'Zmiana %': source_pred.get('change_percent', p['change_percent']),
                        'Dokładność': f"{p.get('test_accuracy', 0):.2%}" if p.get('test_accuracy', 0) > 0 else "N/A"
                    })
                
                df_closed = pd.DataFrame(df_closed_data)
                change_values = [row['Zmiana %'] for row in df_closed_data]
                df_closed['Zmiana_num'] = change_values
                df_closed = df_closed.sort_values('Zmiana_num', ascending=False)
                df_closed = df_closed.drop('Zmiana_num', axis=1)
                df_closed_display = df_closed.copy()
                df_closed_display['Zmiana %'] = df_closed_display['Zmiana %'].apply(lambda x: f"{x:.2f}%")
                df_closed_display = clean_dataframe_for_streamlit(df_closed_display)
                
                table_height = min(420, 120 + len(df_closed_display) * 28)
                st.dataframe(df_closed_display, hide_index=True, use_container_width=True, height=table_height)
                
                selected_closed_symbol = st.selectbox("Wybierz spółkę do szczegółów:", df_closed['Symbol'], key="closed_details_select")
                if st.button("📈 Przejdź do szczegółów (historyczne)", key="closed_details_btn"):
                    navigate_to_details(selected_closed_symbol)
                
                fav_closed_cols = st.columns([3, 1])
                with fav_closed_cols[0]:
                    closed_fav_symbol = st.selectbox("Dodaj do ulubionych (historyczne):", df_closed['Symbol'], key="closed_fav_select")
                with fav_closed_cols[1]:
                    if st.button("⭐ Dodaj (hist.)", key="closed_fav_btn"):
                        toggle_favorite(closed_fav_symbol)
                        st.success(f"Dodano {closed_fav_symbol} do ulubionych")
                        st.rerun()
            else:
                st.subheader("🔴 Historyczne pozycje")
                st.info("Brak historycznych pozycji")
        
        # Wykres zmian procentowych
        fig_changes = go.Figure()
        
        colors = ['green' if p['change_percent'] > 0 else 'red' for p in predictions_data]
        symbols_list = [p['symbol'] for p in predictions_data]
        changes = [p['change_percent'] for p in predictions_data]
        
        # Sortuj dla lepszej wizualizacji
        sorted_data = sorted(zip(symbols_list, changes, colors), key=lambda x: x[1], reverse=True)
        symbols_sorted, changes_sorted, colors_sorted = zip(*sorted_data)
        
        fig_changes.add_trace(go.Bar(
            x=list(symbols_sorted),
            y=list(changes_sorted),
            marker_color=list(colors_sorted),
            text=[f"{c:.2f}%" for c in changes_sorted],
            textposition='outside'
        ))
        
        fig_changes.update_layout(
            title='Przewidywane zmiany cen (%)',
            xaxis_title='Symbol',
            yaxis_title='Zmiana (%)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_changes, use_container_width=True)
    
    with tab2:
        # Użyj selected_symbol z session_state jeśli jest ustawione, w przeciwnym razie użyj lokalnej zmiennej
        current_selected_symbol = st.session_state.get('selected_symbol', selected_symbol)
        
        # Wyświetl nazwę spółki w nagłówku
        stock_name = get_stock_name(current_selected_symbol)
        if stock_name != current_selected_symbol:
            st.header(f"{current_selected_symbol} - {stock_name}")
        else:
            st.header(f"{current_selected_symbol}")
        
        if st.session_state.get('details_fetch_message'):
            st.success(st.session_state.pop('details_fetch_message'))
        if st.session_state.get('details_fetch_error'):
            st.error(st.session_state.pop('details_fetch_error'))
        
        st.info("💡 Aby zaktualizować dane dla tej spółki, użyj przycisku '🚀 Uruchom Daily Update' w zakładce Dashboard.")
        
        st.divider()
        
        # Znajdź dane dla wybranej spółki
        selected_data = next((p for p in predictions_data if p['symbol'] == current_selected_symbol), None)
        
        if selected_data is None:
            st.warning(f"Brak danych dla {current_selected_symbol}")
            
            # Przycisk do pobrania danych tylko dla tej jednej spółki
            if st.button(f"📥 Pobierz dane i wytrenuj model dla {current_selected_symbol}", key=f"fetch_single_{current_selected_symbol}", type="primary"):
                with st.spinner(f"Pobieranie danych i trenowanie modelu dla {current_selected_symbol}..."):
                    success, info_msg = run_single_symbol_update(current_selected_symbol, retrain_weekly=True)
                if success:
                    message = f"✓ Pobrano dane i wytrenowano model dla {current_selected_symbol}."
                    if info_msg:
                        message += f" {info_msg}"
                    st.session_state['details_fetch_message'] = message
                else:
                    st.session_state['details_fetch_error'] = f"Nie udało się pobrać danych dla {current_selected_symbol}: {info_msg}"
                st.rerun()
        else:
            with st.spinner(f"Ładowanie danych dla {current_selected_symbol}..."):
                predictor = get_predictor_for_symbol(current_selected_symbol)
            
            if predictor is None:
                st.error(f"Nie można załadować danych dla {current_selected_symbol}")
                return
            
            # Metryki dla różnych okresów
            st.subheader("Przewidywania na różne okresy")
            
            # Tabela z przewidywaniami na różne okresy
            periods_data = []
            
            # 1 dzień
            if 'predicted_direction' in selected_data:
                periods_data.append({
                    'Okres': '1 dzień',
                    'Kierunek': selected_data.get('predicted_direction', 'N/A'),
                    'Prawdopod.': f"{selected_data.get('direction_probability', 0)*100:.1f}%",
                    'Aktualna cena': f"${selected_data['current_price']:.2f}",
                    'Przewidywana cena': f"${selected_data['predicted_price']:.2f}",
                    'Zmiana': f"${selected_data['change']:.2f}",
                    'Zmiana %': f"{selected_data['change_percent']:.2f}%"
                })
            
            # 1 tydzień
            if 'prediction_week' in selected_data:
                pw = selected_data['prediction_week']
                periods_data.append({
                    'Okres': '1 tydzień (7 dni)',
                    'Kierunek': pw.get('predicted_direction', 'N/A'),
                    'Prawdopod.': f"{pw.get('direction_probability', 0)*100:.1f}%",
                    'Aktualna cena': f"${selected_data['current_price']:.2f}",
                    'Przewidywana cena': f"${pw['predicted_price']:.2f}",
                    'Zmiana': f"${pw['change']:.2f}",
                    'Zmiana %': f"{pw['change_percent']:.2f}%"
                })
            
            # 1 miesiąc
            if 'prediction_month' in selected_data:
                pm = selected_data['prediction_month']
                periods_data.append({
                    'Okres': '1 miesiąc (30 dni)',
                    'Kierunek': pm.get('predicted_direction', 'N/A'),
                    'Prawdopod.': f"{pm.get('direction_probability', 0)*100:.1f}%",
                    'Aktualna cena': f"${selected_data['current_price']:.2f}",
                    'Przewidywana cena': f"${pm['predicted_price']:.2f}",
                    'Zmiana': f"${pm['change']:.2f}",
                    'Zmiana %': f"{pm['change_percent']:.2f}%"
                })
            
            # 6 miesięcy
            if 'prediction_6months' in selected_data:
                p6m = selected_data['prediction_6months']
                periods_data.append({
                    'Okres': '6 miesięcy (180 dni)',
                    'Kierunek': p6m.get('predicted_direction', 'N/A'),
                    'Prawdopod.': f"{p6m.get('direction_probability', 0)*100:.1f}%",
                    'Aktualna cena': f"${selected_data['current_price']:.2f}",
                    'Przewidywana cena': f"${p6m['predicted_price']:.2f}",
                    'Zmiana': f"${p6m['change']:.2f}",
                    'Zmiana %': f"{p6m['change_percent']:.2f}%"
                })
            
            if periods_data:
                df_periods = pd.DataFrame(periods_data)
                df_periods = clean_dataframe_for_streamlit(df_periods)
                st.dataframe(df_periods, width='stretch', hide_index=True)
            
            st.divider()
            
            # Metryki ogólne
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Aktualna cena", f"${selected_data['current_price']:.2f}")
            with col2:
                direction_icon = "↑" if selected_data.get('predicted_direction') == 'UP' else "↓"
                st.metric("Kierunek (1 dzień)", f"{direction_icon} {selected_data.get('predicted_direction', 'N/A')}")
            with col3:
                prob = selected_data.get('direction_probability', 0) * 100
                st.metric("Prawdopodobieństwo", f"{prob:.1f}%")
            with col4:
                st.metric("Dokładność modelu", f"{selected_data.get('test_accuracy', 0):.2%}" if selected_data.get('test_accuracy', 0) > 0 else "N/A")
            
            st.divider()
            
            # Wykres przewidywań na przyszły tydzień
            st.subheader("📅 Przewidywania na przyszły tydzień (7 dni)")
            
            # Generuj przewidywania dla każdego dnia w tygodniu używając historycznych wzorców
            try:
                current_price = predictor.data['Close'].iloc[-1]
                last_date = predictor.data.index[-1]
                
                # Przewiduj całkowitą zmianę na tydzień
                week_prediction = predictor.predict_week()
                total_week_change_pct = week_prediction['change_percent'] / 100
                predicted_week_price = week_prediction['predicted_price']
                
                # Znajdź historyczne tygodnie z podobną zmianą
                data = predictor.data.copy()
                
                # Oblicz tygodniowe zmiany historyczne (5 dni roboczych)
                weekly_changes = []
                weekly_patterns = []
                
                for i in range(5, len(data), 5):  # Co 5 dni (tydzień roboczy)
                    if i >= len(data):
                        break
                    week_start_idx = max(0, i - 5)
                    week_end_idx = i
                    
                    if week_end_idx > week_start_idx:
                        week_start_price = data['Close'].iloc[week_start_idx]
                        week_end_price = data['Close'].iloc[week_end_idx]
                        week_change_pct = (week_end_price - week_start_price) / week_start_price
                        
                        # Pobierz dzienne ceny z tego tygodnia
                        week_data = data.iloc[week_start_idx:week_end_idx+1]
                        week_daily_prices = week_data['Close'].values
                        week_daily_changes = week_data['Close'].pct_change().fillna(0).values
                        
                        weekly_changes.append(week_change_pct)
                        weekly_patterns.append({
                            'change_pct': week_change_pct,
                            'daily_prices': week_daily_prices,
                            'daily_changes': week_daily_changes,
                            'start_price': week_start_price
                        })
                
                # Znajdź najbardziej podobne wzorce (tygodnie z podobną zmianą)
                if weekly_patterns:
                    # Sortuj według podobieństwa do przewidywanej zmiany
                    weekly_patterns.sort(key=lambda x: abs(x['change_pct'] - total_week_change_pct))
                    
                    # Weź 3 najbardziej podobne wzorce i uśrednij je
                    similar_patterns = weekly_patterns[:min(3, len(weekly_patterns))]
                    
                    # Uśrednij dzienne zmiany z podobnych wzorców
                    avg_daily_changes = []
                    for day_idx in range(5):  # 5 dni roboczych
                        day_changes = []
                        for pattern in similar_patterns:
                            if day_idx < len(pattern['daily_changes']):
                                day_changes.append(pattern['daily_changes'][day_idx])
                        if day_changes:
                            avg_daily_changes.append(np.mean(day_changes))
                        else:
                            # Fallback: użyj średniej historycznej zmiany
                            historical_changes = data['Close'].pct_change().dropna()
                            avg_change = historical_changes.mean() if len(historical_changes) > 0 else 0.001
                            avg_daily_changes.append(avg_change)
                    
                    # Jeśli przewidywana zmiana jest większa/mniejsza, skalibruj
                    total_pattern_change = sum(avg_daily_changes)
                    if abs(total_pattern_change) > 0.0001:
                        scale_factor = total_week_change_pct / total_pattern_change
                        avg_daily_changes = [ch * scale_factor for ch in avg_daily_changes]
                else:
                    # Fallback: równomiernie rozłóż zmianę na 5 dni
                    historical_changes = data['Close'].pct_change().dropna()
                    avg_change = historical_changes.mean() if len(historical_changes) > 0 else 0.001
                    avg_daily_changes = [total_week_change_pct / 5] * 5
                
                # Generuj przewidywania dla każdego dnia
                weekly_forecast = []
                forecast_dates = []
                forecast_prices = [current_price]
                forecast_dates.append(last_date)
                
                current_predicted_price = current_price
                
                # 5 dni roboczych + weekend (2 dni bez zmian)
                for day in range(1, 8):
                    forecast_date = last_date + timedelta(days=day)
                    
                    # Sprawdź czy to dzień roboczy (poniedziałek-piątek = 0-4)
                    day_of_week = forecast_date.weekday()
                    
                    if day_of_week < 5:  # Dzień roboczy
                        day_idx = day - 1
                        if day_idx < len(avg_daily_changes):
                            daily_change_pct = avg_daily_changes[day_idx]
                        else:
                            daily_change_pct = 0
                        
                        current_predicted_price = current_predicted_price * (1 + daily_change_pct)
                        
                        # Przewiduj kierunek dla tego dnia - z dopasowaniem cech
                        # Sprawdź jakie cechy model oczekuje (kompatybilność wsteczna)
                        model_expected_features = None
                        if hasattr(predictor.model, 'feature_names_in_'):
                            model_expected_features = list(predictor.model.feature_names_in_)
                        elif hasattr(predictor, 'selected_features') and predictor.selected_features:
                            model_expected_features = predictor.selected_features
                        
                        # Przygotuj features zgodnie z oczekiwaniami modelu
                        if model_expected_features:
                            available_features = [f for f in model_expected_features if f in predictor.features.columns]
                            if len(available_features) != len(model_expected_features) and hasattr(predictor.model, 'feature_names_in_'):
                                # Model sklearn wymaga dokładnego dopasowania
                                last_features_df = pd.DataFrame(index=predictor.features.index[-1:])
                                feature_mapping = {
                                    'ATR': 'ATR_log',
                                    'TR': 'TR_log',
                                    'Volume': 'Volume_log'
                                }
                                
                                for feat in model_expected_features:
                                    if feat in predictor.features.columns:
                                        last_features_df[feat] = predictor.features[feat].iloc[-1:]
                                    elif feat in feature_mapping:
                                        mapped_feat = feature_mapping[feat]
                                        if mapped_feat in predictor.features.columns:
                                            last_features_df[feat] = predictor.features[mapped_feat].iloc[-1:]
                                        else:
                                            last_features_df[feat] = 0
                                    else:
                                        last_features_df[feat] = 0
                                
                                last_features_df = last_features_df[model_expected_features]
                                last_features = last_features_df.values
                            else:
                                last_features = predictor.features[model_expected_features].iloc[-1:].values
                        else:
                            # Model nie ma określonych cech - użyj wszystkich dostępnych
                            last_features = predictor.features.iloc[-1:].values
                        
                        direction_pred = predictor.model.predict(last_features)[0]
                        direction_proba = predictor.model.predict_proba(last_features)[0]
                        up_probability = direction_proba[1] if len(direction_proba) > 1 else 0.5
                        
                        weekly_forecast.append({
                            'day': day,
                            'date': forecast_date,
                            'price': current_predicted_price,
                            'direction': 'UP' if daily_change_pct >= 0 else 'DOWN',
                            'probability': up_probability if daily_change_pct >= 0 else (1 - up_probability),
                            'change_pct': daily_change_pct * 100
                        })
                    else:  # Weekend - cena bez zmian
                        weekly_forecast.append({
                            'day': day,
                            'date': forecast_date,
                            'price': current_predicted_price,
                            'direction': 'FLAT',
                            'probability': 0.5,
                            'change_pct': 0
                        })
                    
                    forecast_prices.append(current_predicted_price)
                    forecast_dates.append(forecast_date)
                
                # Znajdź dołek (minimum) - tylko w dniach roboczych
                workday_prices = [f['price'] for f in weekly_forecast if f['direction'] != 'FLAT']
                workday_dates = [f['date'] for f in weekly_forecast if f['direction'] != 'FLAT']
                
                if workday_prices:
                    min_price = min(workday_prices)
                    min_index = workday_prices.index(min_price)
                    min_date = workday_dates[min_index]
                else:
                    min_price = min(forecast_prices)
                    min_index = forecast_prices.index(min_price)
                    min_date = forecast_dates[min_index]
                
                # Wykres przewidywań na tydzień
                fig_week = go.Figure()
                
                # Linia przewidywań
                fig_week.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    mode='lines+markers',
                    name='Przewidywana cena',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=8, color='#2E86AB')
                ))
                
                # Zaznacz aktualną cenę
                fig_week.add_trace(go.Scatter(
                    x=[forecast_dates[0]],
                    y=[forecast_prices[0]],
                    mode='markers',
                    name='Aktualna cena',
                    marker=dict(size=15, color='green', symbol='circle'),
                    hovertemplate=f"Aktualna cena: ${forecast_prices[0]:.2f}<extra></extra>"
                ))
                
                # Zaznacz dołek (najlepszy moment do zakupu)
                fig_week.add_trace(go.Scatter(
                    x=[min_date],
                    y=[min_price],
                    mode='markers+text',
                    name='💎 Dołek (najlepszy moment do zakupu)',
                    marker=dict(size=20, color='red', symbol='diamond'),
                    text=[f"${min_price:.2f}"],
                    textposition='top center',
                    hovertemplate=f"Data: {min_date.strftime('%Y-%m-%d')}<br>Dołek: ${min_price:.2f}<br>Dzień: {min_index}<extra></extra>"
                ))
                
                # Dodaj linię poziomą dla aktualnej ceny
                fig_week.add_hline(
                    y=forecast_prices[0],
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"Aktualna: ${forecast_prices[0]:.2f}"
                )
                
                fig_week.update_layout(
                    title=f'Przewidywania cen na przyszły tydzień - {current_selected_symbol}',
                    xaxis_title='Data',
                    yaxis_title='Cena ($)',
                    height=500,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    xaxis=dict(
                        tickformat='%Y-%m-%d',
                        tickangle=45
                    )
                )
                
                st.plotly_chart(fig_week, width='stretch')
                
                # Informacje o dołku
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💎 Dołek (najlepszy moment)", f"${min_price:.2f}")
                with col2:
                    st.metric("📅 Data dołku", min_date.strftime('%Y-%m-%d'))
                with col3:
                    days_to_bottom = (min_date - last_date).days
                    st.metric("⏱ Dni do dołku", f"{days_to_bottom} dni")
                
                # Tabela z przewidywaniami na każdy dzień
                st.subheader("📊 Szczegółowe przewidywania na każdy dzień")
                forecast_df = pd.DataFrame([
                    {
                        'Dzień': f"Day {f['day']}",
                        'Data': f['date'].strftime('%Y-%m-%d'),
                        'Przewidywana cena': f"${f['price']:.2f}",
                        'Kierunek': f['direction'],
                        'Prawdopod.': f"{f['probability']*100:.1f}%",
                        'Zmiana od dziś': f"${f['price'] - current_price:.2f}",
                        'Zmiana %': f"{((f['price'] - current_price) / current_price * 100):.2f}%",
                        'Zmiana dzienna': f"{f.get('change_pct', 0):.2f}%" if 'change_pct' in f else "0.00%"
                    }
                    for f in weekly_forecast
                ])
                forecast_df = clean_dataframe_for_streamlit(forecast_df)
                st.dataframe(forecast_df, width='stretch', hide_index=True)
                
            except Exception as e:
                st.warning(f"Nie można wygenerować przewidywań na tydzień: {e}")
            
            st.divider()
            
            # Analiza intraday - zachowanie w ciągu dnia
            st.subheader("📊 Analiza zachowania w ciągu dnia (intraday)")
            
            try:
                with st.spinner("Pobieranie danych intraday (ostatnie 25 dni)..."):
                    intraday_data = get_intraday_data(current_selected_symbol, days=25)
                
                if intraday_data is not None and len(intraday_data) > 0:
                    # Analizuj wzorce
                    patterns = analyze_intraday_patterns(intraday_data)
                    
                    if patterns:
                        # Metryki najlepszych/najgorszych momentów
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("⏰ Średni czas dołku (kup)", patterns['avg_low_time'])
                        with col2:
                            st.metric("⏰ Średni czas szczytu (sprzedaj)", patterns['avg_high_time'])
                        with col3:
                            st.metric("📊 Analizowanych dni", len(patterns['daily_patterns']))
                        with col4:
                            avg_change = np.mean([p['change_pct'] for p in patterns['daily_patterns']])
                            st.metric("📈 Średnia zmiana dzienna", f"{avg_change:.2f}%")
                        
                        # Edge Finder - optymalne momenty transakcji
                        st.subheader("🎯 Edge Finder - Optymalne momenty transakcji")
                        
                        # Znajdź optymalne pary transakcji
                        trading_edges = find_trading_edges(intraday_data, min_profit_pct=0.05, lookback_window=20)
                        
                        if trading_edges:
                            # Pokaż najlepsze okazje w tabeli
                            st.info(f"✅ Znaleziono {len(trading_edges)} optymalnych okazji transakcyjnych!")
                            
                            edges_df = pd.DataFrame([
                                {
                                    'Okazja': f"#{idx+1}",
                                    '🛒 Kup (data/czas)': edge['buy_time'].strftime('%Y-%m-%d %H:%M'),
                                    '🛒 Cena zakupu': f"${edge['buy_price']:.2f}",
                                    '💵 Sprzedaj (data/czas)': edge['sell_time'].strftime('%Y-%m-%d %H:%M'),
                                    '💵 Cena sprzedaży': f"${edge['sell_price']:.2f}",
                                    '💰 Zysk': f"${edge['profit']:.2f}",
                                    '📈 Zysk %': f"{edge['profit_pct']:.2f}%",
                                    '⏱ Czas trwania': f"{edge['duration_minutes']:.0f} min"
                                }
                                for idx, edge in enumerate(trading_edges[:10])  # Pokaż top 10
                            ])
                            edges_df = clean_dataframe_for_streamlit(edges_df)
                            st.dataframe(edges_df, width='stretch', hide_index=True)
                            
                            # Statystyki
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                avg_profit = np.mean([e['profit_pct'] for e in trading_edges])
                                st.metric("📊 Średni zysk", f"{avg_profit:.2f}%")
                            with col2:
                                max_profit = max([e['profit_pct'] for e in trading_edges])
                                st.metric("🏆 Najlepszy zysk", f"{max_profit:.2f}%")
                            with col3:
                                avg_duration = np.mean([e['duration_minutes'] for e in trading_edges])
                                st.metric("⏱ Średni czas", f"{avg_duration:.0f} min")
                            with col4:
                                total_profit = sum([e['profit'] for e in trading_edges])
                                st.metric("💵 Łączny potencjał", f"${total_profit:.2f}")
                        else:
                            st.warning("⚠️ Nie znaleziono optymalnych okazji transakcyjnych w tym okresie.")
                        
                        st.divider()
                        
                        # Wykres świecowy z Edge Finder
                        st.subheader("🕯️ Wykres świecowy z Edge Finder (co minutę)")
                        fig_candlestick = plot_candlestick_intraday(intraday_data, patterns, show_edges=True)
                        if fig_candlestick:
                            st.plotly_chart(fig_candlestick, width='stretch')
                            st.caption("💡 Zielone linie z diamentami (🛒) oznaczają momenty zakupu, czerwone strzałki (💵) - momenty sprzedaży. Linie łączą pary transakcji.")
                        
                        # Rozkład godzin dołków i szczytów
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Rozkład godzin dołków (najlepsze momenty do zakupu)")
                            if patterns['low_hours_dist']:
                                fig_low = px.histogram(
                                    x=patterns['low_hours_dist'],
                                    nbins=24,
                                    labels={'x': 'Godzina', 'y': 'Liczba wystąpień'},
                                    title='Kiedy najczęściej występują dołki?'
                                )
                                st.plotly_chart(fig_low, width='stretch')
                        
                        with col2:
                            st.subheader("Rozkład godzin szczytów (najlepsze momenty do sprzedaży)")
                            if patterns['high_hours_dist']:
                                fig_high = px.histogram(
                                    x=patterns['high_hours_dist'],
                                    nbins=24,
                                    labels={'x': 'Godzina', 'y': 'Liczba wystąpień'},
                                    title='Kiedy najczęściej występują szczyty?'
                                )
                                st.plotly_chart(fig_high, width='stretch')
                        
                        # Tabela z wzorcami dziennymi
                        st.subheader("📋 Wzorce dzienne")
                        patterns_df = pd.DataFrame([
                            {
                                'Data': p['date'].strftime('%Y-%m-%d'),
                                'Otwarcie': f"${p['open']:.2f}",
                                'Zamknięcie': f"${p['close']:.2f}",
                                'Min (dołek)': f"${p['low']:.2f}",
                                'Max (szczyt)': f"${p['high']:.2f}",
                                'Czas dołku': p['low_time'].strftime('%H:%M'),
                                'Czas szczytu': p['high_time'].strftime('%H:%M'),
                                'Zmiana %': f"{p['change_pct']:.2f}%"
                            }
                            for p in patterns['daily_patterns']
                        ])
                        patterns_df = clean_dataframe_for_streamlit(patterns_df)
                        st.dataframe(patterns_df, width='stretch', hide_index=True)
                    else:
                        st.info("Brak wystarczających danych do analizy wzorców intraday")
                else:
                    st.warning("Nie można pobrać danych intraday. Dane minutowe są dostępne tylko dla ostatnich 7 dni i tylko dla niektórych spółek.")
            except Exception as e:
                st.warning(f"Błąd przy analizie intraday: {e}")
            
            # PRZEWIDYWANIA NA PRZYSZŁY TYDZIEŃ - Symulacja intraday
            st.divider()
            st.subheader("🔮 Przewidywania na przyszły tydzień - Symulacja intraday")
            st.caption("💡 Symulacja ruchu cen na podstawie modelu ML i historycznych wzorców. Pokazuje potencjalne dołki, szczyty i okazje transakcyjne.")
            
            try:
                if predictor.model is not None and predictor.data is not None and len(predictor.data) > 0:
                    with st.spinner("Symulowanie przyszłego tygodnia..."):
                        # Symuluj przyszły tydzień
                        simulated_week = simulate_future_week(predictor, days=7)
                        
                        if simulated_week is not None and len(simulated_week) > 0:
                            # Analizuj wzorce w symulacji
                            sim_patterns = analyze_intraday_patterns(simulated_week)
                            
                            if sim_patterns:
                                # Metryki dla przyszłego tygodnia
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("⏰ Średni czas dołku (kup)", sim_patterns['avg_low_time'])
                                with col2:
                                    st.metric("⏰ Średni czas szczytu (sprzedaj)", sim_patterns['avg_high_time'])
                                with col3:
                                    st.metric("📊 Symulowanych dni", len(sim_patterns['daily_patterns']))
                                with col4:
                                    avg_change = np.mean([p['change_pct'] for p in sim_patterns['daily_patterns']])
                                    st.metric("📈 Średnia zmiana dzienna", f"{avg_change:.2f}%")
                                
                                # Edge Finder dla przyszłego tygodnia
                                st.subheader("🎯 Edge Finder - Optymalne momenty transakcji (przyszły tydzień)")
                                
                                # Znajdź optymalne pary transakcji w symulacji
                                future_trading_edges = find_trading_edges(simulated_week, min_profit_pct=0.05, lookback_window=20)
                                
                                if future_trading_edges:
                                    # Pokaż najlepsze okazje w tabeli
                                    st.info(f"✅ Znaleziono {len(future_trading_edges)} potencjalnych okazji transakcyjnych na przyszły tydzień!")
                                    
                                    future_edges_df = pd.DataFrame([
                                        {
                                            'Okazja': f"#{idx+1}",
                                            '🛒 Kup (data/czas)': edge['buy_time'].strftime('%Y-%m-%d %H:%M'),
                                            '🛒 Cena zakupu': f"${edge['buy_price']:.2f}",
                                            '💵 Sprzedaj (data/czas)': edge['sell_time'].strftime('%Y-%m-%d %H:%M'),
                                            '💵 Cena sprzedaży': f"${edge['sell_price']:.2f}",
                                            '💰 Zysk': f"${edge['profit']:.2f}",
                                            '📈 Zysk %': f"{edge['profit_pct']:.2f}%",
                                            '⏱ Czas trwania': f"{edge['duration_minutes']:.0f} min"
                                        }
                                        for idx, edge in enumerate(future_trading_edges[:10])  # Pokaż top 10
                                    ])
                                    future_edges_df = clean_dataframe_for_streamlit(future_edges_df)
                                    st.dataframe(future_edges_df, width='stretch', hide_index=True)
                                    
                                    # Statystyki
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        avg_profit = np.mean([e['profit_pct'] for e in future_trading_edges])
                                        st.metric("📊 Średni zysk", f"{avg_profit:.2f}%")
                                    with col2:
                                        max_profit = max([e['profit_pct'] for e in future_trading_edges])
                                        st.metric("🏆 Najlepszy zysk", f"{max_profit:.2f}%")
                                    with col3:
                                        avg_duration = np.mean([e['duration_minutes'] for e in future_trading_edges])
                                        st.metric("⏱ Średni czas", f"{avg_duration:.0f} min")
                                    with col4:
                                        total_profit = sum([e['profit'] for e in future_trading_edges])
                                        st.metric("💵 Łączny potencjał", f"${total_profit:.2f}")
                                else:
                                    st.warning("⚠️ Nie znaleziono optymalnych okazji transakcyjnych w symulacji przyszłego tygodnia.")
                                
                                st.divider()
                                
                                # Wykres świecowy z Edge Finder dla przyszłego tygodnia
                                st.subheader("🕯️ Wykres świecowy z Edge Finder - Przyszły tydzień (symulacja)")
                                fig_future_candlestick = plot_candlestick_intraday(simulated_week, sim_patterns, show_edges=True)
                                if fig_future_candlestick:
                                    st.plotly_chart(fig_future_candlestick, width='stretch')
                                    st.caption("💡 Symulacja przyszłego tygodnia. Zielone linie z diamentami (🛒) oznaczają potencjalne momenty zakupu, czerwone strzałki (💵) - momenty sprzedaży.")
                                
                                # Tabela z wzorcami dziennymi dla przyszłego tygodnia
                                st.subheader("📋 Przewidywane wzorce dzienne (przyszły tydzień)")
                                future_patterns_df = pd.DataFrame([
                                    {
                                        'Data': p['date'].strftime('%Y-%m-%d'),
                                        'Otwarcie': f"${p['open']:.2f}",
                                        'Zamknięcie': f"${p['close']:.2f}",
                                        'Min (dołek)': f"${p['low']:.2f}",
                                        'Max (szczyt)': f"${p['high']:.2f}",
                                        'Czas dołku': p['low_time'].strftime('%H:%M'),
                                        'Czas szczytu': p['high_time'].strftime('%H:%M'),
                                        'Zmiana %': f"{p['change_pct']:.2f}%"
                                    }
                                    for p in sim_patterns['daily_patterns']
                                ])
                                future_patterns_df = clean_dataframe_for_streamlit(future_patterns_df)
                                st.dataframe(future_patterns_df, width='stretch', hide_index=True)
                            else:
                                st.info("Brak wystarczających danych w symulacji do analizy wzorców.")
                        else:
                            st.warning("Nie udało się wygenerować symulacji przyszłego tygodnia.")
                else:
                    st.warning("Brak modelu lub danych do symulacji. Najpierw wytrenuj model.")
            except Exception as e:
                st.warning(f"Błąd przy symulacji przyszłego tygodnia: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            st.divider()
            
            # Wykres cen (historyczny)
            st.subheader("Wykres cen z przewidywaniami (ostatnie 60 dni)")
            fig_price = plot_stock_price(predictor, days=60)
            if fig_price:
                st.plotly_chart(fig_price, width='stretch')
            
            # Wskaźniki techniczne
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Wskaźniki techniczne")
                fig_tech = plot_technical_indicators(predictor, days=60)
                if fig_tech:
                    st.plotly_chart(fig_tech, width='stretch')
            
            with col2:
                st.subheader("RSI")
                fig_rsi = plot_rsi(predictor, days=60)
                if fig_rsi:
                    st.plotly_chart(fig_rsi, width='stretch')
            
            # Informacje o modelu
            st.divider()
            st.subheader("Informacje o modelu")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Punkty danych", selected_data['data_points'])
            with col2:
                st.metric("Dokładność", f"{selected_data.get('test_accuracy', 0):.2%}" if selected_data.get('test_accuracy', 0) > 0 else "N/A")
            with col3:
                st.metric("Precision", f"{selected_data.get('test_precision', 0):.4f}" if selected_data.get('test_precision', 0) > 0 else "N/A")
            with col4:
                st.metric("F1-Score", f"{selected_data.get('test_f1', 0):.4f}" if selected_data.get('test_f1', 0) > 0 else "N/A")
            
            if selected_data['last_trained']:
                st.caption(f"Ostatnie trenowanie: {selected_data['last_trained']}")
            
            # POPRAWKA DEV: Wyniki out-of-sample i baseline'y
            try:
                # Załaduj pełne informacje o modelu
                predictor = StockPredictor(current_selected_symbol, data_dir='stock_data')
                if predictor.model_file.exists():
                    with open(predictor.model_file, 'r') as f:
                        full_model_info = json.load(f)
                    
                    oos_results = full_model_info.get('out_of_sample')
                    baseline_results = full_model_info.get('baselines')
                    optimal_threshold = full_model_info.get('optimal_threshold', 0.65)
                    
                    if oos_results:
                        st.divider()
                        st.subheader("📊 Test Out-of-Sample (ostatnie 6 miesięcy)")
                        st.caption("💡 Prawdziwy test na danych, których model nie widział podczas trenowania")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy (OOS)", f"{oos_results.get('accuracy', 0):.2%}")
                        with col2:
                            st.metric("F1-Score (OOS)", f"{oos_results.get('f1', 0):.4f}")
                        with col3:
                            st.metric("Total Return", f"{oos_results.get('total_return', 0):.2%}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{oos_results.get('sharpe', 0):.4f}")
                        
                        st.caption(f"Liczba transakcji: {oos_results.get('num_trades', 0)} | Optymalny próg: {optimal_threshold:.2f}")
                    
                    if baseline_results:
                        st.divider()
                        st.subheader("📈 Porównanie z Baseline'ami")
                        st.caption("💡 Sprawdź czy model faktycznie przebija proste strategie")
                        
                        baseline_df = pd.DataFrame([
                            {
                                'Strategia': 'Model ML',
                                'Accuracy': oos_results.get('accuracy', 0) if oos_results else 0,
                                'Total Return': oos_results.get('total_return', 0) if oos_results else 0,
                                'Liczba transakcji': oos_results.get('num_trades', 0) if oos_results else 0
                            },
                            {
                                'Strategia': 'Always UP (Buy & Hold)',
                                'Accuracy': baseline_results.get('always_up', {}).get('accuracy', 0),
                                'Total Return': baseline_results.get('always_up', {}).get('total_return', 0),
                                'Liczba transakcji': baseline_results.get('always_up', {}).get('num_trades', 0)
                            },
                            {
                                'Strategia': 'MA Crossover (20/50)',
                                'Accuracy': baseline_results.get('ma_crossover', {}).get('accuracy', 0),
                                'Total Return': baseline_results.get('ma_crossover', {}).get('total_return', 0),
                                'Liczba transakcji': baseline_results.get('ma_crossover', {}).get('num_trades', 0)
                            },
                            {
                                'Strategia': 'Random',
                                'Accuracy': baseline_results.get('random', {}).get('accuracy', 0),
                                'Total Return': baseline_results.get('random', {}).get('total_return', 0),
                                'Liczba transakcji': baseline_results.get('random', {}).get('num_trades', 0)
                            }
                        ])
                        
                        baseline_df['Accuracy'] = baseline_df['Accuracy'].apply(lambda x: f"{x:.2%}")
                        baseline_df['Total Return'] = baseline_df['Total Return'].apply(lambda x: f"{x:.2%}")
                        baseline_df = clean_dataframe_for_streamlit(baseline_df)
                        st.dataframe(baseline_df, width='stretch', hide_index=True)
                        
                        # Wykres porównawczy
                        if oos_results:
                            fig_baseline = go.Figure()
                            strategies = ['Model ML', 'Always UP', 'MA Crossover', 'Random']
                            returns = [
                                oos_results.get('total_return', 0),
                                baseline_results.get('always_up', {}).get('total_return', 0),
                                baseline_results.get('ma_crossover', {}).get('total_return', 0),
                                baseline_results.get('random', {}).get('total_return', 0)
                            ]
                            
                            colors = ['green' if r > 0 else 'red' for r in returns]
                            fig_baseline.add_trace(go.Bar(
                                x=strategies,
                                y=returns,
                                marker_color=colors,
                                text=[f"{r:.2%}" for r in returns],
                                textposition='outside'
                            ))
                            
                            fig_baseline.update_layout(
                                title='Porównanie Total Return - Model vs Baseline\'y',
                                xaxis_title='Strategia',
                                yaxis_title='Total Return (%)',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_baseline, width='stretch')
            except Exception as e:
                st.caption(f"Brak wyników out-of-sample (może wymagać retrenowania modelu): {e}")
    
    with tab3:
        st.header("Wszystkie spółki - Szczegółowa tabela")
        
        # Rozszerzona tabela
        df_detailed = pd.DataFrame([
            {
                'Symbol': p['symbol'],
                'Nazwa spółki': get_stock_name(p['symbol']),
                'Kierunek': p.get('predicted_direction', 'N/A'),
                'Prawdopod.': p.get('direction_probability', 0),
                'Aktualna cena': p['current_price'],
                'Przewidywana cena': p['predicted_price'],
                'Zmiana ($)': p['change'],
                'Zmiana (%)': p['change_percent'],
                'Dokładność': p.get('test_accuracy', 0) if p.get('test_accuracy', 0) > 0 else None,
                'F1-Score': p.get('test_f1', 0) if p.get('test_f1', 0) > 0 else None,
                'Punkty danych': p['data_points']
            }
            for p in predictions_data
        ])
        
        # Formatowanie
        df_detailed['Aktualna cena'] = df_detailed['Aktualna cena'].apply(lambda x: f"${x:.2f}")
        df_detailed['Przewidywana cena'] = df_detailed['Przewidywana cena'].apply(lambda x: f"${x:.2f}")
        df_detailed['Zmiana ($)'] = df_detailed['Zmiana ($)'].apply(lambda x: f"${x:.2f}")
        df_detailed['Zmiana (%)'] = df_detailed['Zmiana (%)'].apply(lambda x: f"{x:.2f}%")
        df_detailed['Prawdopod.'] = df_detailed['Prawdopod.'].apply(lambda x: f"{x*100:.1f}%")
        df_detailed['Dokładność'] = df_detailed['Dokładność'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        df_detailed['F1-Score'] = df_detailed['F1-Score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        # Sortowanie
        sort_by = st.selectbox("Sortuj według:", 
                               ['Zmiana (%)', 'Symbol', 'Kierunek', 'Dokładność', 'Aktualna cena'],
                               index=0)
        
        if sort_by == 'Zmiana (%)':
            df_detailed = df_detailed.sort_values('Zmiana (%)', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
        elif sort_by == 'Symbol':
            df_detailed = df_detailed.sort_values('Symbol')
        elif sort_by == 'Kierunek':
            df_detailed = df_detailed.sort_values('Kierunek')
        elif sort_by == 'Dokładność':
            df_detailed = df_detailed.sort_values('Dokładność', ascending=False)
        elif sort_by == 'Aktualna cena':
            df_detailed = df_detailed.sort_values('Aktualna cena', ascending=False)
        
        df_detailed = clean_dataframe_for_streamlit(df_detailed)
        
        # Wyświetl tabelę z kolumną przycisków
        num_cols = len(df_detailed.columns)
        header_cols = st.columns([3] * num_cols + [1])
        for idx, col_name in enumerate(df_detailed.columns):
            with header_cols[idx]:
                st.markdown(f"**{col_name}**")
        with header_cols[-1]:
            st.markdown("**Szczegóły**")
        
        st.divider()
        
        # Wyświetl każdy wiersz z przyciskiem
        for idx, (_, row) in enumerate(df_detailed.iterrows()):
            row_cols = st.columns([3] * num_cols + [1])
            for col_idx, col_name in enumerate(df_detailed.columns):
                with row_cols[col_idx]:
                    st.write(row[col_name])
            with row_cols[-1]:
                if st.button("📈", key=f"details_all_{predictions_data[idx]['symbol']}", help="Przejdź do szczegółów"):
                    navigate_to_details(predictions_data[idx]['symbol'])
            if idx < len(df_detailed) - 1:
                st.divider()
        
        # Eksport
        csv = df_detailed.to_csv(index=False)
        st.download_button(
            label="📥 Pobierz jako CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.header("Statystyki modeli klasyfikacji")
        
        # Statystyki metryk klasyfikacji
        accuracy_values = [p['test_accuracy'] for p in predictions_data if p.get('test_accuracy', 0) > 0]
        f1_values = [p['test_f1'] for p in predictions_data if p.get('test_f1', 0) > 0]
        precision_values = [p['test_precision'] for p in predictions_data if p.get('test_precision', 0) > 0]
        recall_values = [p['test_recall'] for p in predictions_data if p.get('test_recall', 0) > 0]
        
        if accuracy_values:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Rozkład dokładności (Accuracy)")
                fig_acc = px.histogram(
                    x=accuracy_values,
                    nbins=20,
                    labels={'x': 'Dokładność', 'y': 'Liczba spółek'},
                    title='Histogram dokładności modeli'
                )
                st.plotly_chart(fig_acc, width='stretch')
            
            with col2:
                st.subheader("Rozkład F1-Score")
                fig_f1 = px.histogram(
                    x=f1_values,
                    nbins=20,
                    labels={'x': 'F1-Score', 'y': 'Liczba spółek'},
                    title='Histogram F1-Score'
                )
                st.plotly_chart(fig_f1, width='stretch')
            
            # Statystyki opisowe
            st.subheader("Statystyki opisowe metryk klasyfikacji")
            stats_df = pd.DataFrame({
                'Metryka': ['Średnia', 'Mediana', 'Min', 'Max', 'Odch. std.'],
                'Accuracy': [
                    np.mean(accuracy_values),
                    np.median(accuracy_values),
                    np.min(accuracy_values),
                    np.max(accuracy_values),
                    np.std(accuracy_values)
                ],
                'F1-Score': [
                    np.mean(f1_values),
                    np.median(f1_values),
                    np.min(f1_values),
                    np.max(f1_values),
                    np.std(f1_values)
                ],
                'Precision': [
                    np.mean(precision_values),
                    np.median(precision_values),
                    np.min(precision_values),
                    np.max(precision_values),
                    np.std(precision_values)
                ],
                'Recall': [
                    np.mean(recall_values),
                    np.median(recall_values),
                    np.min(recall_values),
                    np.max(recall_values),
                    np.std(recall_values)
                ]
            })
            
            stats_df['Accuracy'] = stats_df['Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
            stats_df['F1-Score'] = stats_df['F1-Score'].apply(lambda x: f"{x:.4f}")
            stats_df['Precision'] = stats_df['Precision'].apply(lambda x: f"{x:.4f}")
            stats_df['Recall'] = stats_df['Recall'].apply(lambda x: f"{x:.4f}")
            
            stats_df = clean_dataframe_for_streamlit(stats_df)
            st.dataframe(stats_df, width='stretch', hide_index=True)
        else:
            st.info("Brak danych statystycznych. Uruchom najpierw trenowanie modeli.")
    
    with tab5:
        st.header("🇩🇪 Indeks DAX - Analiza")
        
        if st.session_state.get('dax_fetch_message'):
            st.success(st.session_state.pop('dax_fetch_message'))
        if st.session_state.get('dax_fetch_error'):
            st.error(st.session_state.pop('dax_fetch_error'))
        
            # Załaduj dane DAX tylko jeśli użytkownik kliknął przycisk
        if not st.session_state.dax_main_data_loaded:
            if st.button("📊 Załaduj dane DAX", key="load_dax_main"):
                st.session_state.dax_main_data_loaded = True
                st.rerun()
            dax_info = None
        else:
            with st.spinner("Pobieranie danych DAX..."):
                dax_info = get_dax_data()
        
        if dax_info is None:
            st.warning("⚠️ Brak zapisanego modelu dla indeksu DAX. Uruchom '🚀 Uruchom Daily Update' w zakładce Dashboard aby wytrenować model.")
        else:
            predictor = dax_info['predictor']
            prediction = dax_info['prediction']
            model_info = dax_info['model_info']
            dax_data = dax_info['data']
            
            # Metryki DAX
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = dax_data['Close'].iloc[-1]
            prev_close = dax_data['Close'].iloc[-2] if len(dax_data) > 1 else current_price
            daily_change = current_price - prev_close
            daily_change_pct = (daily_change / prev_close) * 100 if prev_close > 0 else 0
            
            with col1:
                st.metric("Aktualna wartość DAX", f"{current_price:,.0f}", 
                         f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)")
            with col2:
                direction_icon = "↑" if prediction.get('predicted_direction') == 'UP' else "↓"
                st.metric("Przewidywany kierunek", f"{direction_icon} {prediction.get('predicted_direction', 'N/A')}")
            with col3:
                prob = prediction.get('direction_probability', 0) * 100
                st.metric("Prawdopodobieństwo", f"{prob:.1f}%")
            with col4:
                accuracy = model_info.get('test_accuracy', 0)
                st.metric("Dokładność modelu", f"{accuracy:.2%}" if accuracy > 0 else "N/A")
            
            st.divider()
            
            # Wykres DAX
            st.subheader("Wykres indeksu DAX")
            fig_dax = go.Figure()
            
            # Cena
            fig_dax.add_trace(go.Scatter(
                x=dax_data.index[-252:],  # Ostatni rok
                y=dax_data['Close'].iloc[-252:],
                mode='lines',
                name='DAX',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))
            
            # Średnie kroczące
            ma_20 = dax_data['Close'].rolling(20).mean()
            ma_50 = dax_data['Close'].rolling(50).mean()
            
            fig_dax.add_trace(go.Scatter(
                x=dax_data.index[-252:],
                y=ma_20.iloc[-252:],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', dash='dash', width=1)
            ))
            
            if len(dax_data) > 50:
                fig_dax.add_trace(go.Scatter(
                    x=dax_data.index[-252:],
                    y=ma_50.iloc[-252:],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='red', dash='dash', width=1)
                ))
            
            fig_dax.update_layout(
                title='Indeks DAX - Ostatni rok',
                xaxis_title='Data',
                yaxis_title='Wartość indeksu',
                height=500,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_dax, width='stretch')
            
            # Wykres cen z przewidywaniami
            st.subheader("Przewidywania kierunku")
            fig_price = plot_stock_price(predictor, days=60)
            if fig_price:
                st.plotly_chart(fig_price, width='stretch')
            
            # Statystyki DAX
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statystyki DAX (ostatni rok)")
                stats_data = {
                    'Min': dax_data['Close'].iloc[-252:].min(),
                    'Max': dax_data['Close'].iloc[-252:].max(),
                    'Średnia': dax_data['Close'].iloc[-252:].mean(),
                    'Odch. std.': dax_data['Close'].iloc[-252:].std(),
                    'Zmiana roczna': ((dax_data['Close'].iloc[-1] - dax_data['Close'].iloc[-252]) / dax_data['Close'].iloc[-252] * 100) if len(dax_data) > 252 else 0
                }
                
                stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metryka', 'Wartość'])
                stats_df['Wartość'] = stats_df['Wartość'].apply(lambda x: f"{x:,.2f}" if abs(x) > 1 else f"{x:.4f}")
                stats_df = clean_dataframe_for_streamlit(stats_df)
                st.dataframe(stats_df, width='stretch', hide_index=True)
            
            with col2:
                st.subheader("Informacje o modelu")
                model_stats = {
                    'Typ modelu': 'Klasyfikacja kierunku',
                    'Dokładność (test)': f"{model_info.get('test_accuracy', 0):.2%}" if model_info.get('test_accuracy', 0) > 0 else "N/A",
                    'Precision': f"{model_info.get('test_precision', 0):.4f}" if model_info.get('test_precision', 0) > 0 else "N/A",
                    'Recall': f"{model_info.get('test_recall', 0):.4f}" if model_info.get('test_recall', 0) > 0 else "N/A",
                    'F1-Score': f"{model_info.get('test_f1', 0):.4f}" if model_info.get('test_f1', 0) > 0 else "N/A",
                    'Punkty danych': model_info.get('data_points', len(dax_data)),
                    'Liczba cech': model_info.get('feature_count', 0)
                }
                
                model_df = pd.DataFrame(list(model_stats.items()), columns=['Metryka', 'Wartość'])
                model_df = clean_dataframe_for_streamlit(model_df)
                st.dataframe(model_df, width='stretch', hide_index=True)
            
            # Główne składniki DAX
            st.divider()
            st.subheader("Główne składniki DAX")
            
            dax_components = get_dax_components()
            
            # Pobierz dane dla WSZYSTKICH spółek z bazy (dla składników DAX) - tylko jeśli użytkownik kliknął przycisk
            if not st.session_state.dax_data_loaded:
                if st.button("📊 Załaduj dane dla składników DAX", key="load_dax_components"):
                    st.session_state.dax_data_loaded = True
                    st.rerun()
                all_predictions_for_dax = []
            else:
                with st.spinner("Pobieranie danych dla składników DAX..."):
                    all_predictions_for_dax = get_dax_predictions(include_all_from_db=True)
            
            # Sprawdź które składniki mamy w danych
            available_components = [comp for comp in dax_components if any(p['symbol'] == comp for p in all_predictions_for_dax)]
            
            if available_components:
                component_data = [p for p in all_predictions_for_dax if p['symbol'] in available_components]
                
                df_components = pd.DataFrame([
                    {
                        'Symbol': p['symbol'],
                        'Nazwa spółki': get_stock_name(p['symbol']),
                        'Kierunek': p.get('predicted_direction', 'N/A'),
                        'Prawdopod.': f"{p.get('direction_probability', 0)*100:.1f}%",
                        'Aktualna cena': f"${p['current_price']:.2f}",
                        'Zmiana %': f"{p['change_percent']:.2f}%"
                    }
                    for p in component_data
                ])
                
                df_components = clean_dataframe_for_streamlit(df_components)
                
                # Wyświetl tabelę z kolumną przycisków
                num_cols = len(df_components.columns)
                header_cols = st.columns([3] * num_cols + [1])
                for idx, col_name in enumerate(df_components.columns):
                    with header_cols[idx]:
                        st.markdown(f"**{col_name}**")
                with header_cols[-1]:
                    st.markdown("**Szczegóły**")
                
                st.divider()
                
                # Wyświetl każdy wiersz z przyciskiem
                for idx, (_, row) in enumerate(df_components.iterrows()):
                    row_cols = st.columns([3] * num_cols + [1])
                    for col_idx, col_name in enumerate(df_components.columns):
                        with row_cols[col_idx]:
                            st.write(row[col_name])
                    with row_cols[-1]:
                        if st.button("📈", key=f"details_component_{component_data[idx]['symbol']}", help="Przejdź do szczegółów"):
                            navigate_to_details(component_data[idx]['symbol'])
                    if idx < len(df_components) - 1:
                        st.divider()
                
                # Wykres składników
                fig_components = go.Figure()
                
                colors_comp = ['green' if p['change_percent'] > 0 else 'red' for p in component_data]
                symbols_comp = [p['symbol'] for p in component_data]
                changes_comp = [p['change_percent'] for p in component_data]
                
                fig_components.add_trace(go.Bar(
                    x=symbols_comp,
                    y=changes_comp,
                    marker_color=colors_comp,
                    text=[f"{c:.2f}%" for c in changes_comp],
                    textposition='outside'
                ))
                
                fig_components.update_layout(
                    title='Przewidywane zmiany głównych składników DAX (%)',
                    xaxis_title='Symbol',
                    yaxis_title='Zmiana (%)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_components, width='stretch')
            else:
                st.info("Brak danych dla składników DAX. Uruchom najpierw daily_update.py")
            
            # Lista wszystkich spółek niemieckich
            st.divider()
            st.subheader("📋 Wszystkie spółki niemieckie")
            
            # Pobierz listę spółek
            german_stocks = get_all_german_stocks()
            
            # Filtry
            col1, col2 = st.columns(2)
            
            with col1:
                filter_type = st.selectbox(
                    "Filtruj według indeksu:",
                    ['Wszystkie', 'DAX', 'MDAX', 'SDAX', 'TecDAX', 'Ulubione'],
                    index=0
                )
            
            with col2:
                search_term = st.text_input("🔍 Szukaj spółki:", "")
            
            # Wybierz listę do wyświetlenia
            if filter_type == 'Ulubione':
                stocks_to_show = st.session_state.favorites
            elif filter_type == 'DAX':
                stocks_to_show = german_stocks['dax']
            elif filter_type == 'MDAX':
                stocks_to_show = german_stocks['mdax']
            elif filter_type == 'SDAX':
                stocks_to_show = german_stocks['sdax']
            elif filter_type == 'TecDAX':
                stocks_to_show = german_stocks['tecdax']
            else:
                stocks_to_show = german_stocks['all']
            
            # Filtruj według wyszukiwania
            if search_term:
                stocks_to_show = [s for s in stocks_to_show if search_term.upper() in s.upper()]
            
            # Pobierz dane dla WSZYSTKICH spółek z bazy (dla zakładki DAX) - tylko jeśli dane są już załadowane
            if st.session_state.get('dax_data_loaded', False):
                with st.spinner("Pobieranie danych dla wszystkich spółek z bazy..."):
                    all_predictions_data = get_dax_predictions(include_all_from_db=True)
            else:
                all_predictions_data = []
            
            # Sprawdź które mają dane
            available_stocks = [s for s in stocks_to_show if any(p['symbol'] == s for p in all_predictions_data)]
            unavailable_stocks = [s for s in stocks_to_show if s not in available_stocks]
            
            st.info(f"Znaleziono {len(available_stocks)} spółek z danymi z {len(stocks_to_show)} wyświetlonych")
            
            # Tabela z możliwością dodawania do ulubionych
            if available_stocks:
                st.subheader(f"Spółki z danymi ({len(available_stocks)})")
                
                # Przygotuj dane
                stocks_data = []
                for symbol in available_stocks:
                    stock_pred = next((p for p in all_predictions_data if p['symbol'] == symbol), None)
                    if stock_pred:
                        is_favorite = symbol in st.session_state.favorites
                        stocks_data.append({
                            'Symbol': symbol,
                            'Nazwa spółki': get_stock_name(symbol),
                            'Ulubione': '⭐' if is_favorite else '☆',
                            'Kierunek': stock_pred.get('predicted_direction', 'N/A'),
                            'Prawdopod.': f"{stock_pred.get('direction_probability', 0)*100:.1f}%",
                            'Aktualna cena': f"${stock_pred['current_price']:.2f}",
                            'Zmiana %': f"{stock_pred['change_percent']:.2f}%",
                            'Dokładność': f"{stock_pred.get('test_accuracy', 0):.2%}" if stock_pred.get('test_accuracy', 0) > 0 else "N/A"
                        })
                
                if stocks_data:
                    df_stocks = pd.DataFrame(stocks_data)
                    df_display = clean_dataframe_for_streamlit(df_stocks.drop(columns=['Ulubione']))
                    num_cols = len(df_display.columns)
                    
                    header_cols = st.columns([3] * num_cols + [1, 1])
                    for idx, col_name in enumerate(df_display.columns):
                        with header_cols[idx]:
                            st.markdown(f"**{col_name}**")
                    with header_cols[-2]:
                        st.markdown("**Ulubione**")
                    with header_cols[-1]:
                        st.markdown("**Szczegóły**")
                    
                    st.divider()
                    
                    for idx, (_, row) in enumerate(df_display.iterrows()):
                        row_cols = st.columns([3] * num_cols + [1, 1])
                        for col_idx, col_name in enumerate(df_display.columns):
                            with row_cols[col_idx]:
                                st.write(row[col_name])
                        
                        symbol = df_stocks.iloc[idx]['Symbol']
                        is_fav = symbol in st.session_state.favorites
                        fav_label = "⭐" if is_fav else "☆"
                        with row_cols[-2]:
                            if st.button(fav_label, key=f"dax_fav_toggle_{symbol}", help="Dodaj/usuń z ulubionych"):
                                toggle_favorite(symbol)
                                st.rerun()
                        
                        with row_cols[-1]:
                            if st.button("📈", key=f"details_german_{symbol}", help="Przejdź do szczegółów"):
                                navigate_to_details(symbol)
                        
                        if idx < len(df_display) - 1:
                            st.divider()
            
            # Pokaż spółki bez danych (jeśli są)
            if unavailable_stocks and filter_type != 'Ulubione':
                with st.expander(f"Spółki bez danych ({len(unavailable_stocks)})", expanded=False):
                    # Podziel na kolumny dla lepszego wyświetlenia
                    cols_per_row = 5
                    for i in range(0, len(unavailable_stocks), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(unavailable_stocks):
                                symbol = unavailable_stocks[i + j]
                                is_fav = symbol in st.session_state.favorites
                                fav_icon = "⭐" if is_fav else "☆"
                                with col:
                                    st.markdown(f"**{symbol}**")
                                    action_cols = st.columns(2)
                                    with action_cols[0]:
                                        if st.button(fav_icon, key=f"fav_na_{symbol}", help="Dodaj/usuń z ulubionych"):
                                            toggle_favorite(symbol)
                                            st.rerun()
                                    with action_cols[1]:
                                        if st.button("📥", key=f"dax_fetch_{symbol}", help="Pobierz dane z internetu"):
                                            with st.spinner(f"Pobieranie danych dla {symbol}..."):
                                                success, info_msg = run_single_symbol_update(symbol, retrain_weekly=True)
                                            if success:
                                                message = f"✓ Pobrano dane dla {symbol}."
                                                if info_msg:
                                                    message += f" {info_msg}"
                                                st.session_state['dax_fetch_message'] = message
                                            else:
                                                st.session_state['dax_fetch_error'] = f"Nie udało się pobrać danych dla {symbol}: {info_msg}"
                                            st.rerun()
                                    st.caption("Dodaj do ulubionych lub pobierz dane.")
            
            # Statystyki ulubionych
            if st.session_state.favorites:
                st.divider()
                st.subheader(f"⭐ Ulubione ({len(st.session_state.favorites)})")
                
                favorite_predictions = [p for p in all_predictions_data if p['symbol'] in st.session_state.favorites]
                
                if favorite_predictions:
                    df_fav = pd.DataFrame([
                        {
                            'Symbol': p['symbol'],
                            'Nazwa spółki': get_stock_name(p['symbol']),
                            'Kierunek': p.get('predicted_direction', 'N/A'),
                            'Prawdopod.': f"{p.get('direction_probability', 0)*100:.1f}%",
                            'Aktualna cena': f"${p['current_price']:.2f}",
                            'Zmiana %': f"{p['change_percent']:.2f}%"
                        }
                        for p in favorite_predictions
                    ])
                    
                    df_fav = clean_dataframe_for_streamlit(df_fav)
                    
                    # Wyświetl tabelę z kolumną przycisków
                    num_cols = len(df_fav.columns)
                    header_cols = st.columns([3] * num_cols + [1])
                    for idx, col_name in enumerate(df_fav.columns):
                        with header_cols[idx]:
                            st.markdown(f"**{col_name}**")
                    with header_cols[-1]:
                        st.markdown("**Szczegóły**")
                    
                    st.divider()
                    
                    # Wyświetl każdy wiersz z przyciskiem
                    for idx, (_, row) in enumerate(df_fav.iterrows()):
                        row_cols = st.columns([3] * num_cols + [1])
                        for col_idx, col_name in enumerate(df_fav.columns):
                            with row_cols[col_idx]:
                                st.write(row[col_name])
                        with row_cols[-1]:
                            if st.button("📈", key=f"details_dax_fav_{favorite_predictions[idx]['symbol']}", help="Przejdź do szczegółów"):
                                navigate_to_details(favorite_predictions[idx]['symbol'])
                        if idx < len(df_fav) - 1:
                            st.divider()
                else:
                    st.info("Brak danych dla ulubionych spółek. Uruchom najpierw daily_update.py")
                
                # Przycisk do usunięcia wszystkich ulubionych
                if st.button("🗑️ Usuń wszystkie ulubione"):
                    st.session_state.favorites = []
                    save_favorites()
                    st.rerun()
    
    with tab6:
        st.header("📊 Skuteczność predykcji - Analiza logu")
        st.caption("💡 Analiza historycznej skuteczności modelu na podstawie zapisanych predykcji")
        
        if not PREDICTION_LOGGING_AVAILABLE:
            st.warning("⚠️ Moduł logowania predykcji nie jest dostępny. Upewnij się, że pliki prediction_logger.py i calibration.py istnieją.")
        else:
            # Utwórz logger
            logger = PredictionLogger(data_dir='stock_data', use_sqlite=True)
            
            # Przycisk do aktualizacji wyników
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Aktualizuj wyniki predykcji"):
                    with st.spinner("Aktualizowanie wyników..."):
                        update_prediction_outcomes(logger, max_days_lookback=60)
                    st.success("✓ Wyniki zaktualizowane!")
                    st.rerun()
            
            with col2:
                if st.button("🎯 Trenuj kalibrator i znajdź optymalny próg"):
                    with st.spinner("Trenowanie kalibratora i strojenie progu..."):
                        # Trenuj kalibrator
                        current_filter_symbol = st.session_state.get('pred_stats_symbol', 'Wszystkie')
                        selected_symbol = None if current_filter_symbol == 'Wszystkie' else current_filter_symbol
                        calibrator_path = Path('stock_data') / f'{selected_symbol or "global"}_calibrator.pkl'
                        calibrator = train_calibrator(
                            logger,
                            symbol=selected_symbol,
                            horizon_days=filter_horizon,
                            save_path=calibrator_path
                        )
                        
                        # Znajdź optymalny próg
                        threshold_path = Path('stock_data') / f'{selected_symbol or "global"}_threshold.json'
                        threshold_result = find_best_threshold(
                            logger,
                            symbol=selected_symbol,
                            horizon_days=filter_horizon,
                            save_path=threshold_path
                        )
                        
                        if threshold_result:
                            st.success(f"✓ Optymalny próg: {threshold_result['best_threshold']:.2f}")
            
            st.divider()
            
            # Filtry do analizy
            col1, col2, col3 = st.columns(3)
            symbol_filter_options = ['Wszystkie']
            if symbols:
                symbol_filter_options.extend(symbols)
            filter_symbol = st.selectbox("Symbol:", symbol_filter_options, key='pred_stats_symbol')
            if not filter_symbol:
                filter_symbol = 'Wszystkie'
            with col2:
                filter_horizon = st.selectbox("Horyzont (dni):", [None, 1, 5, 30, 180], key='pred_stats_horizon')
            with col3:
                filter_filled = st.selectbox("Status:", ['Wszystkie', 'Z wynikami', 'Bez wyników'], key='pred_stats_filled')
            
            # Pobierz statystyki
            selected_symbol = None if filter_symbol == 'Wszystkie' else filter_symbol
            
            stats = logger.get_statistics(
                symbol=selected_symbol,
                horizon_days=filter_horizon
            )
            
            # Wyświetl statystyki
            if stats['total_predictions'] > 0:
                st.subheader("📈 Statystyki skuteczności")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Liczba predykcji", stats['total_predictions'])
                with col2:
                    st.metric("Hit Rate", f"{stats['hit_rate']:.2%}")
                with col3:
                    st.metric("Średni zwrot", f"{stats['avg_return']:.4f}")
                with col4:
                    st.metric("Liczba transakcji", stats['num_trades'])
                
                # Pobierz szczegółowe dane do wykresów
                df_calibration = logger.get_predictions_for_calibration(
                    symbol=selected_symbol,
                    horizon_days=filter_horizon,
                    min_predictions=1
                )
                
                if len(df_calibration) > 0:
                    st.divider()
                    st.subheader("📊 Wykresy analityczne")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rozkład prawdopodobieństw vs rzeczywiste wyniki
                        st.subheader("Kalibracja prawdopodobieństw")
                        fig_cal = go.Figure()
                        
                        # Grupuj prawdopodobieństwa w przedziały
                        bins = np.arange(0, 1.1, 0.1)
                        df_calibration['proba_bin'] = pd.cut(df_calibration['proba_up'], bins=bins, labels=bins[:-1])
                        
                        # Oblicz średni rzeczywisty zwrot dla każdego przedziału
                        calibration_data = df_calibration.groupby('proba_bin').agg({
                            'return_realized': 'mean',
                            'was_correct': 'mean'
                        }).reset_index()
                        
                        fig_cal.add_trace(go.Scatter(
                            x=calibration_data['proba_bin'],
                            y=calibration_data['was_correct'],
                            mode='lines+markers',
                            name='Rzeczywista trafność',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Linia idealnej kalibracji (y=x)
                        fig_cal.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode='lines',
                            name='Idealna kalibracja',
                            line=dict(color='red', dash='dash', width=1)
                        ))
                        
                        fig_cal.update_layout(
                            title='Kalibracja prawdopodobieństw',
                            xaxis_title='Przewidywane prawdopodobieństwo',
                            yaxis_title='Rzeczywista trafność',
                            height=400
                        )
                        st.plotly_chart(fig_cal, width='stretch')
                    
                    with col2:
                        # Rozkład zwrotów
                        st.subheader("Rozkład zwrotów")
                        fig_returns = px.histogram(
                            df_calibration,
                            x='return_realized',
                            nbins=50,
                            labels={'return_realized': 'Zwrot (log)', 'count': 'Liczba'},
                            title='Rozkład zrealizowanych zwrotów'
                        )
                        st.plotly_chart(fig_returns, width='stretch')
                    
                    # Wykres P&L w czasie
                    st.subheader("💰 P&L w czasie")
                    df_calibration_sorted = df_calibration.sort_values('timestamp_prediction')
                    df_calibration_sorted['cumulative_return'] = df_calibration_sorted['return_realized'].cumsum()
                    
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        x=pd.to_datetime(df_calibration_sorted['timestamp_prediction']),
                        y=df_calibration_sorted['cumulative_return'],
                        mode='lines',
                        name='Skumulowany zwrot',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig_pnl.update_layout(
                        title='Skumulowany zwrot w czasie',
                        xaxis_title='Data predykcji',
                        yaxis_title='Skumulowany zwrot (log)',
                        height=400
                    )
                    st.plotly_chart(fig_pnl, width='stretch')
                    
                    # Tabela szczegółowa
                    st.subheader("📋 Szczegółowa tabela predykcji")
                    display_df = df_calibration[['timestamp_prediction', 'symbol', 'horizon_days', 
                                                  'price_at_prediction', 'proba_up', 'decision',
                                                  'price_at_outcome', 'return_realized', 'was_correct']].copy()
                    display_df['timestamp_prediction'] = pd.to_datetime(display_df['timestamp_prediction'])
                    display_df = display_df.sort_values('timestamp_prediction', ascending=False)
                    display_df = clean_dataframe_for_streamlit(display_df)
                    st.dataframe(display_df.head(100), width='stretch', hide_index=True)
                else:
                    st.info("Brak danych do wyświetlenia. Uruchom najpierw aktualizację wyników.")
            else:
                st.info("Brak zapisanych predykcji. Predykcje są automatycznie logowane podczas przewidywań.")

if __name__ == "__main__":
    main()

