# Przewidywanie Cen Akcji - Machine Learning z Codziennym Uczeniem

System do przewidywania cen akcji na kolejny dzień używając uczenia maszynowego. System automatycznie wczytuje spółki z pliku Excel (`raport.xlsx`) i uczy się codziennie na nowych danych.

## Funkcje

- ✅ **Automatyczne wczytywanie spółek** z pliku Excel (`raport.xlsx`)
- ✅ **Codzienne zbieranie danych** - system zapisuje dane historyczne i pobiera tylko nowe
- ✅ **Codzienne uczenie** - model trenuje się codziennie na aktualnych danych
- ✅ **Przechowywanie danych** - wszystkie dane są zapisywane lokalnie
- ✅ **Wielospółkowe przewidywania** - obsługa wielu spółek jednocześnie
- ✅ **Automatyczne raporty** - generowanie raportów z przewidywań
- ✅ **Interfejs webowy (UI)** - interaktywny dashboard Streamlit do wizualizacji danych

## Instalacja

```bash
pip install -r requirements.txt
```

## Struktura pliku Excel

System oczekuje pliku `raport.xlsx` z kolumną `Symbol` zawierającą symbole spółek (np. `AAPL`, `NVDA.US`, `ENR.DE`).

## Użycie

### Interfejs webowy (UI) - Streamlit Dashboard

Najprostszy sposób na wizualizację danych:

```bash
streamlit run app.py
```

Dashboard zawiera:
- 📊 **Dashboard** - przegląd wszystkich spółek z przewidywaniami
- 📈 **Szczegóły spółki** - szczegółowa analiza wybranej spółki z wykresami
- 📋 **Wszystkie spółki** - rozszerzona tabela z możliwością sortowania
- 📉 **Statystyki** - analiza dokładności modeli

Funkcje UI:
- Interaktywne wykresy (Plotly)
- Wykresy cen z przewidywaniami
- Wskaźniki techniczne (RSI, MACD, średnie kroczące)
- Metryki i statystyki modeli
- Eksport danych do CSV
- Automatyczne odświeżanie danych

### Podstawowe użycie (codzienne uczenie):

```bash
python stock_predictor.py
```

Program automatycznie:
1. Wczyta symbole z pliku `raport.xlsx`
2. Pobierze najnowsze dane dla każdej spółki
3. Zaktualizuje zapisane dane historyczne
4. Wytrenuje modele dla każdej spółki
5. Wygeneruje przewidywania na kolejny dzień
6. Utworzy raport w folderze `reports/`

### Użycie programistyczne:

```python
from stock_predictor import MultiStockPredictor

# Utworzenie multi-stock predictor
multi_predictor = MultiStockPredictor(excel_file='raport.xlsx')

# Wczytaj symbole z pliku Excel
multi_predictor.load_symbols_from_excel()

# Codzienna aktualizacja i uczenie
results = multi_predictor.daily_update()

# Generuj raport
multi_predictor.generate_report(results)
```

### Pojedyncza spółka:

```python
from stock_predictor import StockPredictor

# Utworzenie predyktora
predictor = StockPredictor('AAPL', period='2y')

# Pobranie danych (używa zapisanych danych jeśli dostępne)
predictor.fetch_data(use_saved=True)

# Utworzenie cech
predictor.create_features()

# Trenowanie modelu
predictor.train_model()

# Przewidywanie na kolejny dzień
prediction = predictor.predict_next_day()
print(f"Przewidywana cena: ${prediction['predicted_price']:.2f}")

# Wizualizacja
predictor.plot_predictions(days=60)
```

## Struktura folderów

Po uruchomieniu programu zostaną utworzone następujące foldery:

```
symulator/
├── stock_data/          # Zapisane dane historyczne (CSV)
├── reports/             # Raporty z przewidywań (CSV)
├── plots/               # Wykresy przewidywań (PNG)
├── app.py               # Aplikacja Streamlit (UI)
├── stock_predictor.py    # Główny moduł ML
└── daily_update.py      # Skrypt codziennej aktualizacji
```

## Cechy techniczne modelu

Model używa następujących cech:
- Średnie kroczące (MA 5, 10, 20, 50)
- RSI (Relative Strength Index)
- MACD
- Bollinger Bands
- Wskaźniki wolumenu
- Wskaźniki zmiany ceny

## Model

- **Algorytm**: Random Forest Regressor
- **Parametry**: 100 drzew, max_depth=10
- **Metryki**: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)

## Codzienne uczenie

System został zaprojektowany do codziennego uruchamiania:

1. **Zapis danych**: Wszystkie pobrane dane są zapisywane w `stock_data/`
2. **Incremental update**: System pobiera tylko nowe dane od ostatniej aktualizacji
3. **Codzienne trenowanie**: Model trenuje się codziennie na wszystkich dostępnych danych
4. **Raporty**: Automatyczne generowanie raportów z przewidywań

## Uwagi

- Model przewiduje cenę zamknięcia na następny dzień
- Dokładność zależy od wielu czynników rynkowych
- Używaj przewidywań jako wskazówek, nie jako gwarancji
- System automatycznie obsługuje różne formaty symboli (np. `.US`, `.DE`)

## Przykładowy raport

Po uruchomieniu otrzymasz raport w formacie CSV z kolumnami:
- `symbol`: Symbol spółki
- `current_price`: Aktualna cena
- `predicted_price`: Przewidywana cena na jutro
- `change`: Zmiana w dolarach
- `change_percent`: Zmiana w procentach
