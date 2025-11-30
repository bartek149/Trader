"""
Skrypt do aktualizacji wyników predykcji
Może być uruchamiany codziennie jako cron job lub przy starcie aplikacji
"""
from prediction_logger import PredictionLogger, update_prediction_outcomes
from pathlib import Path


def main():
    """Główna funkcja aktualizacji wyników"""
    print("=" * 80)
    print("AKTUALIZACJA WYNIKÓW PREDYKCJI")
    print("=" * 80)
    
    # Utwórz logger
    logger = PredictionLogger(data_dir='stock_data', use_sqlite=True)
    
    # Zaktualizuj wyniki
    update_prediction_outcomes(logger, max_days_lookback=60)
    
    print("\n✓ Aktualizacja zakończona")


if __name__ == "__main__":
    main()

