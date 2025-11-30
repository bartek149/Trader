"""
Skrypt do codziennego uruchamiania systemu przewidywania cen akcji.
Można go uruchomić ręcznie lub ustawić jako zadanie cron/scheduled task.
"""

from stock_predictor import MultiStockPredictor
from datetime import datetime
import sys

def main():
    print(f"\n{'='*80}")
    print(f"CODZIENNA AKTUALIZACJA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # Utworzenie multi-stock predictor
        multi_predictor = MultiStockPredictor(excel_file='raport.xlsx')
        
        # Wczytaj symbole z pliku Excel
        multi_predictor.load_symbols_from_excel()
        
        # Codzienna aktualizacja - ZAWSZE pobierz wszystkie spółki niemieckie
        results = multi_predictor.daily_update(include_german_stocks=True, all_german_stocks=True)
        
        # Generuj raport
        if results:
            multi_predictor.generate_report(results)
            print(f"\n✓ Aktualizacja zakończona pomyślnie dla {len(results)} spółek")
            return 0
        else:
            print("\n✗ Brak wyników")
            return 1
            
    except Exception as e:
        print(f"\n✗ Błąd: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

