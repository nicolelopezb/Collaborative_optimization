"""
run_demand_extraction.py
--------------------------
Carga tu archivo real de Aerocivil, lo agrega a frecuencia semanal por
ruta, y guarda el resultado en data/processed/demand_weekly.csv

Uso:
    python src/run_demand_extraction.py data/raw/TU_ARCHIVO.csv
    python src/run_demand_extraction.py data/raw/TU_ARCHIVO.csv --year 2025 --month 7
"""

import argparse
import sys
sys.path.insert(0, "src")

from demand import load_aerocivil_operations, monthly_to_weekly_frequency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--out", default="data/processed/demand_weekly.csv")
    args = parser.parse_args()

    routes_monthly = load_aerocivil_operations(args.csv_path, year=args.year, month=args.month)
    routes_weekly = monthly_to_weekly_frequency(routes_monthly)

    routes_weekly.to_csv(args.out, index=False)
    print(f"\nGuardado en: {args.out}")
    print(f"Rutas con demanda semanal > 0: {len(routes_weekly)}")
    print(routes_weekly.sort_values("weekly_frequency", ascending=False).head(10))


if __name__ == "__main__":
    main()
