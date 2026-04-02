from src.catalog import build_sku_catalog
from src.config import load_config
from src.generator import generate_sales


def main():
    cfg = load_config()
    print(f"Config cargada | SKUs: {cfg.n_skus} | {cfg.start_date} → {cfg.end_date}")
    print(f"Canales: {list(cfg.channels.keys())}\n")

    print("Construyendo catálogo de SKUs...")
    catalog = build_sku_catalog(cfg)
    catalog.to_csv(cfg.catalog_path, index=False)
    print(f"{len(catalog)} SKUs → {cfg.catalog_path}\n")

    print("Generando dataset de ventas...")
    df = generate_sales(catalog, cfg)
    df.to_csv(cfg.sales_path, index=False)
    print(f"{len(df):,} filas → {cfg.sales_path}\n")

    print("Resumen:")
    print(f"   Ingresos totales : ${df['revenue'].sum():,.0f}")
    print(
        f"   Ventas por canal :\n{df.groupby('channel')['revenue'].sum().apply(lambda x: f'   ${x:,.0f}').to_string()}"
    )


if __name__ == "__main__":
    main()
