from pathlib import Path
import ssl
import urllib.request

import pandas as pd


def ensure_mall_customers_csv() -> Path:
    """Download Mall_Customers.csv if it is not present locally."""
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "Mall_Customers.csv"

    if output_path.exists():
        return output_path

    # Múltiples mirrors para evitar fallos por enlace caído.
    source_urls = [
        "https://raw.githubusercontent.com/kennedykwangari/Mall-Customer-Segmentation-Data/master/Mall_Customers.csv",
        "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
    ]

    last_error = None
    for source_url in source_urls:
        try:
            # Primer intento: descarga directa estándar.
            df_remote = pd.read_csv(source_url)
            df_remote.to_csv(output_path, index=False)
            return output_path
        except Exception as err:
            last_error = err
            try:
                # Fallback para entornos macOS con problemas de certificados locales.
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(source_url, context=context) as response:
                    raw_data = response.read()
                output_path.write_bytes(raw_data)
                return output_path
            except Exception as err_ssl:
                last_error = err_ssl

    raise RuntimeError(
        "No se pudo descargar Mall_Customers.csv desde los mirrors configurados. "
        "Copia manualmente el archivo en esta carpeta con el nombre Mall_Customers.csv. "
        f"Ultimo error: {last_error}"
    )


if __name__ == "__main__":
    csv_path = ensure_mall_customers_csv()
    df = pd.read_csv(csv_path)
    print(f"Dataset listo en: {csv_path}")
    print("Primeros 5 registros:")
    print(df.head())