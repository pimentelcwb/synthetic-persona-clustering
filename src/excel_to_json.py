import os
import pandas as pd
import json

# pasta de origem
folder = r"your_path"

# percorre todos os arquivos da pasta
for file in os.listdir(folder):
    if file.lower().endswith(".xlsx") and not file.startswith("~$"):
        xlsx_path = os.path.join(folder, file)

        # lê como texto pra preservar tudo
        df = pd.read_excel(xlsx_path, dtype=str)

        # troca strings vazias/espacos por NA e preenche NaN com "NA"
        df = df.replace(r"^\s*$", pd.NA, regex=True).fillna("NA")

        json_path = os.path.join(folder, file[:-5] + ".json")

        data = df.to_dict(orient="records")  # cada linha = 1 objeto com todas as colunas

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"rows": data}, f, ensure_ascii=False, indent=2)

        print(f"Convertido: {file} -> {os.path.basename(json_path)}")

print("Concluído.")