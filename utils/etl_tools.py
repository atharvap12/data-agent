import os
import requests
import pandas as pd
from datetime import datetime
from utils.api_sources import API_SOURCES, call_api, extract_records, resolve_source, source_name

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Fixed output folders. The tools always save here; the LLM cannot choose another folder.
EXTRACT_DIR = os.path.join(PROJECT_ROOT, "data", "extract")
TRANSFORM_DIR = os.path.join(PROJECT_ROOT, "data", "transform")


def new_file_path(folder: str, source: str, format: str) -> str:
    """
    Build a path for a NEW file named <source>_<timestamp>.<format>, e.g.
    data/extract/marketstack_20261108_101500.csv.

    Old files are never overwritten: if a file with that name already exists
    (two runs in the same second), a counter is added: ..._101500_1.csv.

    Args:
        folder (str): Output folder, EXTRACT_DIR or TRANSFORM_DIR.
        source (str): Source name, e.g. "pokemon" or "marketstack".
        format (str): File extension without the dot (csv, json, parquet).

    Returns:
        str: The absolute path of the new file. The folder is created if needed.
    """
    os.makedirs(folder, exist_ok=True)

    stem = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = os.path.join(folder, f"{stem}.{format}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{counter}.{format}")
        counter += 1
    return path


def source_from_file(file_path: str) -> str:
    """
    Read the source back from a file name made by new_file_path, e.g.
    "data/extract/marketstack_20261108_101500.csv" gives "marketstack".
    Files with other names (e.g. the old "extracted_data.csv") give "data".
    """
    prefix = os.path.basename(file_path).split("_")[0].lower()
    return prefix if prefix in API_SOURCES else "data"


class ETLTools:

    def __init__(self):
        pass

    def extract_load(self, source:str, format:str):
        """
        This tool extracts the data from the API (source) and saves it as a new
        file in data/extract (EXTRACT_DIR).

        Args:
            source (str): A registered source name from API_SOURCES (e.g. "pokemon",
                "marketstack") or a full API URL.
            format (str): The format in which to save the data (csv, json, parquet).

        Returns:
            str: A message indicating the success or failure of the operation.

        """

        if format not in ("csv", "json", "parquet"):
            return f"Unsupported format: {format}"

        try:
            url, params = resolve_source(source)
            payload = call_api(url, params=params)

            # Some APIs (e.g. Marketstack) answer with HTTP 200 but put an error in the body
            if isinstance(payload, dict) and "error" in payload:
                return f"Failed to extract data: {payload['error']}"

            records = extract_records(payload)
            if not records:
                return f"Failed to extract data: no records found in the response from {url}"

            # Always a new file: <source>_<timestamp>.<format>
            filename = new_file_path(EXTRACT_DIR, source_name(source), format)

            df = pd.json_normalize(records)
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)

            return f"Data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"


    def transform_load_context(self, file_path:str):
        """
        Read the file to be transformed and return its first 3 rows, so the LLM
        can see the columns and values before writing the transformation code.

        Args:
            file_path (str): The path to the file containing the data to be transformed.
        Returns:
            str: The first 3 rows of the data, or an error message.
        """

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        top_3_rows = str(df.head(3))

        return top_3_rows


    def execute_code(self,code:str):
        """
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """

        # Fresh globals dict so the generated code runs like its own script: its imports are
        # visible inside its functions/lambdas, and it can't touch this method's locals.
        try:
            exec(code, {})
            return "Code executed successfully."
        except Exception as e:
            return f"Failed to execute code: {e}"


if __name__ == "__main__":
    obj = ETLTools()
    path = "C:\\Data_Agent\\data\\extract\\extracted_data.csv"
    print(obj.transform_load_context(path))
          