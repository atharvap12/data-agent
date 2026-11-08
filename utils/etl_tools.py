import os
import requests
import pandas as pd
from utils.api_sources import call_api, extract_records, resolve_source


class ETLTools:

    def __init__(self):
        pass

    def extract_load(self, source:str, output_folder:str, format:str):
        """
        This tool extracts the data from the API (source) and loads it into the
        the desired location (output_folder).

        Args:
            source (str): A registered source name from API_SOURCES (e.g. "pokemon",
                "marketstack") or a full API URL.
            output_folder (str): The folder where the extracted data will be saved.
            format (str): The format in which to save the data (csv, json, parquet).

        Returns:
            str: A message indicating the success or failure of the operation.

        """

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(project_root, output_folder)

        try:
            url, params = resolve_source(source)
            payload = call_api(url, params=params)

            # Some APIs (e.g. Marketstack) answer with HTTP 200 but put an error in the body
            if isinstance(payload, dict) and "error" in payload:
                return f"Failed to extract data: {payload['error']}"

            records = extract_records(payload)
            if not records:
                return f"Failed to extract data: no records found in the response from {url}"

            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            df = pd.json_normalize(records)
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            return f"Data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"


    def transform_load_context(self, file_path:str):
        """
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.
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

        # BEFORE: we called `exec(code)` with no extra arguments. Inside a method,
        # that runs the code with the method's own local variables, which mixes two
        # scopes together. Imports in the generated code (e.g. `import pandas as pd`)
        # then end up as local variables, and code inside a function, lambda or
        # list comprehension can't see them. That causes errors like
        # "NameError: name 'pd' is not defined" even though `pd` was imported.
        #
        # NOW: we pass an empty dict `{}` as the globals. The generated code then runs
        # like its own small, separate script: everything it imports or defines
        # is visible everywhere in that code. It also can't read or change
        # variables from this method by accident.
        try:
            exec(code, {})
            return "Code executed successfully."
        except Exception as e:
            return f"Failed to execute code: {e}"


if __name__ == "__main__":
    obj = ETLTools()
    path = "C:\\Data_Agent\\data\\extract\\extracted_data.csv"
    print(obj.transform_load_context(path))
          