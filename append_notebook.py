import nbformat
import sys

def append_to_notebook(notebook_path, script_path):
    print(f"Appending {script_path} to {notebook_path}...")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()

        md_cell = nbformat.v4.new_markdown_cell("## Optimized RandomForest Training pipeline\nThis section uses SMOTE, feature engineering, and Hyperparameter Tuning to maximize model accuracy.")
        code_cell = nbformat.v4.new_code_cell(code)

        nb.cells.extend([md_cell, code_cell])

        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print("Successfully appended optimized pipeline to the IPYNB notebook.")
    except Exception as e:
        print(f"Error appending to notebook: {e}")

if __name__ == '__main__':
    append_to_notebook('malnuteition-final.ipynb', 'optimize_model.py')
