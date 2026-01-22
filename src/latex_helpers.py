import pandas as pd


def df_to_custom_latex(df: pd.DataFrame, caption: str, label: str   , index_map=None, col_map=None) -> str:
    """
    Converts a Pandas DataFrame into a specific LaTeX table format with resizebox.
    
    Parameters:
    - df: The dataframe containing coalition values.
    - caption: String for the table caption.
    - label: String for the table label.
    - index_map: Dict to rename dataframe index (rows) to LaTeX symbols.
    - col_map: Dict to rename dataframe columns to LaTeX headers.
    """
    
    # 1. Create a copy to avoid modifying the original dataframe
    df_latex = df.copy()

    # 2. Rename Rows (Index) if map is provided
    if index_map:
        df_latex.rename(index=index_map, inplace=True)

    # 3. Rename Columns if map is provided
    # If a column isn't in the map, keep it as is.
    if col_map:
        df_latex.rename(columns=col_map, inplace=True)

    # 4. format the 'tabular' alignment string (e.g., "l|lllllll")
    # First column is 'l|', rest are 'l'
    num_cols = len(df_latex.columns)
    align_str = "l|" + "l" * num_cols

    # 5. Build the Header Row
    # Join column names with " & " and add end line
    header_row = " & " + " & ".join(df_latex.columns) + r" \\"

    # 6. Build the Data Rows
    data_rows = []
    for index, row in df_latex.iterrows():
        # Format numbers to 1 decimal place (change :.1f to :.2f if needed)
        values = " & ".join([f"{x:.1f}" for x in row])
        row_str = f"    {index} & {values} \\\\"
        data_rows.append(row_str)
    
    body_content = "\n".join(data_rows)

    # 7. Construct the Final LaTeX String
    # Using raw strings (r"") to handle backslashes nicely
    latex_code = f"""\\begin{{table}}[ht]
  \\centering
  \\resizebox{{0.5\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{{align_str}}}
    \\toprule
   {header_row}
    \\midrule
{body_content}
    \\bottomrule
    \\end{{tabular}}%
  }}
  \\vspace{{.1cm}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{table}}"""

    return latex_code