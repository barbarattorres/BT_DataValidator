# Data Validator

Data Validator is a sophisticated Python package designed to perform comprehensive data validation within pandas DataFrames. This tool ensures data integrity and conformity by applying a set of predefined validation rules, making it an essential part of data preprocessing workflows.

## Features

- **Custom Validation Rules:** Supports flexible, user-defined rules for different data columns, including checks for data types, uniqueness, and range limitations.
- **Automatic Data Type Detection:** Automatically identifies the nature of data within each column and applies appropriate validation checks, even if the column data types are initially misassigned.
- **Extensive Data Checks:** Provides validations for data type consistency, numerical ranges, and correct formatting of strings such as emails.
- **Error Reporting:** Aggregates and reports errors in a structured format, categorizing them by type and detailing issues found in each DataFrame column.
- **Automation and Manual Control:** Allows both automatic validation upon instantiation and manual validation to fit into various stages of data processing pipelines.

## Installation

To install Data Validator, use pip:

```bash
pip install data_validator
```

## Key Innovative Feature

### Automatic Data Type Detection

Unlike many other data validation tools that rely solely on predefined column data types, DataValidator stands out with its ability to automatically determine the nature of data within each column. This feature ensures that:

- **Date Format Handling:** Columns with multiple different date formats are correctly identified and validated as date columns.
- **Numeric String Conversion:** Columns containing numeric values but set as strings are identified and validated appropriately.

This automatic determination and context-aware validation significantly reduce the need for manual data type correction, making the validation process more intuitive and efficient.

### Comparison with Other Data Validation Tools

While many data validation tools offer standard checks for data types, ranges, and formats, DataValidator's automatic data type detection and context-aware validation differentiate it by:

- **Reducing Manual Effort:** Automatically adjusts to the actual content of the data, reducing the need for manual data type corrections.
- **Enhancing Accuracy:** Ensures that the appropriate validation rules are always applied, improving data integrity.
- **Streamlining Workflows:** Fits seamlessly into data preprocessing pipelines, offering both automated and manual control over the validation process.

## Conclusion

DataValidator is a powerful and user-friendly tool for ensuring data integrity in pandas DataFrames. Its innovative approach to automatic data type detection and comprehensive validation checks make it an invaluable asset for data preprocessing workflows, especially in environments where data quality is critical.
