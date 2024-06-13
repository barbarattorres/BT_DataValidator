# ---------------------- Execute save dataframes function & print results ---------------------- #

# ---------------------- Standard Library Imports ---------------------- #

import datetime
from datetime import datetime
from dateutil import parser as date_parser
import unicodedata
import logging
import re
import sys
import warnings
from fractions import Fraction

# ---------------------- Data Handling and Processing ---------------------- #
import pandas as pd
import numpy as np

# ---------------------- Visualization ---------------------- #
from IPython.display import display, HTML

# ---------------------- Configuration Settings ---------------------- #
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter('ignore')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Set display options for Pandas
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50) 
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.float_format', '{:.4f}'.format)


class EnvironmentChecker:
    """
    A class to check and report the execution environment (interactive or non-interactive).

    Attributes:
        interactive (bool): Determines if the script is running in an interactive environment.

    Methods:
        is_interactive(): Returns True if the environment is interactive.
        log_environment(): Logs the environment type to a logging service.
        print_environment(): Prints the environment type to the standard output.
    """
    def __init__(self, show=False):
        # Determine the interactive state once during instantiation
        self.interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        self.show = show
        
        # Automatically log and print the environment based on the show flag
        if self.show:
            self.log_environment()
            #self.print_environment()

    def is_interactive(self):
        """Return True if code is running in an interactive environment."""
        return self.interactive

    def log_environment(self):
        """Log the current operating environment using the logging library."""
        if self.is_interactive():
            logging.info("Running in an interactive environment (e.g., Jupyter).")
        else:
            logging.info("Running in a non-interactive environment (e.g., standard Python script).")

    def print_environment(self):
        """Print the current operating environment to the standard output."""
        if self.is_interactive():
            print("This is an interactive environment.")
        else:
            print("This is a non-interactive environment.")


env_checker = EnvironmentChecker()

# --------------------------- PREPROCESSING: Data Validation -------------------- #

class DataValidator:
    
    def __init__(self, dataframe, rules, exclude_cols=None, auto_validate=True):
        """
        Intro:
        ==========
        The `DataValidator` class is a sophisticated tool designed for comprehensive data validation 
        within a pandas DataFrame. It facilitates the enforcement of data integrity and conformity by 
        applying a set of predefined validation rules to the DataFrame's columns. This class is crucial 
        in data preprocessing workflows, where ensuring data quality and consistency directly influences 
        the accuracy and reliability of subsequent data analysis or machine learning models.

        Key Features:
        ----------
        - **Custom Validation Rules:** Supports flexible, user-defined rules for different data columns, 
        including data type checks, uniqueness constraints, range limitations, and validations.
        - **Extensive Data Checks:** Implements various checks like data type validation, uniqueness, 
        numerical range compliance, and string format verification (e.g., emails, proper capitalization).
        - **Error Reporting:** Aggregates and reports errors in a structured format, categorizing them by type 
        and detailing issues found in each DataFrame column.
        - **Automated and Manual Validation Control:** Allows automatic validation upon instantiation or manual 
        validation to integrate into different stages of data processing pipelines.
        - **Exclusion Capability:** Excludes certain columns from validation based on user-defined criteria.

        Usage Context:
        ----------
        Useful in scenarios where data quality impacts business decisions, analytical accuracy, or the 
        performance of predictive models. Essential for data cleaning and preparation stages, and can be 
        integrated into data ingestion pipelines or used in ad-hoc data quality assessments.

        Attributes:
        ----------
        - **dataframe (pd.DataFrame):** The dataframe to validate.
        - **rules (dict):** A dictionary where keys are column names and values are validation rules.
        - **exclude_cols (list, optional):** A list of columns to exclude from validation.
        - **error_dfs (dict, optional):** Stores error dataframes for each error type if auto_validate is True.

        Methods:
        ----------
        - **validate():** Validates the DataFrame according to the rules and returns a dictionary of error dataframes.
        - **_check_type(column):** Determines the likely data type of a column based on its content.
        - **_represents_numeric(value):** Checks if a value can be interpreted as numeric.
        - **_represents_alpha(value):** Checks if a value consists purely of alphabetical characters.
        - **_represents_email(value):** Checks if a value is in a valid email format.
        - **_represents_date(value):** Checks if a value can be parsed as a date.
        - **_check_data_type(column, rules):** Validates data types of a column against expected types defined in rules.
        - **_check_uniqueness(column, rules):** Checks if values in a column are unique if required by the rules.
        - **_check_range(column, rules):** Ensures numeric values in a column fall within a specified range.
        - **_check_date_format(column, rules, column_type):** Checks if dates in a column adhere to a specified format, ignoring special characters.
        - **_check_special_chars(column, rules, column_type):** Checks for unwanted special characters in a column.
        - **_contains_special_chars(value, special_chars):** Helper method to detect special characters in a string.
        - **_contains_decomposed_chars(value):** Checks if a value contains decomposed Unicode characters.
        - **_check_accent_chars(column, rules):** Checks for the presence of accent characters in a column.
        - **_check_proper_case(column, column_type):** Ensures that strings in a column are properly capitalized.
        - **_check_consistent_and_disproportional(column, column_type):** Checks if the range of numeric values in a column is disproportionately wide.
        - **_is_range_disproportional(column, threshold_ratio):** Helper method to determine if the range of numeric values is disproportionately large.
        - **_check_email_format(column, column_type):** Validates the format of email addresses in a column.
        - **_check_whitespace_issues(column):** Checks for leading or trailing whitespaces in a column.
        - **_check_no_numerics_in_alphabetical(column, column_type):** Ensures no numeric values are present in columns classified as alphabetical.
        - **_check_no_alpha_in_numeric(column, column_type):** Ensures no alphabetical characters are present in columns classified as numeric
        - **_get_column_nature(column):** Determines the nature of the column based on the type of data it predominantly holds.
        - **_check_contains(column, rules):** Check if each value in the column contains a specified substring.

        Example:
        ----------
        The following example demonstrates how to create an instance of the DataValidator class with a simple set
        of validation rules and a sample DataFrame.

        ```python
        import pandas as pd

        # Define a sample DataFrame
        data = {
            'age': [25, 28, -1, 32, 50],
            'email': ['john@example.com', 'jane.doe@example.com', 'invalid-email', 'alice@example.com', 'bob@example.com']
        }

        df = pd.DataFrame(data)

        # Define validation rules
        rules = {
            'age': {'min': 0, 'max': 100, 'data_type': int},
            'email': {'format': 'email'}
        }

        # Instantiate the DataValidator and perform & print validation
        validator = DataValidator(df, rules)

        """
        self.dataframe = dataframe
        self.rules = rules
        self.env_checker = EnvironmentChecker(show=False)
        self.exclude_cols = self.validate_exclude_cols(exclude_cols)

        if auto_validate:
            self.error_dfs = self.validate()
            self.display_errors()
        else:
            self.error_dfs = {}


    def validate_exclude_cols(self, exclude_cols):
        if exclude_cols is None:
            return []
        if not isinstance(exclude_cols, list):
            logging.info("exclude_cols should be a list of column names. Incorrect format provided.")
            return []
        
        # Check if all columns in exclude_cols are in the dataframe
        missing_cols = [col for col in exclude_cols if col not in self.dataframe.columns]
        if missing_cols:
            logging.info(f"Columns listed in exclude_cols not found in dataframe: {missing_cols}")
        
        return [col for col in exclude_cols if col in self.dataframe.columns]

    def __repr__(self):
        """
        Returns a string representation of the DataValidator object, showing the number of columns and rules applied.
        Returns:
            str: Description of the DataValidator instance.
        """
        return f"<DataValidator: {len(self.dataframe)} error types, {len(self.rules)} rules>"

# --------------------------------------------------------- DISPLAY CHECKS ------------------------------------------------------------- #

    def display_errors(self):
        """
        Display errors using different methods based on the execution environment.
        """
        if self.env_checker.is_interactive():
            # Use HTML display in interactive environments
            self._display_html()
        else:
            # Fallback to console display or logging in non-interactive environments
            self._display_text()

    def _display_html(self):
        # Existing HTML display logic
        fixed_width = '1000px'
        title_font_size = '16px'
        column_widths = {
            'Index': '50px',
            'Column': '150px',
            'Datatype': '100px',
            'Proposed': '100px',
            'Error': '400px',
            'Count': '50px'
        }
        headers_html = "<tr>" + "".join(f"<th style='width: {width}; text-align: left;'>{name}</th>" for name, width in column_widths.items()) + "</tr>"

        for error_type, df in self.error_dfs.items():
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Index'}, inplace=True)
            rows_html = "".join("<tr>" + "".join(f"<td style='width: {column_widths[col]}; text-align: left;'>{cell}</td>" for col, cell in row.items()) + "</tr>" for _, row in df.iterrows())
            table_html = f"<table style='width:{fixed_width}; table-layout: fixed; border-collapse: collapse;'>{headers_html}{rows_html}</table>"
            display(HTML(f"<h4 style='font-size:{title_font_size}; text-align: left;'>{error_type}</h4>{table_html}"))

    def _display_text(self):
        # Fallback display for text-based environments
        for error_type, df in self.error_dfs.items():
            print("\n" + f"{error_type}:")
            print(df.to_string(index=False))

# --------------------------------------------------------- VALIDATION CHECKS ------------------------------------------------------------- #

    def validate(self, show=True):
        """
        Validate the dataframe columns based on the provided rules. This function orchestrates the validation
        by iterating over each column, applying the checks defined in the rules, and aggregating any errors.

        Returns:
            dict: A dictionary containing error dataframes for each error type detected.
        """
        error_data = {}
        proposed_types = {}

        # Loop through each column in the validation rules
        for column, rule_details in self.rules.items():
            if column not in self.dataframe.columns:
                logging.info(f"Column '{column}' specified in rules is not found. Skipping validation_rule.")

        for column in self.dataframe.columns:
            if self.exclude_cols and column in self.exclude_cols:
                continue

            rules = self.rules.get(column, {})
            column_datatype = str(self.dataframe[column].dtype)
            column_errors = []

            # Attempt to determine the data type with error handling
            try:
                type_classification = self._check_type(self.dataframe[column])
                proposed_types[column] = type_classification  
            except Exception as e:
                logging.error(f"Error determining type for {column}: {e}")
                continue

            # Perform each check
            checks = [
                (self._check_data_type, [self.dataframe[column], rules]),
                (self._check_uniqueness, [self.dataframe[column], rules]),
                (self._check_special_chars, [self.dataframe[column], rules, proposed_types[column]]),
                (self._check_accent_chars, [self.dataframe[column], proposed_types[column]]),
                (self._check_consistent_and_disproportional, [self.dataframe[column], proposed_types[column]]),
                (self._check_proper_case, [self.dataframe[column], proposed_types[column]]),
                (self._check_email_format, [self.dataframe[column], proposed_types[column]]),
                (self._check_whitespace_issues, [self.dataframe[column]]),
                (self._check_no_numerics_in_alphabetical, [self.dataframe[column], proposed_types[column]]),
                (self._check_no_alpha_in_numeric, [self.dataframe[column], proposed_types[column]]),
                (self._check_date_format, [self.dataframe[column], rules, proposed_types[column]]),
                (self._check_contains, [self.dataframe[column], rules]), 
            ]

            # Range check, only if 'min' or 'max' is defined
            if 'min' in rules or 'max' in rules:
                checks.append((self._check_range, [self.dataframe[column], rules, proposed_types[column]]))

            for check_func, args in checks:
                try:
                    column_errors.extend(check_func(*args))
                except Exception as e:
                    logging.error(f"Error executing {check_func.__name__} for column {column}: {e}")

            # Aggregate error messages
            for error_type, error_values in column_errors:
                unique_error_values = list(set(error_values))
                count_errors = len(unique_error_values)
                error_data.setdefault(error_type, []).append({'Column': column, 'Datatype': column_datatype, 'Proposed': proposed_types[column], 'Error': unique_error_values, 'Count': count_errors})

        error_dfs = {error_type: pd.DataFrame(data) for error_type, data in error_data.items()}

        if show:
            # Logging the number of different error types found
            logging.info(f" {len(error_dfs)} different error types were detected in the dataframe")
        return error_dfs
    

    def _check_type(self, column):
        """
        Determine the likely data type of a column based on its content.

        Args:
        column (pd.Series): The column to classify.

        Returns:
        str: The identified type of the column.
        """
        numeric_count = 0
        alpha_count = 0
        email_count = 0
        date_count = 0
        time_count = 0
        ip_count = 0
        total_valid = 0

        for value in column.dropna():
            str_value = str(value).strip().rstrip('?')  # Standardize to string and strip trailing characters
            is_numeric = self._is_potentially_numeric(str_value)
            is_alpha = self._represents_alpha(str_value)
            is_email = self._represents_email(str_value)
            is_date = self._represents_date(str_value)
            is_time = self._represents_time(str_value)
            is_ip = self._represents_ip(str_value)

            numeric_count += is_numeric
            alpha_count += is_alpha and not is_numeric  # Only count as alpha if not numeric
            email_count += is_email
            date_count += is_date
            time_count += is_time
            ip_count += is_ip

            total_valid += 1

        # Debugging outputs
        #print(f"\nNumeric: {numeric_count}, Alpha: {alpha_count}, Email: {email_count}, Date: {date_count}, Time: {time_count}, IP: {ip_count}, Total: {total_valid}")

        if numeric_count > total_valid / 2:
            if date_count >= numeric_count * 0.5:  # If half of the numeric values are dates
                return 'date'
            elif time_count >= numeric_count * 0.5:  # If half of the numeric values are times
                return 'time'
            elif ip_count >= numeric_count * 0.5:  # More than half are IP addresses
                return 'ip_address'
            else:
                return 'numeric'
        elif alpha_count > total_valid / 2:
            if email_count > alpha_count * 0.5:
                return 'email'
            else:
                return 'alphabetical'
        else:
            return 'mixed'


    def _is_potentially_numeric(self, value):
        """Attempt to interpret a string as numeric, considering date-like and time-like formats."""
        # Regular expression to detect typical IP address patterns
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$')

        if ip_pattern.match(value):
            # This matches an IP address pattern, so we return False for numeric
            return True
        try:
            # Try to convert directly to float
            float(value)
            return True
        except ValueError:
            # Handle fractions and date-like patterns
            try:
                # Check if it's a fraction (e.g., '1/2')
                Fraction(value)
                return True
            except ValueError:
                # Regex to check for date and time patterns
                if re.match(r'^\d{1,4}([-./])\d{1,2}\1\d{1,4}$', value):  # Date formats
                    return True
                if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', value):  # Time formats
                    return True
                # Regex to identify strings with more digits than alphabetic characters
                digits = sum(c.isdigit() for c in value)
                letters = sum(c.isalpha() for c in value)
                if digits > letters:
                    return True
                return False

    def _represents_alpha(self, value):
        """ Check if the value represents an alphabetical string. """
        return isinstance(value, str) and any(c.isalpha() for c in value)

    def _represents_email(self, value):
        """ Check if the value represents an email. """
        if isinstance(value, str):
            parts = value.split('@')
            return len(parts) == 2 and all(part for part in parts)
        return False

    def _represents_date(self, value):
        """ Check if the value represents a date. """
        # A more comprehensive regex that attempts to capture most common date formats
        date_pattern = re.compile(
            r"""
            \b                      # Start at a word boundary
            (                       # Start of group:
            (?:                   # Try to match (non-capturing):
                \d{1,4}             # Year (1 to 4 digits)
                [-/\.]              # Separator
                \d{1,2}             # Month or day (1 or 2 digits)
                [-/\.]              # Separator
                \d{1,4}             # Day or year (1 to 4 digits)
            )                     # End non-capturing group
            |                     # OR
            (?:                   # Another non-capturing group:
                \d{1,2}             # Month or day (1 or 2 digits)
                [-/\.]              # Separator
                \d{1,2}             # Day or month (1 or 2 digits)
                [-/\.]              # Separator
                \d{2,4}             # Year (2 to 4 digits)
            )                     # End group
            )                       # End of first main group
            \b                      # End at a word boundary
            """, re.VERBOSE)

        if isinstance(value, str) and date_pattern.search(value):
            try:
                # Attempt to parse the date to confirm it's valid
                parsed_date = date_parser.parse(value, fuzzy=True)
                # Check if the parsed date is within a reasonable range (e.g., 1900-2099)
                return 1900 <= parsed_date.year <= 2099
            except (ValueError, TypeError):
                return False
        return False

    def _represents_time(self, value):
        """ Check if the value represents a time format. """
        if isinstance(value, str):
            # Regular expression to match HH:MM or HH:MM:SS
            time_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
            return bool(time_pattern.match(value))
        return False

    def _represents_ip(self, value):
        """ Check if the value represents a valid IP address. """
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$')
        if ip_pattern.match(value):
            parts = value.split('.')
            return all(0 <= int(part) <= 255 for part in parts if part.isdigit())
        return False


# --------------------------------------------------------- AUTO CHECKS ------------------------------------------------------------- #

    def _check_special_chars(self, column, rules, column_type):
        """
        Check for unwanted special characters in a column.

        Args:
            column (pd.Series): The column to be checked.
            rules (dict): Dictionary containing validation rules for special characters.
            column_type (str): Type of the column (e.g., 'email', 'date') which could modify the set of characters to check.

        Returns:
            list: A list of errors found in the column.
        """
        column_errors = []
        default_special_chars = ['.', '?', '!', ';', '*', '+', '/', '{', '(', '§', '$', '%', ':', '@', '&', '#', '"', "'"]
        special_chars = rules.get('special_chars', default_special_chars)

        # Adjust special characters based on column type
        if column_type == 'email':
            special_chars = [char for char in special_chars if char not in ['@', '.']]
        elif column_type == 'date':
            special_chars = [char for char in special_chars if char not in ['.']]
        elif column_type == 'ip_address':
            special_chars = [char for char in special_chars if char not in ['.']]
        elif column_type == 'time':
            special_chars = [char for char in special_chars if char not in [':']]
        elif column_type == 'numeric':
            # For numeric columns, retain only those special characters that are unusual in numeric values
            special_chars = [char for char in special_chars if char not in ['.', '-', '+', 'e', 'E']]

        # Convert column to string if not already, to safely use .str accessor
        if column.dtype != 'object' and column.dtype.name != 'string':
            column = column.astype(str)

        # Check for special characters if not skipped by rules
        if not rules.get('skip_special_chars', False):
            error_mask = self._contains_special_chars(column, special_chars)
            if error_mask.any():
                error_values = column[error_mask].tolist()
                column_errors.append(("Special character check failed", error_values))
        return column_errors


    def _contains_special_chars(self, column, special_chars):
        """
        Helper to check for special characters in a string using vectorized operations.

        Args:
            column (pd.Series): The column of data to check.
            special_chars (list): List of special characters to check for in the column.

        Returns:
            pd.Series: A boolean series indicating which elements contain special characters.
        """
        pattern = '|'.join([re.escape(char) for char in special_chars])  # Escape to handle special regex characters
        contains_specials = column.str.contains(pattern, regex=True, na=False)
        return contains_specials

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_accent_chars(self, column, column_type):
        """
        Check for unwanted accent characters in a column of type 'alphabetical'.
        Only specific diacritical marks are considered acceptable.

        Args:
            column (pd.Series): The column to be checked.
            column_type (str): Expected to be 'alphabetical' for this operation.

        Returns:
            list: A list of error messages detailing unwanted accent characters.
        """
        column_errors = []
        if column_type == 'alphabetical':
            try:
                # Convert to string to ensure consistent processing
                if column.dtype != 'object' and column.dtype != 'string':
                    column = column.astype(str)

                # Normalize and decompose each string to check for specific diacritical marks
                normalized = column.str.normalize('NFD')

                # Check each decomposed string for unwanted accents
                mask = normalized.map(self._contains_unwanted_accents)
                if mask.any():
                    error_values = column[mask].tolist()
                    column_errors.append(("Accent character check failed", error_values))
            except Exception as e:
                logging.error(f"Error checking accent characters in column: {e}")
                column_errors.append(("Error checking accent characters", str(e)))

        return column_errors

    def _contains_unwanted_accents(self, decomposed_string):
        """
        Determine if a decomposed string contains any diacritical marks that are not explicitly allowed.

        Args:
            decomposed_string (str): The Unicode Normalized (NFD) string to check.

        Returns:
            bool: True if unwanted diacritical marks are found, False otherwise.
        """
        allowed_accents = {'ä', 'ü', 'ö', 'Ä', 'Ü', 'Ö'}
        if isinstance(decomposed_string, str):
            try:
                for char in decomposed_string:
                    # Check if the character is a combining diacritical mark
                    if unicodedata.category(char).startswith('M') and unicodedata.normalize('NFC', 'a' + char) not in allowed_accents:
                        return True
                return False
            except TypeError:
                logging.error("Non-string input received while checking for unwanted accents.")
                return False
        else:
            return False  # Ignore non-string types
        
# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_proper_case(self, column, column_type):
        """
        Ensure that names in a column follow a proper case (capitalized properly) according to the rules of natural names
        and also allow for specialized identifiers such as enum names or constants which are typically used in programming
        and configuration files.

        Args:
        column (pd.Series): The column to be checked.
        column_type (str): The expected type of data in the column, used here to verify it's 'alphabetical'.

        Returns:
        list: A list of error messages with the associated incorrect entries if the column entries do not conform
            to the expected capitalization patterns.
        """
        column_errors = []

        if column_type == 'alphabetical':
            try:
                # Convert column to string if it's not already
                if column.dtype != 'object' and column.dtype != 'string':
                    column = column.astype(str)

                # Handle NaN values by replacing them with an empty string to avoid issues with regex operations
                column = column.fillna('')

                # Regex to match names that are properly capitalized, including those with diacritical marks
                pattern = re.compile(
                    r'^([A-ZÀÁÂÄÆÃÅĀÇĆČÈÉÊËĒĖĘÎÏÍĪĮÌÔÖÒÓŒØŌÕÚÜÙÛŪÑŃ]'  # Start with an uppercase letter (with diacritics)
                    r'[a-zàáâäæãåāçćčèéêëēėęîïíīįìôöòóœøōõúüùûūñń\'_]*'  # Followed by any number of lowercase letters (with diacritics), apostrophes, or underscores
                    r'([ -]'  # Space or hyphen delimiters, allowing for parts of names to be split by spaces or hyphens
                    r'[A-ZÀÁÂÄÆÃÅĀÇĆČÈÉÊËĒĖĘÎÏÍĪĮÌÔÖÒÓŒØŌÕÚÜÙÛŪÑŃ]'  # Next part starts with an uppercase letter
                    r'[a-zàáâäæãåāçćčèéêëēėęîïíīįìôöòóœøōõúüùûūñń\'_]*)*)$'  # Followed by lowercase letters, apostrophes, or underscores
                    r'|'  # OR allow for terms that are valid enum identifiers like 'Research_Development'
                    r'^([A-Z][A-Za-z0-9_]+)$'  # Enum style identifiers
                )

                # Create a mask where each valid item returns True; otherwise False
                valid_case_mask = column.str.match(pattern, na=False)

                # Find entries where valid_case_mask is False (invalid cases)
                invalid_cases = ~valid_case_mask

                if invalid_cases.any():
                    error_values = column[invalid_cases].tolist()
                    column_errors.append(("Proper case check failed", error_values))
            except Exception as e:
                logging.error(f"Error in _check_proper_case for column type {column_type}: {e}")
                column_errors.append(("Error during proper case validation", str(e)))

        return column_errors

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_date_format(self, column, rules, column_type):
        """
        Checks if dates in a column adhere to a specified format, ignoring special characters.
        """
        column_errors = []
        if column_type == 'date':  # Only proceed if the column is classified as date type
            expected_format = rules.get('date_format', "%d.%m.%Y")  # Use specified format or default

            # Regex pattern to remove special characters except date separators
            # Allow date separators like . - /
            pattern = re.compile(r"[^0-9.\-\/]")

            # Convert all column entries to string to prevent attribute errors
            column = column.astype(str)  # Convert the entire column to strings

            for value in column:
                cleaned_value = pattern.sub('', value.strip())  # Remove special characters and strip whitespace
                try:
                    # Attempt to parse the date. If it fails, it means the format is not as expected.
                    datetime.strptime(cleaned_value, expected_format)
                except ValueError:
                    # Add to errors if parsing fails
                    column_errors.append(value)

            if column_errors:
                return [("Date format error, expected " + expected_format, column_errors)]
        return []

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_consistent_and_disproportional(self, column, column_type):
        """
        Check if the numeric column's range is disproportionately wide.
        """
        column_errors = []
        if column_type == 'numeric':
            is_disproportional, dynamic_threshold, error = self._is_range_disproportional(column)
            if is_disproportional:
                column_errors.append((f"Range disproportion found with threshold: {dynamic_threshold:.2f}", error))
        return column_errors

    def _represents_numeric(self, value):
        """ Enhanced to handle fractional representations. """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            try:
                # Try to convert fraction (e.g., '1/2' or '½')
                float(Fraction(str(value)))
                return True
            except (ValueError, TypeError):
                return False

    def _calculate_dynamic_threshold(self, numeric_values):
        """
        Dynamically calculates a threshold ratio based on the statistics of the numeric values.

        Args:
        numeric_values (pd.Series): Numeric values from a column.

        Returns:
        float: Calculated dynamic threshold ratio.
        """
        mean = numeric_values.mean()
        std_dev = numeric_values.std()

        # Prevent dividing by zero or excessively low means
        mean = max(abs(mean), 1e-6)  # Set a minimum mean value to avoid division issues

        cv = std_dev / mean
        skewness = numeric_values.skew()
        base_threshold = 10

        # Constrain the CV to prevent excessively high or low values
        cv = min(max(cv, 0.01), 100)

        # Adjust base threshold by half the CV
        adjustment_factor = 1 + 0.5 * cv

        if abs(skewness) > 1:  # Consider skewness for highly skewed distributions
            adjustment_factor *= 1.5

        # Constrain the dynamic threshold itself to prevent exceedingly high/low values
        dynamic_threshold = base_threshold * adjustment_factor
        dynamic_threshold = min(max(dynamic_threshold, 1), 1000)

        return dynamic_threshold

    def _is_range_disproportional(self, column, default_threshold_ratio=10):
        try:
            numeric_values = pd.to_numeric(column, errors='coerce')
            numeric_values.dropna(inplace=True)

            if numeric_values.empty:
                logging.debug("No valid numeric data to analyze.")
                return False, None, []

            dynamic_threshold = self._calculate_dynamic_threshold(numeric_values)
            min_value, max_value = numeric_values.min(), numeric_values.max()
            range_value = max_value - min_value
            unique_count = numeric_values.nunique()
            ratio = range_value / unique_count
            is_disproportional = ratio > dynamic_threshold
            #return is_disproportional, dynamic_threshold, [f"Uniques: {unique_count}, Min: {min_value}, Max: {max_value}, Range: {range_value:.0f}, Ratio: {ratio:.0f}"]
            return is_disproportional, dynamic_threshold, [f"Uniques: {unique_count}, Min: {min_value}, Max: {max_value}, Ratio: {ratio:.0f}"]
        except Exception as e:
            logging.error("Failed to calculate disproportionality: " + str(e))
            return False, None, ["Error calculating disproportionality: " + str(e)]

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_email_format(self, column, column_type):
        """
        Validate the format of email addresses in a column.
        """
        column_errors = []
        if column_type == 'email':
            # Comprehensive regex for email validation:
            email_pattern = re.compile(
                r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+'              # Local part: Usernames with acceptable characters
                r'@[a-zA-Z0-9-]+'                                  # Domain name part before the last dot
                r'(?:\.[a-zA-Z0-9-]+)*'                            # Subdomain(s): Allow multiple subdomains
                r'\.[a-zA-Z]{2,}$'                                 # Top-level domain: at least two letters
            )

            # Convert to string in case of non-string types and check against the regex
            format_issues = column.astype(str).str.match(email_pattern) == False
            
            if format_issues.any():
                # Collect errors
                column_errors.append(("Email format check failed", column[format_issues].tolist()))
        return column_errors


    def _check_whitespace_issues(self, column):
        """
        Check for leading or trailing whitespaces in a DataFrame column using vectorized operations.

        Args:
        column (pd.Series): The column to be checked.

        Returns:
        list: A list of tuples with error messages and the values that have whitespace issues.
        """
        column_errors = []
        try:
            # Ensure column is in string format if it's categorized as alphabetical
            if column.dtype != 'object' and column.dtype != 'string':
                column = column.astype(str)

            # Vectorized check for leading or trailing whitespaces
            leading_spaces = column.str.startswith(' ')
            trailing_spaces = column.str.endswith(' ')

            # Combine masks to find any leading or trailing whitespace issues
            whitespace_issues = leading_spaces | trailing_spaces

            if whitespace_issues.any():
                # Get unique entries with whitespace issues for reporting
                error_values = column[whitespace_issues].unique().tolist()
                column_errors.append(("Whitespace issues found", error_values))
        except Exception as e:
            # Log the exception details
            logging.error(f"Error checking whitespace in column: {e}")
            column_errors.append(("Error during whitespace validation", str(e)))

        return column_errors


    def _check_no_numerics_in_alphabetical(self, column, column_type):
        """
        Check that no numeric values are present in columns classified as alphabetical,
        distinguishing between numerics embedded in strings and purely numeric values.

        Parameters:
        - column (pd.Series): The column to be checked.
        - column_type (str): Expected to be 'alphabetical' for this operation.

        Returns:
        - list: A list containing tuples of error messages and the list of values that failed the validation.
        """
        column_errors = []

        if column_type != 'alphabetical':
            #logging.error(f"Incorrect column type '{column_type}' provided. Expected 'alphabetical'.")
            return column_errors

        try:
            # Check for purely numeric entries
            if column.dtype == 'int' or column.dtype == 'float':
                column_errors.append(("Numeric values in alphabetical column", column.tolist()))
            else:
                # Convert to string to ensure consistent processing
                column_str = column.astype(str)
                # Check for numeric characters using a regex pattern
                numeric_issues = column_str.str.contains(r'\d', regex=True)
                
                if numeric_issues.any():
                    # Collect values that have numeric characters
                    error_values = column[numeric_issues].tolist()
                    column_errors.append(("Numeric chars found in alphabetical column", error_values))
        except Exception as e:
            logging.error(f"Error checking numeric characters in column: {e}")
            column_errors.append(("Error during numeric character validation", str(e)))

        return column_errors


    def _check_no_alpha_in_numeric(self, column, column_type):
        """
        Ensure that no alphabetical characters are present in columns classified as numeric.
        """
        column_errors = []
        if column_type == 'numeric':
            # List of special characters to be excluded
            special_chars = ['?', '!',';', '+', '/', '%','@', '&', '"']

            # Define a check for special characters
            def contains_special_chars(s):
                return any(char in s for char in special_chars)

            # Define a check to see if the value is numeric
            def is_not_numeric(s):
                try:
                    float(s)  # Attempt to convert to float
                    return False  # It's numeric, no problem here
                except ValueError:
                    return True  # It's not numeric

            # Apply checks:
            non_numeric_issues = column.apply(lambda x: isinstance(x, str) and not contains_special_chars(x)) 
            
            if non_numeric_issues.any():
                column_errors.append(("Non-numeric chars found in numeric column", column[non_numeric_issues].tolist()))
        return column_errors


    def _get_column_nature(self, column):
        """
        Determine the nature of the column based on the type of data it predominantly holds.
        """
        if column.dtype.kind in 'biufc':
            return 'numeric'
        elif column.dtype.name == 'category':
            numeric_count = sum(pd.to_numeric(column.cat.categories, errors='coerce').notna())
            non_numeric_count = sum(pd.to_numeric(column.cat.categories, errors='coerce').isna())
            if numeric_count / (numeric_count + non_numeric_count) > 0.5:
                return 'numeric'
            else:
                return 'non_numeric'
        return 'mixed'
    
# --------------------------------------------------------- USER RULES ------------------------------------------------------------- #

    def _check_data_type(self, column, rules):
        """ Check if data types in a column match the expected types in rules. """
        column_errors = []
        if 'data_type' in rules:
            expected_dtype = rules['data_type']
            if not isinstance(column.dtype, expected_dtype):
                error_message = f"Expected datatype {expected_dtype}, found {column.dtype}"
                column_errors.append((error_message, column.tolist()))
        return column_errors

    def _check_uniqueness(self, column, rules):
        """ Check if values in a column are unique if required by rules. """
        column_errors = []
        if 'unique' in rules and rules['unique']:
            if column.duplicated().any():
                column_errors.append(("Duplicate values found", column[column.duplicated()].tolist()))
        return column_errors


    def _check_range(self, column, rules, column_type):
        """
        Check if values in a numeric or date column fall within a specified range,
        but only if 'min' or 'max' are explicitly provided in the rules and only for 'numeric' or 'date' column types.
        """
        # Skip if range is not defined or column type is inappropriate
        if 'min' not in rules and 'max' not in rules:
            logging.info(f"No range constraints provided for column: {column.name}")
            return []
        if column_type not in ['numeric', 'date']:
            logging.info(f"Skipping range check for non-numeric/date column: {column.name}")
            return []

        column_errors = []

        try:
            if column_type == 'numeric':
                # Convert column to numeric, safely ignoring non-numeric characters
                numeric_column = pd.to_numeric(column, errors='coerce')
                min_val = rules.get('min', numeric_column.min())
                max_val = rules.get('max', numeric_column.max())
                range_issues = (numeric_column < min_val) | (numeric_column > max_val)
            elif column_type == 'date':
                # Convert column to datetime, handling date format cleanly
                date_column = pd.to_datetime(column, errors='coerce')
                min_val = pd.to_datetime(rules.get('min', date_column.min()))
                max_val = pd.to_datetime(rules.get('max', date_column.max()))
                range_issues = (date_column < min_val) | (date_column > max_val)

            # Collect any rows where the range check fails
            if range_issues.any():
                column_errors.append(("Range check failed", column[range_issues].tolist()))
        except Exception as e:
            logging.error(f"Error processing range check for column {column.name}: {e}")
            column_errors.append(("Error processing range check", str(e)))

        return column_errors


    def _check_contains(self, column, rules):
        """
        Check if each value in the column contains a specified substring.

        Parameters:
        - column (pd.Series): The column to be checked.
        - rules (dict): Dictionary containing the 'contains' key with the substring as value.

        Returns:
        - list: A list containing tuples of error messages and the list of values that failed the validation.
        """
        column_errors = []
        substring = rules.get('contains')

        # Skip if no substring is specified
        if substring is None:
            #logging.info("No substring specified for checking.")
            return column_errors

        try:
            # Convert column to string and check if it contains the specified substring
            # We use case=False to make the check case-insensitive
            contains_issue = ~column.astype(str).str.contains(substring, case=False, na=False, regex=False)

            # Collect values that do not contain the substring
            if contains_issue.any():
                error_values = column[contains_issue].tolist()
                column_errors.append((f"Missing substring '{substring}'", error_values))

        except Exception as e:
            logging.error(f"Error during substring check: {e}")
            column_errors.append(("Error during substring check", str(e)))

        return column_errors
