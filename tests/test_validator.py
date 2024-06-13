
import unittest
import pandas as pd
from data_validator.validator import DataValidator


class TestDataValidator(unittest.TestCase):

    def setUp(self):
        # This method will be run before each test
        data = {
            'age': [25, -1, 110, 'unknown', 50, 27, 45, 68, 'thirty', 21],
            'email': ['john@example.com', 'jane..doe@example.com', 'not-an-email', '', 'alice@example.com', 'bob@', 'chris@site', 'david@web.com', 'eve@internet.com', 'frank@domain..com'],
            'height': [5.9, 6.1, 'six feet', None, 5.5, 5.8, 5.3, '5,7', '5.4ft', '170cm'],
            'weight': [160, 180, '80kg', 140, 150, '150 lbs', 'seventy', 130, 125, 'unknown'],
            'join_date': ['01-01-2020', '2020/02/02', 'March 5, 2019', '04/04/18', '2019-05-15', 'June 06, 2016', '07.07.2017', 'August 8 2018', '9/9/2019', '10-10-2020'],
            'rating': [4.5, 4.7, '4.8', 5.0, 'five', '4,2', 4.9, '4.6', None, 4.3],
            'status': ['Active', 'active', 'Inactive', 'ACTIVE', 'inactive', 'active', 'Active', 'active', 'Active', 'INACTIVE'],
            'department': ['Sales', 'Engineering', 'HR', 'sales', 'Hr', 'Engineering', 'hr', 'Sales', 'SALES', 'engineering'],
            'employee_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
            'session_time': ['30:00', '25:50', 'one hour', '20:30', '15:45', '50:00', '40:20', '35:35', '30:30', '25:25'],
            'country_code': ['US', 'CA', 'MX', 'us', 'ca', 'mx', 'Us', 'Ca', 'Mx', 'USA'],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '192.168.0.', '256.300.1.1', '192.168.0.256', '10.0.0.0/24', '192.168.1.256', '172.16.0.300', '10.10.10.10']
        }


        self.df = pd.DataFrame(data)
        self.validator = DataValidator(self.df, {}, auto_validate=False)

    def test_implicit_validation(self):
        # Execute validation
        self.validator.validate(show=True)
        # Display failed checks based on the `error_dfs`
        if self.validator.error_dfs:  # Check if there are any errors at all
            for error_type, error_df in self.validator.error_dfs.items():
                print(f"Error Type: {error_type}, Errors: {error_df}")
            self.fail(f"Validation errors found: {list(self.validator.error_dfs.keys())}")
        else:
            self.assertTrue(True, "No validation errors found.")

    def test_check_mixed_proposed_type(self):
        # Execute validation
        self.validator.validate(show=False)
        # Iterate through all error dataframes
        for error_type, error_df in self.validator.error_dfs.items():
            # Check if any 'Proposed' column entry is 'mixed'
            if (error_df['Proposed'] == 'mixed').any():
                self.fail(f"Column proposed as 'mixed' found in error type: {error_type}")

if __name__ == '__main__':
    unittest.main()

