def is_numeric(s):
    # Check if the string contains only numeric characters (0-9)
    return all(char.isdigit() for char in s)

def is_alphanumeric(s):
    # Check if the string contains only alphanumeric characters (letters and digits)
    return s.isalnum()

def test_string_type(s):
    if is_numeric(s):
        print(f'The string "{s}" contains only numeric data.')
    elif is_alphanumeric(s):
        print(f'The string "{s}" contains alphanumeric data.')
    else:
        print(f'The string "{s}" does not contain only numeric or alphanumeric data.')

# Test cases
test_string_type("12345 34.5")        # Output: The string "12345" contains only numeric data.
test_string_type("Hello123")     # Output: The string "Hello123" contains alphanumeric data.
test_string_type("abc")          # Output: The string "abc" contains alphanumeric data.
test_string_type("Hello, World")