import re


def replace_integer(text, new_integer):
    pattern = r"(\D*)(\d+)$"
    match = re.match(pattern, text)

    if match:
        prefix = match.group(1)
        updated_text = prefix + str(new_integer)
        return updated_text
    else:
        return text


original_text = "# NCOLS_ODD: 254"
new_integer = 20
updated_text = replace_integer(original_text, new_integer)
print(updated_text)