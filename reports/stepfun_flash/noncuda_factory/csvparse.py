def parse_csv(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    row: list[str] = []
    field: list[str] = []
    i = 0
    in_quotes = False
    n = len(text)

    while i < n:
        ch = text[i]

        if in_quotes:
            if ch == '"':
                # Look ahead: "" is an escaped quote, lone " ends quoting
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                else:
                    in_quotes = False
                    i += 1
                    continue
            else:
                field.append(ch)
                i += 1
                continue
        else:
            # Not in quotes
            if ch == '"':
                in_quotes = True
                i += 1
                continue
            elif ch == ',':
                row.append(''.join(field))
                field = []
                i += 1
                continue
            elif ch == '\n':
                row.append(''.join(field))
                rows.append(row)
                row = []
                field = []
                i += 1
                continue
            elif ch == '\r':
                # Handle \r\n as a single newline
                if i + 1 < n and text[i + 1] == '\n':
                    i += 1  # skip the \n in the next iteration
                row.append(''.join(field))
                rows.append(row)
                row = []
                field = []
                i += 1
                continue
            else:
                field.append(ch)
                i += 1
                continue

    # End of string: flush last field and last row
    field_str = ''.join(field)
    if in_quotes:
        # Unterminated quote — still include what we have
        pass
    row.append(field_str)
    rows.append(row)

    return rows
