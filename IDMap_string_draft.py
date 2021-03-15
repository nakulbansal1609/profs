def string_IDmap(value):
    byt_value = value.encode('utf-8')
    int_value = int.from_bytes(byt_value, 'little')
    int_value = int(str(int_value)[:11])
    return int_value



