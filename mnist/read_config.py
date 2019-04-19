def delete_last_empty_line(s):
    end_index = len(s) - 1
    while(end_index >= 0 and (s[end_index] == "\n" or s[end_index] == "\r")):
        end_index -= 1
    s = s[:end_index + 1]
    return s

def read_file(file_name):
    with open(file_name, "r") as f:
        s = f.read()
        s = delete_last_empty_line(s)
        s_l = s.split("\n")
        for i, l in enumerate(s_l):
            if l.endswith("\r"):
                s_l[i] = s_l[i][:-1]
    return s_l

def check_bool(val):
    if val == 'true' or val == "True":
        val = True
    if val == 'false' or val == "False":
        val = False
    return val

def check_digit(val):
    if val.replace(".","").isdigit():
        val = float(val)
        if val.is_integer():
            val = int(val)
    return val  

def check_list(val):
    if isinstance(val, str):
        if len(val) > 1 and val[0] == '[':
            val = val[1:-1]
            val = val.split(',')
            for i in range(len(val)):
                val[i] = check_bool(val[i])
                val[i] = check_digit(val[i])
    return val

def check_none(val):
    if isinstance(val, str):
        if val == "None" or val == "none":
            return None
    return val

def read(file_name):
    model_param = dict()
    param = read_file(file_name)
    for li in param:
        li = li.replace(" ", "")
        if len(li) == 0 or li[0] == "#":
            continue
        name, val = li.split(":")

        val = check_digit(val)
        val = check_bool(val)
        val = check_list(val)
        val = check_none(val)

        model_param[name] = val

    return model_param
