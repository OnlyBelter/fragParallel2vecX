import json


def write_list_by_json(a_list, sep='\t'):
    return sep.join([json.dumps(i) for i in a_list]) + '\n'


def read_json_line(a_line, sep='\t'):
    """
    read json file line by line
    :param a_line: one line
    :param sep:
    :return: a list separated by '\t'
    """
    a_line = a_line.strip().split(sep)
    return [json.loads(i) for i in a_line]
