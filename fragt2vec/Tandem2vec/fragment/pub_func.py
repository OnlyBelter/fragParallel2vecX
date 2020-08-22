import json


def write_list_by_json(a_list, sep='\t'):
    return sep.join([json.dumps(i) for i in a_list]) + '\n'
