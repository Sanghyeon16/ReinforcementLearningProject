import os

def abs_path(xml_file):
    path = os.path.dirname(__file__)
    path = os.path.join(path, xml_file)
    return path
