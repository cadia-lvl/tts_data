

def dp_to_file(src_path:str, out_path:str):
    '''
    Given a path to a source file with G2P tokens
    in the standard format, append the diphones as the
    4th column in each line in a new file at out_path

    Input arguments:
    * src_path (str): The path to the G2P tokens
    * out_path (str): The target
    '''