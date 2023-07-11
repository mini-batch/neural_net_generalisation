import json

def save_json(obj, path : str) -> None:
    with open(path, "w") as outfile:
        outfile.write(json.dumps(obj, indent=4))

def load_json(path : str) -> dict:
    with open(path, "r") as infile:
        return json.load(infile)
