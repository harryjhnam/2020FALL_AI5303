import json

class Params():

    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save_json(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
