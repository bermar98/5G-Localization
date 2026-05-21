import json
from datetime import datetime


class Storage:
    def __init__(self, filepath):
    
        self.filepath = filepath

    def save(self, fingerprint, position):

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "position": position,
            "fingerprint": fingerprint
        }

        try:

            with open(self.filepath, "r", encoding="utf-8") as file:

                data = json.load(file)

        except (FileNotFoundError, json.JSONDecodeError):

            data = []

        data.append(entry)

        with open(self.filepath, "w", encoding="utf-8") as file:

            json.dump(data, file, indent=4)