import json
import sys

def read_file(file_path):
    
    with open(file_path) as f:
        lines = f.read()

        js = json.loads(lines)
        idx = 0

        for key in js.keys():
            print(json.dumps({"idx": idx, "target": 0, "func": js[key]["func_after"]}))
            idx += 1
            print(json.dumps({"idx": idx, "target": 1, "func": js[key]["func_before"]}))
            idx += 1
        
if (len(sys.argv) < 2):
    print("file name is required as cmd line arg")
    exit(1)

file_name = sys.argv[1]
read_file(file_name)


