
def split_file(filename):
    with open(filename) as f:
        file = f.read()
        lines = file.split("\n")
        for line in lines:
            names = line.split(" ")
            for name in names:
                print(name)

split_file("case_files")
