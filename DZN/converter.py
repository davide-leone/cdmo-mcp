import sys

def read_input_file(input_file):
    with open(input_file, 'r') as file:
        data = file.readlines()

    # Read number of couriers (m) and items (n)
    m = int(data[0].strip())
    n = int(data[1].strip())

    # Read maximum load sizes for each courier (l)
    l = list(map(int, data[2].strip().split()))

    # Read sizes of the items (s)
    s = list(map(int, data[3].strip().split()))

    # Read distance matrix (D)
    D = []
    for i in range(4, 4 + (n + 1)):
        D.append(list(map(int, data[i].strip().split())))

    return m, n, l, s, D

def write_dzn_file(output_file, m, n, l, s, D):
    with open(output_file, 'w') as file:
        file.write(f"m = {m};\n")
        file.write(f"n = {n};\n")

        file.write("l = [")
        file.write(", ".join(map(str, l)))
        file.write("];\n")

        file.write("s = [")
        file.write(", ".join(map(str, s)))
        file.write("];\n")

        file.write("D = [|\n")
        for row in D:
            file.write(", ".join(map(str, row)))
            file.write("|\n")
        file.write("];\n")

def main(input_file, output_file):
    m, n, l, s, D = read_input_file(input_file)
    write_dzn_file(output_file, m, n, l, s, D)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python converter.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)