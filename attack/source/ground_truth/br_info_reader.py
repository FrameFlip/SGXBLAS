
def read_br_info() -> int:
    br_idx: int = 0
    with open("./br_info.tmp") as info_f:
        for line in info_f:
            if line[0] != ' ':
                br_idx = int(line.strip().split('\t')[0])

    return br_idx


def main():
    print(read_br_info())


if __name__ == "__main__":
    main()