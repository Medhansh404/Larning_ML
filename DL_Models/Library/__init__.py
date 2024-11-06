import argparse as arg


# input: the input i form an array
# opt: optimiser
#
def parse():
    parser = arg.ArgumentParser(description='Welcome')
    parser.add_argument('numbers', type=list, help="An integer array")
    args = parser.parse_args()
    l_arg = args.numbers
    l = l_arg[0]
    input = l_arg[1:l_arg[0]]
    opt = l_arg[l + 1]
    eta = l_arg[l + 2]
    epochs = l_arg[l + 3]
    b_size = l_arg[-1]
    return input, opt, eta, epochs, b_size


if __name__ == "__main__":
    inp, opt, e, epochs, bs = parse()
    print(inp, opt, e, epochs, bs)
