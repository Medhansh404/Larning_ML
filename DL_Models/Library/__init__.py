import argparse as arg
import Optimisers as optim

# input: the input i form an array
# opt: optimiser
#

def parse_arg():
    parser = arg.ArgumentParser(description='Welcome')
    parser.add_argument(
        'numbers',
        type=float,
        nargs='+',
        help="A number array"
    )
    args = parser.parse_args()
    l_arg = args.numbers
    l = int(l_arg[0])
    input = l_arg[1:2*l+1]
    opt = l_arg[2*l + 1]
    eta = l_arg[2*l + 2]
    epochs = l_arg[2*l + 3]
    b_size = l_arg[-1]
    return input, opt, eta, epochs, b_size


if __name__ == "__main__":
    inp, opt, e, epochs, bs = parse_arg()
    # we got the input array now
    size = len(inp)
    x = inp[:(size//2)+1]
    y = inp[size//2:size]
    model = optim.main()