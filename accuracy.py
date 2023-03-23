import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--iid', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='fashion-mnist')
    parser.add_argument('--model', type=str, default='cnn')

    args = parser.parse_args()

    rootpath = './log'

    # full participation
    full_acc = []
    full_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
                        format(args.dataset, args.model, args.epochs, 1.0, args.iid, 'full'), 'r')
    for acc in full_accfile.readlines():
        full_acc.append(float(acc))
    full_accfile.close()

    # random sampling
    rand_acc = []
    rand_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
                        format(args.dataset, args.model, args.epochs, args.frac, args.iid, 'full'), 'r')
    for acc in rand_accfile.readlines():
        rand_acc.append(float(acc))
    rand_accfile.close()

    # power-of-choice
    power_acc = []
    power_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
                         format(args.dataset, args.model, args.epochs, 1.0, args.iid, 'power-of-choice'), 'r')
    for acc in power_accfile.readlines():
        power_acc.append(float(acc))
    power_accfile.close()

    # ideal
    ideal_acc = []
    ideal_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
                         format(args.dataset, args.model, args.epochs, 0.3, args.iid, 'ideal'), 'r')
    for acc in ideal_accfile.readlines():
        ideal_acc.append(float(acc))
    ideal_accfile.close()

    # # practical
    # practical_acc = []
    # prac_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
    #                     format(args.dataset, args.model, args.epochs, 1.0, args.iid, 'practical'), 'r')
    # for acc in prac_accfile.readlines():
    #     practical_acc.append(float(acc))
    # prac_accfile.close()

    # # divfl
    # divfl_acc = []
    # divfl_accfile = open(rootpath + '/accfile_fed_{}_{}_{}_C{}_iid{}_{}.dat'.
    #                      format(args.dataset, args.model, args.epochs, 1.0, args.iid, 'divfl'), 'r')
    # for acc in divfl_accfile.readlines():
    #     divfl_acc.append(float(acc))
    # divfl_accfile.close()

    plt.figure()
    plt.plot(range(len(full_acc)), full_acc, linestyle='--', label='Full')
    plt.plot(range(len(rand_acc)), rand_acc, label='Random')
    plt.plot(range(len(power_acc)), power_acc, label='Power-of-choice')
    # plt.plot(range(len(divfl_acc)), divfl_acc, label='DivFL')
    plt.plot(range(len(ideal_acc)), ideal_acc, label='Ours (ideal)')
    # plt.plot(range(len(practical_acc)), practical_acc, label='Ours (practical)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_iid{}_acc.png'.format(args.dataset, args.model, args.epochs, args.iid))