from glv_train_online import *
from LoadData import LoadData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patient arguments')
    parser.add_argument('--number', '-n', default=1, help='Patient No., int type, requested', type=int)
    parser.add_argument('--t', default=0, type=int)
    parser.add_argument('--patient', default='patient001', help='Patient No., str type, requested', type=str)
    parser.add_argument('--g_mode', '-m', default='online', help='Setting online or offline learning', type=str)
    args = parser.parse_args()
    i = args.number

    if len(str(i)) == 1:
        args.patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        args.patientNo = "patient0" + str(i)
    else:
        args.patientNo = "patient" + str(i)
    print(args.patientNo)
    mk_dir(args.patientNo, glv_dir='./analysis-dual-sigmoid/online')
    alldata = LoadData().Double_Drug()
    glv_train_online(args, alldata[args.patientNo])

