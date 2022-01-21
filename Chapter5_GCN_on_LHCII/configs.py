import  argparse

args = argparse.ArgumentParser()
args.add_argument('--seed', type=int, default= 1)
args.add_argument('--best_epoch_num', type=int, default= 0)
args.add_argument('--best_loss_valid', type=float, default= 1e10)
args.add_argument('--earlystopping', type=bool, default=True)
args.add_argument('--patient', type=int, default= 100)
args.add_argument('--prefix', type = str, default = "gcn_net_trained_12_16batched_attn_MSELoss")
args.add_argument('--batch_size', type=int, default= 16)
args.add_argument('--max_epochs', type=int, default= 500)

args.add_argument('--hid1', type=int, default=32)
args.add_argument('--hid2', type=int, default=64)
args.add_argument('--hid3', type=int, default=128)

args = args.parse_args()
print(args)