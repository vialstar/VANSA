import argparse

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-batch_size', type=int, default=1024)
    arg.add_argument('-margin', type=float, default=6.0)
    arg.add_argument('-dim', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.001)
    arg.add_argument('-lrd', type=float, default=0.001)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, default=0)
    arg.add_argument('-adv_num', type=int, default=1)
    arg.add_argument('-disen_weight', type=float, default=0.01)
    arg.add_argument('-miss_type', type=str, default=None)
    arg.add_argument('-miss_prop', type=float, default=None)
    
    # 新增的参数
    arg.add_argument('--attack_types', type=str, nargs='+', default=['FGSM', 'PGD', 'CW'], help='List of attack methods')
    arg.add_argument('--epsilon_types', type=float, nargs='+', default=[0.01, 0.03, 0.1], help='List of epsilon values')
    arg.add_argument('--attack_iters_types', type=int, nargs='+', default=[10, 20, 30], help='List of attack iteration counts')
    arg.add_argument('--step_size_types', type=float, nargs='+', default=[0.001, 0.005, 0.01], help='List of step sizes')
    
    # 策略网络的学习率
    arg.add_argument('--policy_model_lr', type=float, default=0.01, help='Learning rate for the policy network')

    return arg.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)