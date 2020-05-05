import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


# 去分配每个多进程核心处理的数据段 根据时间顺序分割
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# 把每一次滑动的array输出到queue里面
def queue_wrapper(queue, f, index, *args):
    queue.put((f(*args), index))

