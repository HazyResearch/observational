# -*- coding: utf-8 -*-

import sys

def main(lines):
    nums = [int(x) for x in list(lines.pop(0))]
    print(''.join([str(x)for x in nums]))
     # 最初の数とか
    n, m = [int(x) for x in lines.pop(0).split()]
    values = lines.pop(0).split(' ')
    int_values = [int(x) for x in lines.pop(0).split()]

    # Dynamic Programming
    dp = [[0]*(n) for _ in range(n)]
    dp = [[0]*(n) for _ in range(m)]
    dp = [['0']*(n) for _ in range(n)]
    dp = [['0']*(n) for _ in range(m)]
    
    # アルファベット全部
    alpha = [chr(x) for x in range(97, 123)]

    for i in range(n):
        dp_copy = dp[i].copy()


if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)