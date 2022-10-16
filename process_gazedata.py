# -*- coding: utf-8 -*-

import sys
import os
import csv
from pprint import pprint
import datetime


def main():
    datas = list()
    with open('exp/test.tsv', encoding='utf-8', newline='') as f:
        _datas = list(csv.reader(f, delimiter='\t'))
        column = [_datas[0][0], _datas[0][4]] + ['height', 'width'] + _datas[0][8:10]
        start_time = _datas[1][4]
        height, width = _datas[1][6:8]
        for data in _datas[1:]:
            if '' in data:
                datas.append(None)
            else:
                datas.append({"time":round((float)(data[0])/1000, 2), "xy":[(int)(x) for x in data[8:10]]})
    """
        print(column)
        print("Start time:", start_time)
        print("height:", height)
        print("width:", width)
        pprint(datas[-100:-50])
    """
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_file = open(time + '_beautified' + '.csv', 'w')
    writer = csv.writer(csv_file)
    writer.writerow(column)
    for data in datas:
        if data is None:
            writer.writerow('')
        else:
            writer.writerow([data['time'], start_time, height, width] + data['xy'])


if __name__ == '__main__':
    main()
