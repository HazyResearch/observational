# -*- coding: utf-8 -*-

import sys
import csv
from pprint import pprint
import datetime

from scipy.fftpack import ss_diff


def main():
    # read csv exclude none
    exp_reader = get_csv_reader('exp/test_beautified.csv')
    exp_timestamp_reader = get_csv_reader('exp/test_timestamp.csv')
    column = exp_reader.pop(0)
    # process_csv
    process_csv(exp_reader, exp_timestamp_reader)


def process_csv(exps, timestamps):
    # get datetime to split data
    time_by_image: list[datetime.datetime] = get_datetime_objects(
        [x[0] for x in timestamps])
    exp_date = time_by_image[0].strftime('%Y-%m-%d')
    time_exp = get_datetime_objects_from_date_time(exps, 1, 0, exp_date)


# start time + elapsed time
#  ['764.82', '22:35:39.180', '1080', '1920', '1308', '259']
def get_datetime_objects_from_date_time(datas: list, start_time_index: int, elapsed_time_index: int, date: str, format: str = '%Y-%m-%d %H:%M:%S.%f'):
    time_strs = []
    for data in datas:
        date_time = datetime.datetime.strptime(
            "{0} {1}".format(date, data[start_time_index]), format)
        sec, milsec = [0, 0]
        if '.' in data[elapsed_time_index]:
            sec, milsec = [int(x) for x in data[elapsed_time_index].split('.')]
        else:
            sec = int(data[elapsed_time_index])
        date_time += datetime.timedelta(seconds=sec, milliseconds=milsec)
        time_strs.append(date_time)
    return time_strs


def get_datetime_objects(datas: list, format='%Y-%m-%d %H:%M:%S.%f'):
    objects = list()
    for index, data in enumerate(datas):
        try:
            objects.append(datetime.datetime.strptime(data, format))
        except:
            print("Error index: %d, data: %s" % (index, data))
            continue
    return objects


def get_csv_reader(filename):
    exp_file = open(filename, 'r')
    return [x for x in csv.reader(exp_file) if len(x) != 0]


if __name__ == '__main__':
    main()
