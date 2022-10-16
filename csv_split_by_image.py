# -*- coding: utf-8 -*-

import sys
import csv
from pprint import pprint
import datetime


def main():
    # read csv exclude none
    exp_reader = get_csv_reader('exp/test_beautified.csv')
    exp_timestamp_reader = get_csv_reader('exp/test_timestamp.csv')
    column = exp_reader.pop(0)
    # process_csv
    process_csv(exp_reader, exp_timestamp_reader)


def process_csv(exps, timestamps):
    # get datetime to split data
    timestamps_by_image = get_datetime_objects(
        [[x[0], x[1]] for x in timestamps])
    exp_date = timestamps_by_image[0][1].strftime('%Y-%m-%d')
    timestamps_exp = get_datetime_objects_from_date_time(exps, 2, 1, exp_date)
    # get beautified timestamp data
    exps, timestamps_exp, timestamps_by_image = remove_unnecessary_timestamps(
        exps, timestamps_exp, timestamps_by_image)
    make_csv_split_by_image(timestamps_exp, exps,
                            timestamps_by_image, timestamps)


# make csv of timestamps which is splited by the image
def make_csv_split_by_image(timestamps_exp, datas_of_exp, timestamps_by_image, datas_of_image):
    for index, data_of_image in enumerate(datas_of_image):
        # get filename
        filename = data_of_image[-1].replace('.', '_')
        start_time = timestamps_by_image[index][1]
        print(filename)
        with open('csv/' + filename + '.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ['time', 'height', 'width', 'Gaze point X', 'Gaze point Y'])
            next_start_time = timestamps_by_image[index + 1][1]
            
            '''
            pprint(timestamps_exp[0:5])
            pprint(datas_of_exp[0:5])
            print('')
            pprint(timestamps_exp[-5:])
            pprint(datas_of_exp[-5:])
            '''

            # add data while do not exceed next start time
            while True:
                # [45854, datetime.datetime(2022, 10, 15, 22, 48, 24, 50000)]]
                # [45850, '764.8', '22:35:39.180', '1080', '1920', '1311', '257']
                if len(timestamps_exp) == 0:
                    return
                if timestamps_exp[0][1] < next_start_time:
                    time = str(timestamps_exp.pop(0)[1])
                    height, width, X, Y = datas_of_exp.pop(0)[3:]
                    csv_writer.writerow([time, height, width, X, Y])
                else:
                    break

# remove unnecessary data like calibration


def remove_unnecessary_timestamps(exps, exp_timestamps, image_timestamps):
    # remove data which is prior to exp
    while True:
        if exp_timestamps[0][1] < image_timestamps[0][1]:
            #print("remove {}".format(exp_timestamps.pop(0)))
            "remove {}".format(exp_timestamps.pop(0))
            exps.pop(0)
        else:
            break
    '''
    pprint(exp_timestamps[0:10])
    print('')
    pprint(image_timestamps[0:2])
    print('')
    '''
    # remove data which is later than exp
    end_time = image_timestamps[-1][1] + datetime.timedelta(seconds=6)
    while True:
        if exp_timestamps[-1][1] > end_time:
            #print("remove {}".format(exp_timestamps.pop(-1)))
            "remove {}".format(exp_timestamps.pop(-1))
            exps.pop(-1)
        else:
            break
    '''
    pprint(exp_timestamps[-11:-1])
    print('')
    pprint(image_timestamps[-2:])
    print('')
    '''
    image_timestamps.append([-1, end_time])
    return exps, exp_timestamps, image_timestamps


# start time + elapsed time
# exsample ['764.82', '22:35:39.180', '1080', '1920', '1308', '259']
def get_datetime_objects_from_date_time(datas: list, start_time_index: int, elapsed_time_index: int, date: str, format: str = '%Y-%m-%d %H:%M:%S.%f'):
    time_strs = []
    for data in datas:
        # calc exptime -> start time + elapsed time(milliseconds)
        date_time = datetime.datetime.strptime(
            "{0} {1}".format(date, data[start_time_index]), format)
        milsec = int(float(data[elapsed_time_index]) * 1000)
        date_time += datetime.timedelta(milliseconds=milsec)
        time_strs.append([data[0], date_time])
    return time_strs


def get_datetime_objects(datas: list, format='%Y-%m-%d %H:%M:%S.%f'):
    objects = list()
    for data in datas:
        try:
            objects.append(
                [data[0], datetime.datetime.strptime(data[1], format)])
        except:
            print("Error index: %d, data: %s" % (data[0], data[1]))
            continue
    return objects


def get_csv_reader(filename):
    exp_file = open(filename, 'r')
    # include CSV index
    return [[index+1] + x for index, x in enumerate(csv.reader(exp_file)) if len(x) != 0]


# debug
def check_list(time_exp, time_by_image):
    print(time_by_image[0])
    pprint(time_exp[0:10])
    print('')
    print(time_by_image[-1])
    pprint(time_exp[-10:])


if __name__ == '__main__':
    main()
