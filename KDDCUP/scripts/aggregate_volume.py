# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Calculate volume for each 20-minute time window.
"""
import math
from datetime import datetime,timedelta


def avgVolume(in_file, out_file, path):

    # Step 1: Load volume data
    fr = open(path + in_file, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to caculate and store volume per time window
    volumes = {}  # key: time window value: dictionary
    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        tollgate_id = each_pass[1]
        direction = each_pass[2]
        car_mod = each_pass[3]  #
        #print int(car_mod)
        has_etc = each_pass[4]  #

        pass_time = each_pass[0]
        pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
        #print pass_time
        start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)

        if start_time_window not in volumes:
            volumes[start_time_window] = {}
        if tollgate_id not in volumes[start_time_window]:
            volumes[start_time_window][tollgate_id] = {}
        if direction not in volumes[start_time_window][tollgate_id]:
            #volumes[start_time_window][tollgate_id][direction] = 1
            volumes[start_time_window][tollgate_id][direction] = {}
            volumes[start_time_window][tollgate_id][direction]['volume'] = 0
            volumes[start_time_window][tollgate_id][direction]['has_etc'] = 0
            volumes[start_time_window][tollgate_id][direction]['car_mod'] = \
            [0, 0, 0, 0, 0, 0, 0, 0] # vehicle model range 0~7

        volumes[start_time_window][tollgate_id][direction]['volume'] += 1
        volumes[start_time_window][tollgate_id][direction]['has_etc'] += int(has_etc)
        volumes[start_time_window][tollgate_id][direction]['car_mod'][int(car_mod)] += 1

    # Step 3: format output for tollgate and direction per time window
    fw = open(path + out_file, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', 
        '"volume"', '"has_etc"', '"car_model"']) + '\n')
    time_windows = list(volumes.keys())
    time_windows.sort()
    for time_window_start in time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        for tollgate_id in volumes[time_window_start]:
            for direction in volumes[time_window_start][tollgate_id]:
               out_line = ','.join(['"' + str(tollgate_id) + '"', 
			                     '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(direction) + '"',
                                 '"' + str(volumes[time_window_start][tollgate_id][direction]['volume']) + '"',
                                 '"' + str(volumes[time_window_start][tollgate_id][direction]['has_etc']) + '"',
                                 '"' + str(volumes[time_window_start][tollgate_id][direction]['car_mod']) + '"',
                               ]) + '\n'
               fw.writelines(out_line)
    fw.close()

def main():
    import ipdb
    #ipdb.set_trace()
    #path = '../dataSets/training/'  # set the data directory
    #in_file = 'volume(table 6)_training.csv'
    path = '../dataSets/testing_phase1/'
    in_file = 'volume(table 6)_test1.csv'
    out_file = 'test_20min_avg_volume.csv'
    avgVolume(in_file, out_file, path)

if __name__ == '__main__':
    main()



