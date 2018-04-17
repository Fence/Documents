#coding:utf-8
import re
import os
import pdb
import argparse
import numpy as np
from tqdm import tqdm

class DataProcessing():
    def __init__(self, args):
        self.save_file = args.save_file
        self.refine_val = args.refine_val
        self.mov_time_win = args.mov_time_win
        self.data_dir = '../dataSets/'
        self.train_volume_file = 'training/training_20min_avg_volume_new.csv'
        self.train_weather_file = 'training/weather (table 7)_training_update.csv'
        self.links_file = 'training/links (table 3).csv'
        self.routes_file = 'training/routes (table 4).csv'
        self.train_time_file = 'training/training_20min_avg_travel_time.csv'
        self.test_volume_file = 'testing_phase1/test_20min_avg_volume.csv'
        self.test_weather_file = 'testing_phase1/weather (table 7)_test1.csv'
        
        self.test_day = 12
        self.std_pre = 10  # standard presure: 1013.25 Pa
        self.com_tmp = 1   # ordinary temperature: 25.0 C
        self.com_rh = 1    # ordinary humidity: 40.0%
        self.wind_dirt = 8 # wind direction: 8
        self.wind_rank = 1 # wind speed rank: 2.0m/s
        self.rain_rank = 1 # rain rank unit: 25.0mm
        self.day_bits = 5       # 3
        self.itv_bits = 18      # 4
        self.gate_dirt_bits = 5 # 4
        self.lanes_bits = 3     # 3
        #self.car_mod_bits = 8  # 8
        #self.has_etc_bits = 3  # 3
        self.day_hot = [3, 5, 7]
        self.h_hot = 11
        self.m_hot = 7
        self.win_bits = 5   # bits of time step of 2 hours
        self.hist_v_bits = 6 + self.mov_time_win*self.win_bits   # 6
        self.wter_bits = [1, 1, 2, 3, 3, 2, 5] #
        
        self.split_bit = self.day_bits + self.itv_bits + self.gate_dirt_bits + self.lanes_bits + \
                        self.hist_v_bits #self.car_mod_bits + self.has_etc_bits + 
        self.weekend = [(9,24), (9,25), (10,15), (10,16), (10, 22), (10, 23)]
        self.holiday = [(9,15), (9,16),(9,17), (10,1), (10,2), (10,3), \
                        (10,4), (10,5), (10,6), (10,7)]
        self.lanes = [self.repeat_coding(2, self.lanes_bits), \
                        self.repeat_coding(2, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits)] 
        self.test_v_data = self.get_hist_volume(self.test_volume_file, 0)
        self.test_w_data = self.get_weather(self.test_weather_file)


    def k_hot(self, num, length, k=1): # similar to one-hot
        code = np.zeros(length)
        code[num] = k
        return list(code)

    def repeat_coding(self, value, length):
        code = [value for i in xrange(length)]
        return code

    def padding_data(self, item, tg, itv):
        num = len(tg)
        if itv > tg[num-1][2]+1: #假设不存在一天都没有车通过的情况
            for n in xrange(tg[num-1][2]+1, itv):
                tmp = [int(b) for b in item]
                tmp[2] = n
                tg.append(tmp)
        elif itv < tg[num-1][2]: #时间已经到第二天
            for n in xrange(tg[num-1][2]+1, 72):
                tmp = [int(b) for b in item]
                tmp[2] = n
                tg.append(tmp)
            for n in xrange(itv):
                tmp = [int(b) for b in item]
                tmp[2] = n
                tmp[1] -= 1
                if tmp[1] == 1 and tmp[0] == 10:
                    tmp[1] = 30
                    tmp[0] = 9
                tg.append(tmp)


    def get_hist_volume(self, file_name, padding=1):
        tg10 = [] # mm--dd--itv--volume
        tg11 = []
        tg20 = []
        tg30 = []
        tg31 = []
        with open(self.data_dir + file_name) as f:
            titles = f.readline()
            contents = f.readlines()
            for line in contents:
                a = re.findall(r'\d+', line)
                #print len(a), a
                itv = int(a[4])*3 + int(a[5])/20
                item = a[2:4] + [itv] + [a[14]] # mm--dd--itv--volume
                if a[0] == '1' and a[13] == '0':
                    if len(tg10) and padding:
                        self.padding_data(item, tg10, itv)
                    tg10.append([int(b) for b in item])
                elif a[0] == '1' and a[13] == '1':
                    if len(tg11) and padding:
                        self.padding_data(item, tg11, itv)
                    tg11.append([int(b) for b in item])
                elif a[0] == '2' and a[13] == '0':
                    if len(tg20) and padding:
                        self.padding_data(item, tg20, itv)
                    tg20.append([int(b) for b in item])
                elif a[0] == '3' and a[13] == '0':
                    if len(tg30) and padding:
                        self.padding_data(item, tg30, itv)
                    tg30.append([int(b) for b in item])
                else:
                    if len(tg31) and padding:
                        self.padding_data(item, tg31, itv)
                    tg31.append([int(b) for b in item])
        v_data = [tg10, tg11, tg20, tg30, tg31]

        if self.save_file:
            out_file = self.data_dir + file_name.split('/')[0] + '/out_volume.txt'
            print out_file
            with open(out_file, 'w') as fw:
                fw.write('gd, mm, dd, itv, volume\n')
                for j, d in enumerate(v_data):
                    for k in d:
                        fw.write('%d,%s\n'%(j+1, ','.join([str(l) for l in k])))

        return v_data


    def get_weather(self, file_name):
        w_data = {} #(m, d, itv) [p--sp--wd--ws--tp--rh--pr]
        with open(self.data_dir + file_name) as f:
            titles = f.readline()
            contents = f.readlines()
            for i in xrange(len(contents)):
                line = contents[i]
                a = re.findall(r'[\d\.]+', line)
                #print len(a), a
                assert len(a) == 11
                hour = int(a[3])
                for i in xrange(hour*3, hour*3+9):
                    dt = (int(a[1]), int(a[2]), i)
                    w_data[dt] = [float(b) for b in a[4:]]
        if self.save_file:
            out_file = self.data_dir + file_name.split('/')[0] + '/task2_weather2.txt'
            print out_file
            with open(out_file, 'w') as fw:
                fw.write('m--d--itv--p--sp--wd--ws--tp--rh--pr\n')
                for j in w_data.keys():
                    fw.write(str(j)+' '+str(w_data[j])+'\n')

        return w_data


    def update_test_data(self, last_x_dict, y, valid=0):
        next_x = []
        next_x_dict = []
        assert len(last_x_dict) == len(y)
        for d in xrange(len(last_x_dict)):
            x_d = last_x_dict[d]
            hour = x_d['itv'][:12].index(self.h_hot)
            mins = x_d['itv'][12:].index(self.m_hot)
            itv = hour*6 + mins
            # assume that hours (11-hot) is more important than mins (7-hot)
            x_d['itv'] = self.k_hot(itv/6, 12, self.h_hot) + self.k_hot(itv%6, 6, self.m_hot)
            x_d['hist_v'][:-1] = x_d['hist_v'][1:]
            x_d['hist_v'][-1] = y[d][0]
            x_d['time_win'] = [t+1 for t in x_d['time_win']]
            if valid:
                wtr = self.w_data[(x_d['day'][0], x_d['day'][1], itv)]
            else:
                wtr = self.test_w_data[(x_d['day'][0], x_d['day'][1], itv)]
            x_d['weather'] = self.construct_weather(wtr)
            x_d_list = x_d['day'] + x_d['itv'] + x_d['gate_dirt'] + x_d['lanes'] + \
                        x_d['hist_v'] + x_d['time_win'] + x_d['weather']
            next_x_dict.append(x_d)
            next_x.append(np.array(x_d_list))

        return np.array(next_x), next_x_dict


    def init_test_data(self):
        x_test = []
        x_test_dict = []
        raw_data = []
        gd = [[1,0], [1,1], [2,0], [3,0], [3,1]]
        for idx in xrange(5, len(self.test_v_data[0]), 6):
            for tg in xrange(len(self.test_v_data)):
                i = self.test_v_data[tg][idx]
                x_i, x_i_d, hist_v, weather = self.convert_data(
                    self.test_v_data, tg, idx, i, self.test_w_data, 1)
                if not x_i:
                    continue
                x_test.append(np.array(x_i))
                x_test_dict.append(x_i_d)
                tmp_raw = {}
                tmp_raw['gate'] = str(gd[tg][0])
                tmp_raw['dirt'] = str(gd[tg][1])
                tmp_raw['date'] = '2016-' + '%0*d'%(2, i[0]) + '-' + '%0*d'%(2, i[1]) 
                tmp_raw['start_time'] = '%0*d'%(2, i[2]/3) + ':' + '%0*d'%(2, i[2]%3*20) + ':00'
                tmp_raw['end_time'] = '%0*d'%(2, (i[2]+1)/3) + ':' + '%0*d'%(2, (i[2]+1)%3*20) + ':00'
                tmp_raw['itv'] = i[2]
                tmp_raw['hist_v'] = hist_v
                tmp_raw['weather'] = weather
                raw_data.append(tmp_raw)
                #raw_data.append([tg] + i + [hist_v] + [weather])

        x_test = np.array(x_test)

        return x_test, x_test_dict, raw_data



    def combine_train_data(self):
        x_train = []
        y_train = []
        x_train_dict = []
        self.v_data = self.get_hist_volume(self.train_volume_file)
        self.w_data = self.get_weather(self.train_weather_file)
        raw_data = []
        x_val = [[],[],[],[],[]]
        y_val = [[],[],[],[],[]]
        x_val_dict = [[],[],[],[],[]]
        for idx in xrange(6, len(self.v_data[0])):
            for tg in xrange(len(self.v_data)):
                i = self.v_data[tg][idx]
                x_i, x_i_d, hist_v, weather = self.convert_data(self.v_data, tg, idx, i, self.w_data)
                if not x_i:
                    continue
                if i[0] == 10 and i[1] > self.test_day:
                    if self.refine_val:
                        if 24 <= i[2] <= 29 or 51 <= i[2] <= 56: 
                            x_val[tg].append(np.array(x_i))
                            y_val[tg].append(i[-1])
                            x_val_dict[tg].append(x_i_d)
                            #raw_data.append([tg] + i + [hist_v] + [weather])
                    else:
                        x_val[tg].append(np.array(x_i))
                        y_val[tg].append(i[-1])
                        x_val_dict[tg].append(x_i_d)
                        #raw_data.append([tg] + i + [hist_v] + [weather])
                else:
                    x_train.append(np.array(x_i))
                    y_train.append(i[-1])
                    x_train_dict.append(x_i_d)
                raw_data.append([tg] + i + [hist_v] + [weather])

        if self.save_file:
            with open(self.data_dir+'raw_data4.csv', 'w') as f:
                f.write('gd, mm, dd, it, volume, hist_volume, weather\n')
                for rd in raw_data:
                    hv = ' '.join([str(w) for w in rd[-2]])
                    rd[-2] = hv
                    wd = ' '.join([str(w) for w in rd[-1]])
                    rd[-1] = wd
                    f.write(','.join([str(r) for r in rd])+'\n')

            with open(self.data_dir+'xy_train4.txt', 'w') as f:
                f.write('day, itv, gate-dirt, lanes, hist_v, weather, volume\n')
                for i in xrange(len(x_train)):
                    for j in xrange(len(x_train[i])):
                        if j < self.split_bit:
                            f.write('%d, '%int(x_train[i][j]))
                        else:
                            f.write('%s, '%str(x_train[i][j]))
                    f.write('%d\n'%int(y_train[i]))

            with open(self.data_dir+'xy_test3.txt', 'w') as f:
                f.write('day, itv, gate-dirt, lanes, hist_v, weather, volume\n')
                for i in xrange(len(x_val)):
                    for j in xrange(len(x_val[i])):
                        for k in xrange(len(x_val[i][j])):
                            if k < self.split_bit:
                                f.write('%d, '%int(x_val[i][j][k]))
                            else:
                                f.write('%s, '%str(x_val[i][j][k]))
                        f.write('%d\n'%int(y_val[i][j]))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train.shape = (y_train.shape[0], 1)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        return x_train, y_train, x_val, y_val, x_train_dict, x_val_dict


    def convert_data(self, v_data, tg, idx, i, w_data, test_init=0):
        x_i = []
        x_i_dict = {}
        hist_v = 0
        weather = []
        if test_init:
            i[2] += 1 # itv += 1, assert itv < 72
        if (i[0],i[1],i[2]) in w_data.keys():
            weather = self.construct_weather(w_data[(i[0],i[1],i[2])])
            x_i_dict['weather'] = weather
            if self.mov_time_win:
                if test_init:
                    last_6v = np.array(v_data[tg][idx-5:idx+1])
                    time_win = self.repeat_coding(0, self.win_bits)
                else:
                    # m = itv%6; v_data[idx-m-6, idx-m]
                    m = i[2]%6 
                    last_6v = np.array(v_data[tg][idx-m-6:idx-m])
                    time_win = self.repeat_coding(m, self.win_bits)
                hist_v = list(last_6v[:,-1])
            else:
                if test_init:
                    last_6v = np.array(v_data[tg][idx-5:idx+1])
                else:
                    last_6v = np.array(v_data[tg][idx-6:idx])
                hist_v = list(last_6v[:,-1])
                time_win = []
            x_i_dict['hist_v'] = hist_v 
            x_i_dict['time_win'] = time_win
            x_i_dict['lanes'] = self.lanes[tg]
            x_i_dict['gate_dirt'] = self.k_hot(tg, 5, 3)
            # assume that hours (11-hot) is more important than mins (7-hot)
            x_i_dict['itv'] = self.k_hot(i[2]/6, 12, self.h_hot) + self.k_hot(i[2]%6, 6, self.m_hot)
            #x_i_dict['itv'] = self.k_hot(i[2]/18, 4) # 72/18 = 4 bits
            if (i[0],i[1]) in self.weekend:
                x_i_dict['day'] = i[:2] + self.k_hot(1, 3, self.day_hot[1])
                #x_i_dict['day'] = i[:2] + self.repeat_coding(5, self.day_bits) # 3 bits
            elif (i[0],i[1]) in self.holiday:
                x_i_dict['day'] = i[:2] + self.k_hot(2, 3, self.day_hot[2])
                #x_i_dict['day'] = i[:2] + self.repeat_coding(7, self.day_bits)
            else:
                x_i_dict['day'] = i[:2] + self.k_hot(0, 3, self.day_hot[0])
                #x_i_dict['day'] = i[:2] + self.repeat_coding(3, self.day_bits)        

            x_i = x_i_dict['day'] + x_i_dict['itv'] + x_i_dict['gate_dirt'] + x_i_dict['lanes'] + \
                    x_i_dict['hist_v'] + x_i_dict['time_win'] + x_i_dict['weather']

        return x_i, x_i_dict, hist_v, weather


    def construct_weather(self, wtr):
        [pre, sea_pre, wind_d, wind_sp, temp, rel_hum, pcp] = wtr
        pre     = self.repeat_coding(pre/self.std_pre,       self.wter_bits[0]) 
        sea_pre = self.repeat_coding(sea_pre/self.std_pre,   self.wter_bits[1])
        wind_d  = self.repeat_coding(wind_d/self.wind_dirt,  self.wter_bits[2])
        wind_sp = self.repeat_coding(wind_sp/self.wind_rank, self.wter_bits[3])
        temp    = self.repeat_coding(temp/self.com_tmp,      self.wter_bits[4])
        rel_hum = self.repeat_coding(rel_hum/self.com_rh,    self.wter_bits[5])
        pcp     = self.repeat_coding(pcp/self.rain_rank,     self.wter_bits[6])
        weather = pre + sea_pre + wind_d + wind_sp + temp + rel_hum + pcp

        return weather



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', type=int, default=0, help='')
    parser.add_argument('--refine_val', type=int, default=1, help='')
    parser.add_argument('--mov_time_win', type=int, default=1, help='')
    args = parser.parse_args()
    pdb.set_trace()
    test = DataProcessing(args)
    x_train, y_train, x_val, y_val, xtd, xvd = test.combine_train_data()
    x_test, x_test_dict, raw_data = test.init_test_data()
    print x_train.shape
    print y_train.shape
    print x_val.shape
    print y_val.shape
    print x_test.shape

