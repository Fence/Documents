#coding:utf-8
import re
import time
import pdb
import argparse
import numpy as np
from tqdm import tqdm

class DataProcessing():
    def __init__(self):
        self.data_dir = '../dataSets/'
        self.train_volume_file = 'training/training_20min_avg_volume_new.csv'
        self.train_weather_file = 'training/weather (table 7)_training_update.csv'
        self.links_file = 'training/links (table 3).csv'
        self.routes_file = 'training/routes (table 4).csv'
        self.train_time_file = 'training/training_20min_avg_travel_time.csv'
        self.test_volume_file = 'testing_phase1/test_20min_avg_volume.csv'
        self.test_weather_file = 'testing_phase1/volume(table 6)_test1.csv'
        
        self.test_day = 12
        self.std_pre = 1#1013.25
        self.com_tmp = 1#25.0
        self.com_rh = 1#25.0
        self.wind_dirt = 1#360.0
        self.wind_rank = 1#2.0
        self.rel_humid = 1#50.0
        self.rain_rank = 1#25.0
        self.day_bits = 3 #3
        self.itv_bits = 18 #4
        self.gate_dirt_bits = 5 #4
        self.lanes_bits = 3 #3
        self.car_mod_bits = 8 #
        self.has_etc_bits = 3 #3
        self.hist_v_bits = 3 #3
        self.wter_bits = [1, 1, 4, 4, 4, 4, 4] #
        
        self.split_bit = self.day_bits + self.itv_bits + self.gate_dirt_bits + self.lanes_bits + \
                        self.car_mod_bits + self.has_etc_bits + self.hist_v_bits
        self.weekend = [(9,24), (9,25), (10,15), (10,16), (10, 22), (10, 23)]
        self.holiday = [(9,15), (9,16),(9,17), (10,1), (10,2), (10,3), \
                        (10,4), (10,5), (10,6), (10,7)]
        self.lanes = [self.repeat_coding(2, self.lanes_bits), \
                        self.repeat_coding(2, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits), \
                        self.repeat_coding(1, self.lanes_bits)] 
        self.test_v_data = self.get_hist_volume(self.test_volume_file, 0)
        self.test_w_data = self.get_weather(self.test_weather_file, 0)
        self.test_days = [(10,18), (10,19), (10,20), (10,21), (10,22), (10,23), (10,24)]
        self.test_hours = [6, 7, 8, 15, 16, 17]


            

    def one_hot(self, num, length):
        code = np.zeros(length)
        code[num] = 1
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


    def convert_data(self, v_data, tg, idx, i, w_data):
        gate_dirt = [[1,0,0,0],[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,1,1]]
        lanes = [self.repeat_coding(2, self.lanes_bits), 
        self.repeat_coding(2, self.lanes_bits), self.repeat_coding(1, self.lanes_bits),
        self.repeat_coding(1, self.lanes_bits), self.repeat_coding(1, self.lanes_bits)] 
        # gate 3 bits, dirt 1, lanes 3 bits
        if (i[0],i[1]) in self.weekend:
            #day = self.one_hot(1, 3)
            day = self.repeat_coding(2, self.day_bits) # 3 bits
        elif (i[0],i[1]) in self.holiday:
            #day = self.one_hot(2, 3)
            day = self.repeat_coding(3, self.day_bits)
        else:
            #day = self.one_hot(0, 3)
            day = self.repeat_coding(1, self.day_bits)
        #itv = self.one_hot(i[2]/6, 12) + self.one_hot(i[2]%6, 6) # 18 bits
        itv = self.one_hot(i[2]/18, 4) # 72/18 = 4 bits
        #gate_dirt = self.one_hot(tg, 5)
        last_6v = np.array(v_data[tg][idx-6:idx])
        hist_car_mod = list(sum(last_6v[:,3:11]))
        hist_has_etc = self.repeat_coding(sum(last_6v[:,-2]), self.has_etc_bits) 
        #self.repeat_coding(i[11], self.has_etc_bits) # 3 bits
        hist_v = self.repeat_coding(sum(last_6v[:,-1]), self.hist_v_bits) # 3 bits
        if (i[0],i[1],i[2]) in w_data.keys():
            [pre, sea_pre, wind_d, wind_sp, temp, rel_hum, pcp] = w_data[(i[0],i[1],i[2])]
            pre     = self.repeat_coding(pre/self.std_pre,       self.wter_bits[0]) 
            sea_pre = self.repeat_coding(sea_pre/self.std_pre,   self.wter_bits[1])
            wind_d  = self.repeat_coding(wind_d/self.wind_dirt,  self.wter_bits[2])
            wind_sp = self.repeat_coding(wind_sp/self.wind_rank, self.wter_bits[3])
            temp    = self.repeat_coding(temp/self.com_tmp,      self.wter_bits[4])
            rel_hum = self.repeat_coding(rel_hum/self.com_rh,    self.wter_bits[5])
            pcp     = self.repeat_coding(pcp/self.rain_rank,     self.wter_bits[6])
            weather = pre + sea_pre + wind_d + wind_sp + temp + rel_hum + pcp
            x_i = day + itv + gate_dirt[tg] + lanes[tg] + hist_car_mod + \
            hist_has_etc + hist_v + weather
        else:
            x_i = []
            weather = []
        return x_i,hist_v,weather


    def get_hist_volume(self, file_name, save_file):
        tg10 = [] # mm--dd--itv--[car_mod]--has_etc--volume
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
                #assert len(a) == 24
                itv = int(a[4])*3 + int(a[5])/20
                #item = a[2:4] + [itv] + a[16:] + [a[15], a[14]] 
                item = a[2:4] + [itv] + [a[14]] # mm--dd--itv--volume
                if a[0] == '1' and a[13] == '0':
                    if len(tg10):
                        self.padding_data(item, tg10, itv)
                    tg10.append([int(b) for b in item])
                elif a[0] == '1' and a[13] == '1':
                    if len(tg11):
                        self.padding_data(item, tg11, itv)
                    tg11.append([int(b) for b in item])
                elif a[0] == '2' and a[13] == '0':
                    if len(tg20):
                        self.padding_data(item, tg20, itv)
                    tg20.append([int(b) for b in item])
                elif a[0] == '3' and a[13] == '0':
                    if len(tg30):
                        self.padding_data(item, tg30, itv)
                    tg30.append([int(b) for b in item])
                else:
                    if len(tg31):
                        self.padding_data(item, tg31, itv)
                    tg31.append([int(b) for b in item])
        v_data = [tg10, tg11, tg20, tg30, tg31]

        if save_file:
            out_file = self.data_dir + file_name.split('/')[0] + '/out_volume.csv.txt'
            print out_file
            with open(out_file, 'w') as fw:
                #fw.write('gd,mm,dd,itv,car_mod,has_etc,volume\n')
                fw.write('gd, mm, dd, itv, volume\n')
                for j, d in enumerate(v_data):
                    for k in d:
                        #cm = ' '.join([str(c) for c in k[4:12]])
                        #tk = k[:3] + [cm] + k[11:]
                        fw.write('%d,%s\n'%(j+1, ','.join([str(l) for l in k])))

        return v_data


    def get_weather(self, file_name, save_file):
        w_data = {} #(m, d, itv) [p--sp--wd--ws--tp--rh--pr]
        with open(self.data_dir + file_name) as f:
            titles = f.readline()
            contents = f.readlines()
            for i in xrange(640, len(contents)):
                line = contents[i]
                a = re.findall(r'[\d\.]+', line)
                #print len(a), a
                assert len(a) == 11
                hour = int(a[3])
                for i in xrange(hour*3, hour*3+9):
                    dt = (int(a[1]), int(a[2]), i)
                    w_data[dt] = [float(b) for b in a[4:]]
        if save_file:
            out_file = self.data_dir + file_name.split('/')[0] + '/task2_weather2.txt'
            print out_file
            with open(out_file, 'w') as fw:
                fw.write('m--d--itv--p--sp--wd--ws--tp--rh--pr\n')
                for j in w_data.keys():
                    fw.write(str(j)+' '+str(w_data[j])+'\n')
        #print 'len(w_data) = %d'%len(w_data)
        return w_data


    def combine_data3(self, save_file):
        x_train = []
        y_train = []
        v_data = self.get_hist_volume(self.train_volume_file, save_file)
        w_data = self.get_weather(self.train_weather_file, save_file)
        raw_data = []
        x_test = [[],[],[],[],[]]
        y_test = [[],[],[],[],[]]
        assert len(v_data[0]) == len(v_data[1]) ==len(v_data[2]) == len(v_data[3]) == len(v_data[4])
        for idx in xrange(6, len(v_data[0])):
            for tg in xrange(len(v_data)):
                i = v_data[tg][idx]
                x_i,hist_v,weather = self.convert_data(v_data, tg, idx, i, w_data)
                if not x_i:
                    continue
                if i[0] == 10 and i[1] > self.test_day: #keep the last 5 days for test 
                    x_test[tg].append(np.array(x_i))
                    y_test[tg].append(i[-1])
                else:
                    x_train.append(np.array(x_i))
                    y_train.append(i[-1])
                raw_data.append([tg]+i+hist_v+weather)

        if save_file:
            with open(self.data_dir+'raw_data3.csv', 'w') as f:
                f.write('gd, mm, dd, it, car_model, has_etc, volume, hist_volume, weather\n')
                for rd in raw_data:
                    cm = ' '.join([str(c) for c in rd[4:12]])
                    wd = ' '.join([str(w) for w in rd[14+self.hist_v_bits:]])
                    f.write(','.join([str(rd[0]), str(rd[1]), str(rd[2]), str(rd[3]), 
                        cm, str(rd[12]), str(rd[13]), str(rd[14]), wd])+'\n')
            with open(self.data_dir+'xy_train3.txt', 'w') as f:
                f.write('day, iterval, gate-direction, lanes, car_model, \
                    has_etc, hist_v, weather, volume\n')
                for i in xrange(len(x_train)):
                    for j in xrange(len(x_train[i])):
                        if j < self.split_bit:
                            f.write('%d, '%int(x_train[i][j]))
                        else:
                            f.write('%s, '%str(x_train[i][j]))
                    f.write('%d\n'%int(y_train[i]))
            with open(self.data_dir+'xy_test3.txt', 'w') as f:
                f.write('day, iterval, gate-direction, lanes, car_model, \
                    has_etc, hist_v, weather, volume\n')
                for i in xrange(len(x_test)):
                    for j in xrange(len(x_test[i])):
                        for k in xrange(len(x_test[i][j])):
                            if k < self.split_bit:
                                f.write('%d, '%int(x_test[i][j][k]))
                            else:
                                f.write('%s, '%str(x_test[i][j][k]))
                        f.write('%d\n'%int(y_test[i][j]))

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


    def combine_test_data(self, last_x, itv):
        next_x = []
        for d in xrange(len(self.test_days)):
            x_d = last_x[d]
            x_d['itv'] = self.one_hot(itv/6, 12) + self.one_hot(itv%6, 6)

        

        return np.array(x_test), np.array(y_test)


    def combine_train_data(self, save_file):
        x_train = []
        y_train = []
        v_data = self.get_hist_volume(self.train_volume_file, save_file)
        w_data = self.get_weather(self.train_weather_file, save_file)
        raw_data = []
        x_test = [[],[],[],[],[]]
        y_test = [[],[],[],[],[]]
        assert len(v_data[0]) == len(v_data[1]) ==len(v_data[2]) == len(v_data[3]) == len(v_data[4])
        for idx in xrange(6, len(v_data[0])):
            for tg in xrange(len(v_data)):
                i = v_data[tg][idx]
                x_i,hist_v,weather = self.convert_data2(v_data, tg, idx, i, w_data)
                if not x_i:
                    continue
                if i[0] == 10 and i[1] > self.test_day: #keep the last 5 days for test 
                    x_test[tg].append(np.array(x_i))
                    y_test[tg].append(i[-1])
                else:
                    x_train.append(np.array(x_i))
                    y_train.append(i[-1])
                raw_data.append([tg]+i+hist_v+weather)

        return np.array(x_test), np.array(y_test)


    def convert_data2(self, v_data, tg, idx, i, w_data):
        x_i = {}
        x_i_list = []
        hist_v = []
        weather = []
        if (i[0],i[1],i[2]) in w_data.keys():
            [pre, sea_pre, wind_d, wind_sp, temp, rel_hum, pcp] = w_data[(i[0],i[1],i[2])]
            pre     = self.repeat_coding(pre/self.std_pre,       self.wter_bits[0]) 
            sea_pre = self.repeat_coding(sea_pre/self.std_pre,   self.wter_bits[1])
            wind_d  = self.repeat_coding(wind_d/self.wind_dirt,  self.wter_bits[2])
            wind_sp = self.repeat_coding(wind_sp/self.wind_rank, self.wter_bits[3])
            temp    = self.repeat_coding(temp/self.com_tmp,      self.wter_bits[4])
            rel_hum = self.repeat_coding(rel_hum/self.com_rh,    self.wter_bits[5])
            pcp     = self.repeat_coding(pcp/self.rain_rank,     self.wter_bits[6])
            weather = pre + sea_pre + wind_d + wind_sp + temp + rel_hum + pcp
            
            x_i['weather'] = weather
            last_6v = np.array(v_data[tg][idx-6:idx])
            x_i['hist_vol'] = self.repeat_coding(sum(last_6v[:,-1]), self.hist_v_bits) # 3 bits
            #x_i['hist_has_etc'] = self.repeat_coding(sum(last_6v[:,-2]), self.has_etc_bits) 
            #x_i['hist_car_mod'] = list(sum(last_6v[:,3:11]))
            x_i['lanes'] = self.lanes[tg]
            x_i['gate_dirt'] = self.one_hot(tg, 5)
            x_i['itv'] = self.one_hot(i[2]/6, 12) + self.one_hot(i[2]%6, 6) # 18 bits
            #x_i['itv'] = self.one_hot(i[2]/18, 4) # 72/18 = 4 bits
            if (i[0],i[1]) in self.weekend:
                #x_i['day'] = self.one_hot(1, 3)
                x_i['day'] = self.repeat_coding(2, self.day_bits) # 3 bits
            elif (i[0],i[1]) in self.holiday:
                #x_i['day'] = self.one_hot(2, 3)
                x_i['day'] = self.repeat_coding(3, self.day_bits)
            else:
                #x_i['day'] = self.one_hot(0, 3)
                x_i['day'] = self.repeat_coding(1, self.day_bits)        

        for d in x_i:
            x_i_list += d 
        return x_i_list, hist_v, weather



if __name__ == '__main__':
    pdb.set_trace()
    test = DataProcessing()
    #test.get_hist_volume(1)
    #test.get_weather(1)
    x_train,y_train, x_test,y_test = test.combine_data3(1)
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    #test.combine_data_short(save_file=True)