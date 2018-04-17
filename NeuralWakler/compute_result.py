import os
import re
import ipdb

def compute_result(result_dir, end_idx):
    f_measure = 0.0
    idx = range(9)
    idx.append(-1)
    counter = 0
    #ipdb.set_trace()
    for i in idx:
        text_name = 'fold%d_ep1.txt'%i
        if not os.path.exists(result_dir+text_name):
            continue
        with open(result_dir+text_name) as f:
            content = f.readlines()
            ed = []
            fm = []
            for line in content:
                if re.search('end_idx', line):
                    ed.append(int(line.split()[-1]))
                elif re.search('f_measure', line):
                    fm.append(float(line.split()[-1]))
                    
            for j in xrange(len(ed)):
                if ed[j] == end_idx:
                    f_measure += fm[j]
                    counter += 1
    f_measure /= counter
    print 'f_measure of %s is %f'%(result_dir, f_measure) 


def compute_ts_result(result_dir):
    f_measure = 0.0
    #ipdb.set_trace()
    if os.path.exists(result_dir+'result.txt'):
        with open(result_dir+'result.txt') as f:
            content = f.readlines()
            ed = []
            fm = []
            for line in content:
                if re.search('end_idx', line):
                    ed.append(int(line.split()[-1]))
                elif re.search('f_measure', line):
                    fm.append(float(line.split()[-1]))
            assert len(fm) == len(ed)
            new_ed = []
            new_fm = []
            for i in xrange(len(fm)/3):
                new_ed.append(ed[i*3])
                new_fm.append(sum(fm[i*3:i*3+3])/3.0)
            print 'result dir: %s'%result_dir
            for j in xrange(len(new_ed)):
                print '%d\t%f'%(new_ed[j],new_fm[j])


if __name__ == '__main__':
    result_dirs = ['cooking', 'wikihow', 'windows2k', 'sail', 'long_wikihow', 'long_windows']
    end_idices = [1023, 745, 1544, 3519]
    #for i in range(len(end_idices)):
    #    compute_result('./results/%s/'%result_dirs[i], end_idices[i])
    for i in xrange(len(result_dirs)):
        compute_ts_result('./results/ts_results/%s/'%result_dirs[i])