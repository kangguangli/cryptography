import time
import functools
import os
from scapy import *


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Time: %f " % (end_time - start_time))
        return result

    return wrapper

def merge():

    path = 'data/abnormal'
    filename = 'output/all.pcap'

    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if f.find('pcap') != -1]
    writer = PcapWriter(filename)

    for f in fs:
        try:
            s = PcapReader(f)
            while True:
                try:
                    p = s.read_packet()
                    writer.write(p)
                except EOFError:
                    break
            s.close()
            writer.flush()
        except Exception as e:
            print('Error', e)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    merge()
