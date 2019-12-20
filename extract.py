from scapy.all import *

import os
import pandas as pd

from ppu import *

from tools import log_time


@log_time
def build_examples(path: str, filename: str) :

    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if f.find('pcap') != -1]

    unique_packet = set()
    packet_ids = set()
    writer = PcapWriter(filename)

    for f in fs:
        try:
            s = PcapReader(f)
            while True:
                try:
                    p = s.read_packet()
                    layers = [layer.name for i, layer in enumerate(expand(p), 0)]
                    unique_packet.update(layers)
                    packet_id = getPID(layers)
                    if packet_id not in packet_ids:
                        p.show()
                        writer.write(p)
                        packet_ids.add(packet_id)
                except EOFError:
                    #print('Finish', f)
                    break
            s.close()
            writer.flush()

        except Exception as e:
            print('Error', e)

    print(unique_packet)
    writer.flush()  
    writer.close()


def processFile(f : str, ppus : set, label : int):
    try:
        s = PcapReader(f)
        while True:
            try:
                p = s.read_packet()
                packet_id = getPIDFromPkt(p)
                ppus[packet_id].getData(p, label)
            except EOFError:
                break
            except Exception as e:
                # print(e)
                print('-----------------Pkt Process Error')
                p.show()
                print('-----------------')
                raise e
        s.close()

    except Exception as e:
        raise e


@log_time
def extract(input_path : str, output_path : str, example_file : str, label : int):

    fs = os.listdir(input_path)
    fs = [os.path.join(input_path, f) for f in fs]

    pkts = rdpcap(example_file)
    ppus = [getPIDFromPkt(p) for p in pkts]
    ppus = list(set(ppus))

    ppus = [(pid, GeneralPacket(pid, output_path)) for pid in ppus]
    ppus = dict(ppus) 

    for f in fs:
        processFile(f, ppus, label)

    [ppu.close() for i, ppu in ppus.items()]
    total = sum([ppu.count for i, ppu in ppus.items()])
    print('total packet:', total)


if __name__ == "__main__":

    all_pakcet_path = 'data/abnormal'
    normal_packet_path = 'data/normal'
    abnormal_packet_path = 'data/abnormal'
    example_file = 'output/example.pcap' 
    
    # not reslove default
    load_layer('http') 

    build_examples(all_pakcet_path, example_file)

    #extract(normal_packet_path, 'output/normal', example_file, 0)
    extract(abnormal_packet_path, 'output/abnormal', example_file, 1)

