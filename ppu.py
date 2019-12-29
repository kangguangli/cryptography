from scapy.all import *
from scapy.layers import http

import pandas as pd

configs = {
    'mask': [
        # 'padding',
        # 'raw'
    ]
}


def generalLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'src' in key or 'dst' in key:
            # TODO
            # src = 2002:dca5:f85::dca5:f85 [6to4 GW: 220.165.15.133] from ips_2002-dca5-f85--dca5-f85_20190131_202924_543872217.pcap
            fields = item.split('.') if '.' in item and ':' not in item else item.split(':')
            fields = fields[:6]
            if len(fields) < 6 and 'IPv6' in layer.name:  # make sure IPv6 and fill 6 address
                fields += [0 for x in range(6 - len(fields))]
            for i in range(len(fields)):
                data['_'.join([layer.name, key, str(i)])] = fields[i]
        elif 'data' in key or 'load' in key:
            data['_'.join([layer.name, key])] = item  # hash(item)
        elif 'options' in key:
            for k, i in dict(item).items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def tcpOptionLayer(op: dict):
    data = {
        # "EOL": 0,
        # "NOP": 1,
        "MSS": None,
        "WScale": None,
        "SAckOK": None,
        "SAck": None,
        "Timestamp": None,
        "AltChkSum": None,
        "AltChkSumOpt": None,
        "Mood": None,
        "UTO": None,
        "TFO": None,
    }
    for key, item in op.items():
        if key in data:
            data[key] = item
        else:
            pass

    return data


def tcpLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'options' in key:
            item = tcpOptionLayer(dict(item))
            for k, i in item.items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def bootpLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if any([s in key for s in ['chaddr', 'sname', 'file']]):
            data['_'.join([layer.name, key])] = hash(item)
        else:
            data['_'.join([layer.name, key])] = item

    return data


def dhcpOptionLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'options' in key:
            new_item = [i for i in item if type(i) == tuple]
            new_item = dict(new_item)
            for k, i in new_item.items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def dnsLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'qd' == key or 'ar' == key:
            if item is None:
                continue
            if type(item) == list:
                item = item[0]  # TODO
            for k, i in item.fields.items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def snmpLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'PDU' in key:
            for k, i in item.fields.items():
                if isinstance(item, ASN1_Object):
                    data['_'.join([layer.name, key, k])] = i.val
                elif 'varbindlist' in key:
                    data['_'.join([layer.name, key, k])] = i[0].oid.val  # TODO
        elif isinstance(item, ASN1_Object):
            data['_'.join([layer.name, key])] = item.val
        else:
            data['_'.join([layer.name, key])] = item

    return data


def icmpv6NDNLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'tgt' in key:
            fields = item.split('.') if '.' in item else item.split(':')
            for i in range(len(fields)):
                data['_'.join([layer.name, key, str(i)])] = fields[i]
        else:
            data['_'.join([layer.name, key])] = item

    return data


def icmpv6NDOSLLALayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'lladdr' in key:
            fields = item.split('.') if '.' in item else item.split(':')
            for i in range(len(fields)):
                data['_'.join([layer.name, key, str(i)])] = fields[i]
        else:
            data['_'.join([layer.name, key])] = item

    return data


def ikeTransLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'transforms' in key:
            for k, i in dict(item).items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def ikeProposalLayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'trans' == key:
            item = ikeTransLayer(item)
            for k, i in item.items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def isakmpSALayer(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'prop' in key:
            item = ikeProposalLayer(item)
            for k, i in item.items():
                data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def ipv6EH(layer):
    data = {}
    for key, item in layer.fields.items():
        if 'options' in key:
            if type(item) == list:
                for i in item:
                    new_item = generalLayer(i)
                    for k, i in new_item.items():
                        data['_'.join([layer.name, key, k])] = i
            else:
                item = generalLayer(item)
                for k, i in item.items():
                    data['_'.join([layer.name, key, k])] = i
        else:
            data['_'.join([layer.name, key])] = item

    return data


def httpRequest(layer):
    data = {}

    fileds = ['Method', 'Path', 'Http-Version'] + http.REQUEST_HEADERS + http.GENERAL_HEADERS
    for f in fileds:
        data[f] = None

    for key, item in layer.fields.items():
        if key in fileds:
            data[key] = item
        else:
            pass

    return data


def httpResponse(layer):
    data = {}

    fields = ['Status-Code', 'Reason-Phrase', 'Http-Version'] + http.RESPONSE_HEADERS + http.GENERAL_HEADERS
    for f in fields:
        data[f] = None

    for key, item in layer.fields.items():
        if key in fields:
            data[key] = item
        else:
            pass

    return data


all_layers = {
    'IPv6 Extension Header - Hop-by-Hop Options Header': ipv6EH,
    'Raw': generalLayer,
    'NTPHeader': generalLayer,
    'ICMPv6 Packet Too Big': generalLayer,
    'ICMPv6 Neighbor Discovery - Neighbor Advertisement': icmpv6NDNLayer,
    'ICMPv6 Echo Reply': generalLayer,
    'TCP in ICMP': tcpLayer,
    'ICMPv6 Time Exceeded': generalLayer,
    'Ethernet': generalLayer,
    'DNS': dnsLayer,
    'BOOTP': bootpLayer,
    'DHCP options': dhcpOptionLayer,
    'IP': generalLayer,
    'IPv6': generalLayer,
    'ICMPv6 Neighbor Discovery Option - Source Link-Layer Address': icmpv6NDOSLLALayer,
    'SNMP': snmpLayer,
    'ICMPv6 Neighbor Discovery - Neighbor Solicitation': icmpv6NDNLayer,
    'ISAKMP SA': isakmpSALayer,
    'ISAKMP': generalLayer,
    'Authenticator': generalLayer,
    'ICMPv6 Destination Unreachable': generalLayer,
    'ISAKMP Vendor ID': generalLayer,
    'IPv6 in ICMPv6': generalLayer,
    'Padding': generalLayer,
    'UDP': generalLayer,
    'ICMPv6 Echo Request': generalLayer,
    'ICMPv6 Parameter Problem': generalLayer,
    'UDP in ICMP': generalLayer,
    'TCP': tcpLayer,
    'HTTP Response': httpResponse,
    'HTTP Request': httpRequest,
    'HTTP 1': generalLayer
}


def getPID(layers: list):
    return '_'.join(layers)


def expand(x: Packet):
    if x.name.lower() not in configs['mask']:
        yield x
    while x.payload:
        x = x.payload
        if x.name.lower() not in configs['mask']:
            yield x


def getPIDFromPkt(p: Packet):
    layers = [layer.name for i, layer in enumerate(expand(p), 0)]
    return getPID(layers)


class GeneralPacket:

    def __init__(self, packet_id, path):
        self.id = packet_id
        self.count = 0
        self.run = False
        self.path = os.path.join(path, self.id) + '.csv'
        self.buffer = []
        # self.pkt_writer = PcapWriter(os.path.join('', path, self.id) + '.pcap')

    def getData(self, pkt, label):

        data = {}
        data['label'] = label
        data['time'] = float(pkt.time)

        for _, layer in enumerate(expand(pkt), 0):
            try:
                data.update(all_layers[layer.name](layer))
            except KeyError:
                data.update(generalLayer(layer))
        self.count += 1
        self.buffer.append(data)
        if self.count % 1000 == 0:
            self.write()

    def close(self):

        if len(self.buffer) != 0:
            self.write()

        # self.pkt_writer.close()

    def increase(self):
        self.count += 1

    def write(self):
        if not self.run:
            pd.DataFrame(self.buffer).to_csv(self.path, index=False)
            self.run = True
        else:
            pd.DataFrame(self.buffer).to_csv(self.path, header=None, mode="a", index=False)
        self.buffer = []
