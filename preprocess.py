import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from urllib.parse import unquote

vica_path_backup = None


def ipv42ipv6(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns
    for col in columns:
        if 'IP_src' in col or 'IP_dst' in col:
            df['_'.join(['IPv6', col.split('_')[1], '{}']).format(str(int(col[-1]) + 2))] = df[col]
            df = df.drop(columns=[col])
        elif 'IP' in col:
            df = df.rename(columns={col: col.replace('IP', 'IPv6')})
    df['IPv6_src_0'] = 0
    df['IPv6_src_1'] = 0
    df['IPv6_dst_0'] = 0
    df['IPv6_dst_1'] = 0
    return df


def basic_data() -> pd.DataFrame:
    abnormal_ipv4_files = [
        'Ethernet_IP_TCP_HTTP 1_Raw.csv',
        'Ethernet_IP_TCP_HTTP 1_HTTP Request.csv',
        'Ethernet_IP_TCP_HTTP 1_HTTP Response_Raw.csv',
    ]

    abnormal_ipv6_files = [
        'Ethernet_IPv6_TCP_HTTP 1_Raw_Padding.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Response_Raw_Padding.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Request_Raw_Padding.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Request_Padding.csv'
    ]

    normal_ipv6_files = [
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Request.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Response.csv',
        # 'Ethernet_IPv6_TCP_HTTP 1_HTTP Request_Raw.csv',
        # 'Ethernet_IPv6_TCP_HTTP 1_HTTP Response_Raw.csv',
    ]

    abnormal_ipv4 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv4_files]
    abnormal_ipv4 = [ipv42ipv6(df) for df in abnormal_ipv4]
    abnormal_ipv6 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv6_files]
    normal_ipv6 = [pd.read_csv(os.path.join('output/normal', f)) for f in normal_ipv6_files]

    dfs = abnormal_ipv4 + abnormal_ipv6 + normal_ipv6

    df = pd.concat(dfs, ignore_index=True)
    return df


def basic_process(df: pd.DataFrame, use_path: bool = False) -> pd.DataFrame:
    df['Path'] = df['Path'].fillna(df['Raw_load'])
    df = df.dropna(thresh=0.4 * df.shape[0], axis=1)  # drop cols with too many na value

    not_process = [
        # 'Cookie',
        'Date',
        # 'Path',
        'Host',
        'Accept',
        'Connection',
        'TCP_options_Timestamp',
        'Via',
        'Server',
        'Referer'
    ]
    not_process = [col for col in not_process if col in df.columns]
    global vica_path_backup
    vica_path_backup = df['Path'].astype(str)
    df = df.drop(not_process, axis=1)

    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna('0').apply(lambda x: re.sub(r'[^0-9]', '', x))  # special process

    df = df.fillna(0)  # fill na

    drop_cols = [x for x in df.columns if len(df[x].unique()) < 2 or len(df[x].unique()) >= 0.99 * df.shape[0]]
    df = df.drop(drop_cols, axis=1)

    for col in ['TCP_flags', 'TCP_reserved', 'Method', 'ETag']:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    num_cols = [
        'Ethernet_dst_0',
        'Ethernet_dst_1',
        'Ethernet_dst_2',
        'Ethernet_dst_3',
        'Ethernet_dst_4',
        'Ethernet_dst_5',
        'Ethernet_src_0',
        'Ethernet_src_1',
        'Ethernet_src_2',
        'Ethernet_src_3',
        'Ethernet_src_4',
        'Ethernet_src_5',
        'IPv6_dst_0',
        'IPv6_dst_1',
        'IPv6_dst_2',
        'IPv6_dst_3',
        'IPv6_dst_4',
        'IPv6_dst_5',
        'IPv6_src_0',
        'IPv6_src_1',
        'IPv6_src_2',
        'IPv6_src_3',
        'IPv6_src_4',
        'IPv6_src_5',
        'Age',
    ]
    num_cols = [col for col in num_cols if col in df.columns]

    df[num_cols] = df[num_cols].applymap(lambda x: int(x, 16) if type(x) == str else x)

    if use_path:
        df = n_gram_path_process(df)
    else:
        df = df.drop('Path', axis=1)

    return df


def vica_further_process(df: pd.DataFrame) -> pd.DataFrame:
    add_info_from_path(df)  # there are also soem "cheating" attribute
    # ont_hot(df, "Method") # if you would like to change some method to one hot
    # add_spam_score(df) # can be slower, and the performance is not so good
    further_drop_useless_column(df)
    return df


def n_gram_path_process(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 1))
    df['Path'] = df['Path'].apply(lambda x: 'b""' if type(x) != str else x)
    df['Path'] = df['Path'].apply(lambda _p: unquote(eval(_p).decode()))
    vectorizer.fit(df[df['Path'] != ''])
    f = vectorizer.transform(df['Path']).toarray()
    n = f.shape[1]
    df.drop('Path', axis=1, inplace=True)
    for i in range(n):
        df[f'Path{i}'] = f[:, i]
    # print(df['Path'])
    # df['Path'] = df['Path'].apply(lambda x: list(vectorizer.transform([x]).toarray()[0]))
    # print(df.axes)
    return df


def further_drop_useless_column(df):
    not_process = [
        'Ethernet_type',
        'IPv6_fl',
        'IPv6_hlim',
        'IPv6_nh',
        'IPv6_plen',
        'TCP_ack',
        'TCP_chksum',
        'TCP_dataofs',
        'TCP_dport',
        'TCP_flags',
        'TCP_seq',
        'Ethernet_dst_0',
        'Ethernet_dst_1',
        'Ethernet_dst_2',
        'Ethernet_dst_3',
        'Ethernet_dst_4',
        'Ethernet_dst_5',
        'Ethernet_src_0',
        'Ethernet_src_1',
        'Ethernet_src_2',
        'Ethernet_src_3',
        'Ethernet_src_4',
        'Ethernet_src_5',
        'IPv6_dst_0',
        'IPv6_dst_1',
        'IPv6_dst_2',
        'IPv6_dst_3',
        'IPv6_dst_4',
        'IPv6_dst_5',
        'IPv6_src_0',
        'IPv6_src_1',
        'IPv6_src_2',
        'IPv6_src_3',
        'IPv6_src_4',
        'IPv6_src_5',
    ]
    not_process = [col for col in not_process if col in df.columns]
    df = df.drop(not_process, axis=1)


def ont_hot(df, col_name):
    one_hot = pd.get_dummies(pd.Series(df[col_name]))
    df.drop([col_name], axis=1)
    df = df.join(one_hot)


def add_info_from_path(df):
    global vica_path_backup
    df['path_len'] = [len(s) if isinstance(s, str) else 0 for s in vica_path_backup]
    # the following may be cheating
    # df['has_http'] = [s.find('http') == -1 for s in vica_path_backup]
    # df['has_quote'] = [s.find('%22') == -1 for s in vica_path_backup]
    # df['has_SELECT'] = [s.upper().find('SELECT') == -1 for s in vica_path_backup]
    # df['post_php'] = [s.find('.php') == -1 for s in vica_path_backup]


def add_spam_score(df):
    global vica_path_backup

    src_columns = ['IPv6_src_0', 'IPv6_src_1', 'IPv6_src_2', 'IPv6_src_3', 'IPv6_src_4', 'IPv6_src_5']
    ipv6_src = df[src_columns].astype(str).apply(lambda x: '.'.join(x), axis=1)
    all_ip = pd.unique(ipv6_src)
    df['spam_score'] = 0

    for ip in all_ip:
        if sum(ipv6_src == ip) < 15 or ip == "2408.877d.30.1.nan.100":  # server ip
            continue
        pathes = vica_path_backup[ipv6_src == ip].tolist()
        pathes = list(set([s for s in pathes if s]))
        if len(pathes) < 5:
            continue
        pathes.sort()
        distance = 0
        for i in range(len(pathes) - 1):
            distance += levenshteinDistance(pathes[i], pathes[i + 1])
        df['spam_score'][ipv6_src == ip] = distance / (len(pathes) - 1)


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return 1 - distances[-1] / len(s2)
