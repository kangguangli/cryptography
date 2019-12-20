import os
import re
import pandas as pd


def ipv42ipv6(df : pd.DataFrame) -> pd.DataFrame :

    columns = df.columns
    for col in columns:
        if 'IP_src' in col or 'IP_dst' in col:
            df['_'.join(['IPv6', col.split('_')[1], '{}']).format(str(int(col[-1]) + 2))] = df[col]
            df = df.drop(columns = [col])
        elif 'IP' in col:
            df = df.rename(columns = {col : col.replace('IP', 'IPv6')})
    df['IPv6_src_0'] = 0
    df['IPv6_src_1'] = 0
    df['IPv6_dst_0'] = 0
    df['IPv6_dst_1'] = 0
    return df

# join request and response
def merge_normal(path : str) -> pd.DataFrame :

    df_request = pd.read_csv(os.path.join(path, normal_ipv6_files[0]))
    df_response = pd.read_csv(os.path.join(path, normal_ipv6_files[0]))

    df_request = df_request.sort_values(by = 'time')
    df_response = df_response.sort_values(by = 'time')

    df = df_request.join(df_response, lsuffix='_caller', rsuffix='_other')

    return df
    
# join request and response
def merge_abnormal(path : str) -> pd.DataFrame :

    dfs_ipv4 = [pd.read_csv(os.path.join(path, f)) for f in abnormal_ipv4_files]
    dfs_ipv6 = [ipv42ipv6(df) for df in dfs_ipv4]  
    df = dfs_ipv6[0].join(dfs_ipv6[1], lsuffix='_caller', rsuffix='_other')
    dfs_ipv6 = [pd.read_csv(os.path.join(path, f)) for f in abnormal_ipv6_files]
    dfs_ipv6 = dfs_ipv6[0].join(dfs_ipv6[1], lsuffix='_caller', rsuffix='_other')
    df = pd.concat([df, dfs_ipv6], ignore_index = True)

    return df


def tcp_abnormal_data() -> pd.DataFrame:

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
        # 'Ethernet_IPv6_TCP_HTTP 1_HTTP Request.csv',
        # 'Ethernet_IPv6_TCP_HTTP 1_HTTP Response.csv',
    ]

    abnormal_ipv4 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv4_files]
    abnormal_ipv4 = [ipv42ipv6(df) for df in abnormal_ipv4] 
    abnormal_ipv6 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv6_files]
    normal_ipv6 = [pd.read_csv(os.path.join('output/normal', f)) for f in normal_ipv6_files]  

    dfs = abnormal_ipv4 + abnormal_ipv6 + normal_ipv6

    df = pd.concat(dfs, ignore_index = True)
    return df

def tcp_abnormal_processs(df:pd.DataFrame) -> pd.DataFrame:

    df['Path'] = df['Path'].fillna(df['Raw_load'])
    df = df.dropna(thresh = 0.4 * df.shape[0], axis = 1) # drop cols with too many na value

    not_process = [
        #'Cookie',
        'Date',
        'Path',
        'Host',
        'Accept',
        'Connection',
        'TCP_options_Timestamp',
        'Via',
        'Server',
        'Referer',
        'Padding_load',
        'Raw_load'
    ]
    df = df.drop(not_process, axis = 1)

    drop_cols = [x for x in df.columns if len(df[x].unique()) < 2 or len(df[x].unique()) >= 0.99 * df.shape[0]]
    df = df.drop(drop_cols, axis = 1)

    df['Age'] = df['Age'].fillna('0').apply(lambda x:re.sub(r'[^0-9]','', x)) # special process

    df = df.fillna(0) #fill na 

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
        'Age'
    ]

    df[num_cols] = df[num_cols].applymap(lambda x : int(x, 16) if type(x) == str else x)
    
    return df


def basic_data() -> pd.DataFrame:

    abnormal_ipv4_files = [
        #'Ethernet_IP_TCP_HTTP 1.csv',
        'Ethernet_IP_TCP_HTTP 1_HTTP Request.csv',
        'Ethernet_IP_TCP_HTTP 1_HTTP Response.csv',
    ]

    abnormal_ipv6_files = [
        #'Ethernet_IPv6_TCP_HTTP 1.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Request.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Response.csv',
    ]

    normal_ipv6_files = [
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Request.csv',
        'Ethernet_IPv6_TCP_HTTP 1_HTTP Response.csv',
    ]

    abnormal_ipv4 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv4_files]
    abnormal_ipv4 = [ipv42ipv6(df) for df in abnormal_ipv4] 
    abnormal_ipv6 = [pd.read_csv(os.path.join('output/abnormal', f)) for f in abnormal_ipv6_files]
    normal_ipv6 = [pd.read_csv(os.path.join('output/normal', f)) for f in normal_ipv6_files]

    dfs = abnormal_ipv4 + abnormal_ipv6 + normal_ipv6

    df = pd.concat(dfs, ignore_index = True)
    return df


def basic_process(df: pd.DataFrame) -> pd.DataFrame :

    df = df.dropna(thresh = 0.4 * df.shape[0], axis = 1) # drop cols with too many na value

    not_process = [
        #'Cookie',
        'Date',
        'Path',
        'Host',
        'Accept',
        'Connection',
        'TCP_options_Timestamp',
        'Via',
        'Server',
        'Referer'
    ]
    df = df.drop(not_process, axis = 1)

    drop_cols = [x for x in df.columns if len(df[x].unique()) < 2 or len(df[x].unique()) >= 0.99 * df.shape[0]]
    df = df.drop(drop_cols, axis = 1)

    df['Age'] = df['Age'].fillna('0').apply(lambda x:re.sub(r'[^0-9]','', x)) # special process

    df = df.fillna(0) #fill na 

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
        'Age'
    ]

    df[num_cols] = df[num_cols].applymap(lambda x : int(x, 16) if type(x) == str else x)
    
    return df