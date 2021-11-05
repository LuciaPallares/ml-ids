import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

poss_attr = {'protocol_type': ['tcp','udp', 'icmp'], 'service' : ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
            'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 
            'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
            'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 
            'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'], 
            'flag' : [ 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'],'land': ['0', '1'], 'logged_in' : ['0', '1'],'is_host_login' :['0','1'], 
            'is_guest_login' : ['0','1']}

def calculate_totals(train_data):
    #Calculate the total number of samples, the samples that correspond to attacks and how many samples are for each attack
    attacks = 0
    back = 0
    buffer_overflow = 0
    ftp_write = 0
    guess_passwd = 0
    imap = 0
    ipsweep = 0
    land = 0
    loadmodule = 0
    multihop = 0
    neptune = 0
    nmap = 0
    normal = 0
    perl = 0
    phf = 0
    pod = 0
    portsweep = 0
    rootkit = 0
    satan = 0
    smurf = 0
    spy = 0
    teardrop = 0
    warezclient = 0
    warezmaster = 0
    
    #Get the number of samples for each type:

    attacks = len(train_data[train_data['class'] != 'normal'].index)
    normal = len(train_data[train_data['class'] == 'normal'].index)
    back = len(train_data[train_data['class'] == 'back'].index)
    buffer_overflow = len(train_data[train_data['class'] == 'buffer_overflow'].index)
    ftp_write = len(train_data[train_data['class'] == 'ftp_write'].index)
    guess_passwd = len(train_data[train_data['class'] == 'guess_passwd'].index)
    imap = len(train_data[train_data['class'] == 'imap'].index)
    ipsweep = len(train_data[train_data['class'] == 'ipsweep'].index)
    land = len(train_data[train_data['class'] == 'land'].index)
    loadmodule = len(train_data[train_data['class'] == 'loadmodule'].index)
    multihop = len(train_data[train_data['class'] == 'multihop'].index)
    neptune = len(train_data[train_data['class'] == 'neptune'].index)
    nmap = len(train_data[train_data['class'] == 'nmap'].index)
    perl = len(train_data[train_data['class'] == 'perl'].index)
    phf = len(train_data[train_data['class'] == 'phf'].index)
    pod = len(train_data[train_data['class'] == 'pod'].index)
    portsweep = len(train_data[train_data['class'] == 'portsweep'].index)
    rootkit = len(train_data[train_data['class'] == 'rootkit'].index)
    satan = len(train_data[train_data['class'] == 'satan'].index)
    smurf = len(train_data[train_data['class'] == 'smurf'].index)
    spy = len(train_data[train_data['class'] == 'spy'].index)
    teardrop = len(train_data[train_data['class'] == 'teardrop'].index)
    warezclient = len(train_data[train_data['class'] == 'warezclient'].index)
    warezmaster = len(train_data[train_data['class'] == 'warezmaster'].index)
    
    #Attack types
    dos = back + land + neptune + pod + smurf + teardrop
    u2r = buffer_overflow + loadmodule + perl + rootkit
    r2l = ftp_write + guess_passwd + imap + multihop + phf + spy + warezclient + warezmaster
    probe = ipsweep + nmap + portsweep + satan

    
    print("Number of attributes: ",len(train_data.columns))
    print("Number of samples that represent ATTACKS: ", attacks)
    print("Number of samples that represent NORMAL TRAFFIC: ", normal)
    print("Number of total samples: ", normal + attacks)
    print("Number of DoS attacks: ", dos)
    print("Number of PROBE attacks: ", probe)
    print("Number of R2L attacks: ", r2l)
    print("Number of U2R attacks: ", u2r)
    print("--------------------------------------------------------")
    print("Number of BACK attacks: ", back)
    print("Number of BUFFER OVERFLOW attacks: ", buffer_overflow)
    print("Number of FTP WRITE attacks: ", ftp_write)
    print("Number of GUESS PASSWD attacks: ", guess_passwd)
    print("Number of IMAP attacks: ", imap)
    print("Number of IPSWEEP attacks: ", ipsweep)
    print("Number of LAND attacks: ", land)
    print("Number of LOADMODULE attacks: ", loadmodule)
    print("Number of MULTIHOP attacks: ", multihop)
    print("Number of NEPTUNE attacks: ", neptune)
    print("Number of NMAP attacks: ", nmap)
    print("Number of PERL attacks: ", perl)
    print("Number of PHF attacks: ", phf)
    print("Number of POD attacks: ", pod)
    print("Number of PORTSWEEP attacks: ", portsweep)
    print("Number of ROOTKIT attacks: ", rootkit)
    print("Number of SATAN attacks: ", satan)
    print("Number of SMURF attacks: ", smurf)
    print("Number of SPY attacks: ", spy)
    print("Number of TEARDROP attacks: ", teardrop)
    print("Number of WAREZCLIENT attacks: ", warezclient)
    print("Number of WAREZMASTER attacks: ", warezmaster)

def compare_att_2_type(pd, attribute):
    #Compare the attributes to the type of data (normal/attack)
    #Receives the attribute that wants to be studied
    #attacks_pd = dataframe with all the samples that are classified as attacks
    #total_attack = number of samples classified as attacks
    #att_values = possible values that a certain attribute can have
    #total_samples_attrib = total samples with the attribute value to study
    #match = number of samples that are an attacks with an specified attribute value
    #Returns in a list the number of attacks that have that value attribute, the percentage above the total of attacks, the percentage that are attacks above the total samples
    #that contain that attribute value and the percentage that are not attacks respect to the total of samples with that value
    

    attacks_pd = pd[pd['class'] != 'normal']
    not_attacks_pd = pd[pd['class'] == 'normal']
    total_attakcs = len(pd[pd['class'] != 'normal'].index)
    
    #Create list with the possible values that the attribute can have
    att_values = poss_attr[attribute]
    
    value_att = {}
    for i in att_values:
        match = len(attacks_pd[attacks_pd[attribute] == i].index)
        n_at_m = len(not_attacks_pd[not_attacks_pd[attribute] == i].index)
        total_samples_attrib = len(pd[pd[attribute] == i].index)
        value_att[i] = [match]
        value_att[i].append((match/total_attakcs)*100)
        if(total_samples_attrib != 0):
            value_att[i].append((match/total_samples_attrib)*100)
            value_att[i].append((n_at_m/(match + n_at_m))*100)
        else:
            value_att[i].append(0)
            value_att[i].append(0)
        
        

    return value_att

def draw_histogram(data, attrib, at_value):
    if(attrib not in poss_attr.keys()):
        err = "Attribute not found"
        return err
    else:
        if(at_value not in poss_attr[attrib]):
            err = "Not a possible value"
            return err
        else:
            at = data[data['class'] != 'normal']
            go = data[data['class'] == 'normal']
            #Total of samples with that value
            tot_val = len(data[data[attrib] == at_value].index)
            #Total attacks with that value
            at_val = len(at[at[attrib] == at_value].index)
            go_val = len(go[go[attrib] == at_value].index)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            plt.ylim(0,tot_val)
            ax.bar(['attacks \n {}'.format(at_val)], [at_val], color="r")
            ax.bar(['good \n {}'.format(go_val)], [go_val], color="g")
            
            fig.savefig('figures/{}-{}.png'.format(attrib,at_value))
            corr = "Figure saved!"
            return corr

def get_outliers(data):
    total_outliers = 0
    for i in poss_attr: 
        for j in poss_attr[i]:
            all_values = data[data[i] == j] #All the samples with a concrete value for an attribute
            if((all_values['class'] == 'normal').all() or (all_values['class'] != 'normal').all()):
                total_outliers += len(all_values.index)
    return total_outliers


    
            

def main():
    train_data = pd.read_csv("data/KDDTrain+.txt")
    
    train_data.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','?']#

    
    #f = open("stats/stats4nsl.txt", "w")
    
    
    
    calculate_totals(train_data)
    #protocol_type = compare_att_2_type(train_data,'protocol_type')
    #print("For protocol_type attribute: ",protocol_type )
    #f.write(str(protocol_type))
    #f.write("\n \n")
    #print("--------------------------------------------")
    #service = compare_att_2_type(train_data,'service')
    #print("For service attribute: ", service)
    #f.write(str(service))
    #f.write("\n \n")
    #print("--------------------------------------------")
    #flag = compare_att_2_type(train_data,'flag')
    #print("For flag attribute: ",flag)
    #f.write(str(flag))
    #f.write("\n \n")
    #print("--------------------------------------------")
    #logged_in = compare_att_2_type(train_data,'logged_in')
    #print("For logged_in attribute: ",logged_in)
    #print("--------------------------------------------")
    #is_host_login = compare_att_2_type(train_data,'is_host_login')
    #print("For is_host_login attribute: ",is_host_login)
    #print("--------------------------------------------")
    #is_guest_login = compare_att_2_type(train_data,'is_guest_login')
    #print("For is_guest_login attribute: ",is_guest_login)
    #print("--------------------------------------------")
    #f.close()

    #print(len(train_data.index))
    #shown_at = []
    #for i in train_data.index:
    #    if(train_data['class'][i] not in shown_at and train_data['class'][i] != 'normal'):
    #        shown_at.append(train_data['class'][i])

    #print(shown_at)
    #print(len(shown_at))
    ##Get the number of outliers; atributes that for a value are all atacks or all normal
    print(get_outliers(train_data))
    
    ##Code to obtain all the histograms for non numeric data
    #for i in poss_attr:
    #    for j in poss_attr[i]:
    #        print(draw_histogram(train_data,i,j))
    
    #print(get_outliers(train_data))
    #print(train_data)
    
if __name__ == '__main__':
    main()


