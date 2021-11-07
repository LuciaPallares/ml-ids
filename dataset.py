import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

poss_attr = {'protocol_type': ['tcp','udp', 'icmp'], 'service' : ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
            'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 
            'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
            'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 
            'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'], 
            'flag' : [ 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'],'land': [0, 1], 'logged_in' : [0, 1],'is_host_login' :[0,1], 
            'is_guest_login' : [0, 1]}
real_attr = ['duration', 'src_bytes','dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

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
    #Returns in a list the number of attacks that have that value attribute, the percentage over the total of attacks, the percentage that are attacks over the total samples
    #that contain that attribute value and the percentage that are not attacks over the total of samples with that value
    

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

def non_num_histogram(data, attrib, at_value):
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
            fig.savefig('figures/non-numeric/{}-{}.png'.format(attrib,at_value))
            plt.close()
            corr = "Figure saved!" + str(at_value)
            return corr

def get_outliers(data):
    #Calculate the number of outliers (total_outliers) and a list with a pair attribute-value of those that are outliers (at_val)
    total_outliers = 0
    at_val = []
    for i in poss_attr: 
        for j in poss_attr[i]:
            all_values = data[data[i] == j] #All the samples with a concrete value for an attribute
            if((all_values['class'] == 'normal').all() or (all_values['class'] != 'normal').all()):
                at_val.append([i,j])
                total_outliers += len(all_values.index)
    print('The total number of outliers is: {}'.format(total_outliers))
    return total_outliers,at_val

def num_histograms(data):
    #Study each attribute with a continuous value and see how many samples are attacks and how many are not depending on the attribute
    #One histogram represents the samples that are attacks 
    #The other histogram represents the samples that are legitimate traffic
    #The x ascis are values between the minimum and the maximum values of each attribute
    for i in real_attr:    
        att = data[data['class'] != 'normal']
        not_att = data[data['class'] == 'normal']
        min_at = min(data[i])
        max_at = max(data[i])
        fig = plt.figure()
        bins = np.linspace(min_at, max_at)
        x = att[i]
        y = not_att[i]
        plt.hist(x, bins, alpha=0.5, label='attacks')
        plt.hist(y, bins, alpha=0.5, label='normal')
        plt.legend(loc='upper right')
        fig.savefig('figures/numeric/{}.png'.format(i))
        plt.close()

def remove_outliers(data, at_value):
    aux_data = data
    for i in at_value:
        aux_data = aux_data.loc[aux_data[i[0]] != i[1]]

    return aux_data



    
            

def main():
    train_data = pd.read_csv("data/KDDTrain+.txt")
    
    train_data.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','?']

    ##Compute the number of samples that represent attacks, the number of samples that are legitimate traffic, and how many samples are for each type of attack
    calculate_totals(train_data)

    #f = open("stats/stats4nsl.txt", "w")
    
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

    

    ##Get the number of outliers; atributes that for a value are all atacks or all normal
    outl, at_val = get_outliers(train_data)
    p_outl = (outl/len(train_data.index))*100
    print("Percentage of outliers over the total of samples: {}".format(p_outl))
    
    ##Remove the outliers obtained in the list at_val
    data_wo_outliers = remove_outliers(train_data,at_val)
    print(len(data_wo_outliers.index))
    print(len(train_data.index))
   
    ##Obtain all the histograms for numeric values of the attributes
    #num_histograms(train_data)
    
    ##Obtain all the histograms for non numeric values of the attributes
    #for i in poss_attr:
    #    for j in poss_attr[i]:
    #        print(non_num_histogram(train_data,i,j))

    #att = train_data[train_data['class'] != 'normal']
    #not_att = train_data[train_data['class'] == 'normal']

    #min_at1 = min(train_data['dst_host_srv_count'])
    #max_at1 = max(train_data['dst_host_srv_count'])
    #bins = np.linspace(min_at1, max_at1)
    #x1 = att['dst_host_srv_count']
    #y1 = not_att['dst_host_srv_count']
    
    #fig = plt.figure()
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #fig.set_figheight(5)
    #fig.set_figwidth(15)
    #ax1.hist(x1, bins, alpha=0.5, label='attacks')
    #ax1.hist(y1, bins, alpha=0.5, label='normal')
    #ax1.legend(loc='upper right')
    #ax1.set_title('dst_host_srv_count')
    
    #min_at2 = min(train_data['dst_host_diff_srv_rate'])
    #max_at2 = max(train_data['dst_host_diff_srv_rate'])
    #bins2 = np.linspace(min_at2, max_at2)
    #x2 = att['dst_host_diff_srv_rate']
    #y2 = not_att['dst_host_diff_srv_rate']
    #ax2.hist(x2, bins2, alpha=0.5, label='attacks')
    #ax2.hist(y2, bins2, alpha=0.5, label='normal')
    #ax2.legend(loc='upper right')
    #ax2.set_title('dst_host_diff_srv_rate')

    #min_at3 = min(train_data['srv_count'])
    #max_at3= max(train_data['srv_count'])
    #bins3 = np.linspace(min_at3, max_at3)
    #x3 = att['srv_count']
    #y3 = not_att['srv_count']
    #ax3.hist(x3, bins3, alpha=0.5, label='attacks')
    #ax3.hist(y3, bins3, alpha=0.5, label='normal')
    #ax3.legend(loc='upper right')
    #ax3.set_title('srv_count')
    #fig.savefig('numeric-ex.png')
    #plt.close()
   
    
if __name__ == '__main__':
    main()


