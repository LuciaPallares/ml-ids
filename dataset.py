from typing import Protocol
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


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
attacks = ['back','buffer_overflow','ftp_write','guess_passwd','imap','ipsweep','land','loadmodule','multihop','neptune','nmap','perl','phf','pod','portsweep',
            'rootkit','satan','smurf','spy','teardrop','warezclient','warezmaster']
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

    tot = normal + attacks
    print("Number of attributes: ",(len(train_data.columns)))
    print("Number of samples that represent ATTACKS: ", attacks,";Porcentaje:", round(attacks*100/(normal+attacks),3)," %")
    print("Number of samples that represent NORMAL TRAFFIC: ",normal, ";Porcentaje:", round(normal*100/(normal+attacks),3)," %")
    print("Number of total samples: ", normal + attacks)
    print("Number of DoS attacks: ", dos, ";Porcentaje:", round(dos*100/tot,3)," %")
    print("Number of PROBE attacks: ", probe, ";Porcentaje:", round(probe*100/tot,3)," %")
    print("Number of R2L attacks: ", r2l, ";Porcentaje:", round(r2l*100/tot,3)," %" )
    print("Number of U2R attacks: ", u2r, ";Porcentaje:", round(u2r*100/tot,3)," %")
    print("--------------------------------------------------------")
    print("Number of BACK attacks: ", back, ";Porcentaje:", round(back*100/tot,3)," %")
    print("Number of BUFFER OVERFLOW attacks: ", buffer_overflow, ";Porcentaje:", round(buffer_overflow*100/tot,3)," %")
    print("Number of FTP WRITE attacks: ",  ftp_write,  ";Porcentaje:",round(ftp_write*100/tot,3)," %")
    print("Number of GUESS PASSWD attacks: ",  guess_passwd,  ";Porcentaje:",round(guess_passwd*100/tot,3)," %")
    print("Number of IMAP attacks: ", imap,  ";Porcentaje:",round(imap*100/tot,3)," %")
    print("Number of IPSWEEP attacks: ", ipsweep, ";Porcentaje:", round(ipsweep*100/tot,3)," %")
    print("Number of LAND attacks: ", land,  ";Porcentaje:",round(land*100/tot,3)," %")
    print("Number of LOADMODULE attacks: ", loadmodule, ";Porcentaje:", round(loadmodule*100/tot,3)," %")
    print("Number of MULTIHOP attacks: ", multihop,  ";Porcentaje:",round(multihop*100/tot,3)," %")
    print("Number of NEPTUNE attacks: ", neptune,  ";Porcentaje:",round(neptune*100/tot,3)," %")
    print("Number of NMAP attacks: ", nmap,  ";Porcentaje:",round(nmap*100/tot,3)," %")
    print("Number of PERL attacks: ", perl, ";Porcentaje:", round(perl*100/tot,3)," %")
    print("Number of PHF attacks: ", phf,  ";Porcentaje:",round(phf*100/tot,3)," %")," %"
    print("Number of POD attacks: ", pod,  ";Porcentaje:",round(pod*100/tot,3)," %")
    print("Number of PORTSWEEP attacks: ", portsweep, ";Porcentaje:",round(portsweep*100/tot,3)," %")
    print("Number of ROOTKIT attacks: ", rootkit,  ";Porcentaje:",round(rootkit*100/tot,3)," %")
    print("Number of SATAN attacks: ", satan,  ";Porcentaje:",round(satan*100/tot,3)," %")
    print("Number of SMURF attacks: ", smurf,  ";Porcentaje:",round(smurf*100/tot,3)," %")
    print("Number of SPY attacks: ", spy,  ";Porcentaje:",round(spy*100/tot,3)," %")
    print("Number of TEARDROP attacks: ", teardrop, ";Porcentaje:", round(teardrop*100/tot,3)," %")
    print("Number of WAREZCLIENT attacks: ", warezclient, ";Porcentaje:", round(warezclient*100/tot,3)," %")
    print("Number of WAREZMASTER attacks: ", warezmaster,  ";Porcentaje:",round(warezmaster*100/tot,3)," %")
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.bar(['dos \n {} %'.format(round(dos*100/tot,3))], [dos*100/tot], color="r")
    ax.bar(['probe \n {} %'.format(round(probe*100/tot,3))], [probe*100/tot], color="g")
    ax.bar(['r2l \n {} %'.format(round(r2l*100/tot,3))], [r2l*100/tot], color="b")
    ax.bar(['u2r \n {} %'.format(round(u2r*100/tot,3))], [u2r*100/tot], color="y")
    plt.ylabel('total % ')    

    fig.savefig('figures/type_attack.png')

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
    for i in poss_attr: #Attribute
        for j in poss_attr[i]: #Value
            all_values = data[data[i] == j] #All the samples with a concrete value for an attribute
            if((all_values['class'] == 'normal').all() or (all_values['class'] != 'normal').all()):
                at_val.append([i,j])
                total_outliers += len(all_values.index)

    #Check if of the samples calculated are duplicaded, i.e, if all the samples that are 'flag' == 'RSTOS0' are also 'service' == 'aol' we don't need to count them
    #individually
    rep = 0
    at_val_aux = copy.deepcopy(at_val)
    for i in at_val_aux:    
        aux = data[data[i[0]] == i[1]]
        for j in at_val_aux:
            if(j[0] != i[0]):
                rep += len((aux[aux[j[0]] == j[1]]).index)
        at_val_aux.remove([i[0],i[1]])
    total_outliers -= rep
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

    aux_data.reset_index(drop=True, inplace=True)

    return aux_data
def one_hot_encod(train_data, test_data):
    feat_2_enc = ['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login']
    aux_train = copy.deepcopy(train_data)
    aux_test = copy.deepcopy(test_data)
    train_subset_2_enc = aux_train.loc[:,feat_2_enc]
    test_subset_2_enc = aux_test.loc[:,feat_2_enc]
    aux_train.drop(columns=['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login'],inplace = True)
    aux_test.drop(columns=['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login'],inplace = True)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_subset_2_enc.values)
    encoded_train_data = enc.transform(train_subset_2_enc.values).toarray()
    encoded_test_data = enc.transform(test_subset_2_enc.values).toarray()
    #cat = [  i for x in enc.categories_ for i in x if(i!=0 and i!=1)]
    cat = []
    count = 0
    for x in enc.categories_:
        for i in x:
            if(i != 0 and i != 1):
                cat.append(i)
            else:
                if(count == 0):
                    cat.append('land_{}'.format(i))
                if(count == 1):
                    cat.append('logged_in_{}'.format(i))
                if(count == 2):
                    cat.append('is_host_login_{}'.format(i))
                if(count == 3):
                    cat.append('is_guest_login_{}'.format(i))
                count=+1

    train_OneHot = pd.DataFrame(encoded_train_data, columns = cat)
    test_OneHot = pd.DataFrame(encoded_test_data, columns = cat)
    train_OneHot = pd.concat([train_OneHot, aux_train], axis = 1)
    test_OneHot = pd.concat([test_OneHot, aux_test], axis = 1)

    return [train_OneHot, test_OneHot]

def compute_pca(train_data, test_data, data_type):
    #Here we are first applying a correspondence between the nominal values of the attribute and a number in order to apply then a PCA.
    #We need to apply this only on the attributes that are numeric so we need to select those (drop protocol, service and flag) 
    aux_train = copy.deepcopy(train_data)
    aux_test = copy.deepcopy(test_data)
    if(data_type == 'wo_out'):
        #Comment for hot encoding
        aux_train.drop(columns=['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login'],inplace = True)
        aux_test.drop(columns=['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login'],inplace = True)
        features_train = list(aux_train.columns)
        features_test = list(aux_test.columns)
        #Remove'class' from features. Comment for hot encoding
        features_train = features_train[:-1]
        features_test = features_test[:-1]
    else:
        features_train = list(aux_train.columns)
        features_test = list(aux_test.columns)
        features_train.remove('class')
        features_test.remove('class')

    #Apply normalization based on standard deviation
    
    x_train = aux_train.loc[:, features_train].values
    x_test = aux_test.loc[:,features_test].values
    
    #Extract 'class' attribute
    y_train = aux_train.loc[:,['class']].values
    y_test = aux_test.loc[:,['class']].values

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    #Select the number of components.
    #To select the number of components we will apply PCA for different number of components and will see the variance explained for each
    
    #The number of attributes introduced will be given by the length of features
    #tot_att = len(features)
    #print(tot_att)
    #exp_var_r = []
    #ran = np.arange(1,tot_att)
    #for n in ran:
    #    pca_n = PCA(n_components=n).fit(x)
    #    exp_var_r.append(round(sum(list(pca_n.explained_variance_ratio_))*100,2)) #Variance ratio explained by the n components

    #print(exp_var_r)
    #fig = plt.figure(figsize=(15,10))
    #ax = fig.add_subplot(1,1,1)
    #ax.set_xlabel('number of components')   
    #ax.set_ylabel('total variance explained in %')
    #plt.ylim(0,110)
    #plt.xticks(ran)
    #ax.plot(ran,exp_var_r, marker='o', linestyle='--', color='b')
    #ax.axhline(y=95, color='r', linestyle='-')
    #ax.axvline(x=21, color='g', linestyle='--')
    #ax.text(1, 90, '95%', color = 'red', fontsize=16)
    #ax.text(21, 90, '21 components', color = 'g', fontsize=16)
    #fig.savefig('figures/pca_comp.png')


    ##Apply PCA
    pca = PCA(n_components=0.95) #Here we esablish that we want an explained variance ratio above 95%
    principalComponents_train = pca.fit_transform(x_train)
    principalComponents_test = pca.transform(x_test)
    print('Total variance explained after applying PCA for {} is {}'.format(data_type,round(sum(list(pca.explained_variance_ratio_))*100, 2)))
    print('Number of components used to achieve this {}'.format(pca.n_components_))
    col = ['component {}'.format(i) for i in np.arange(1,(pca.n_components_ +1))]

    principalDf_train = pd.DataFrame(data = principalComponents_train, columns = col)
    
    principalDf_test = pd.DataFrame(data = principalComponents_test, columns = col)
    
    if(data_type == 'wo_out'):
        finalDf_train = pd.concat([principalDf_train, aux_train['class']], axis = 1)
        finalDf_test = pd.concat([principalDf_test, aux_test['class']], axis = 1)
        #finalDf_train = pd.concat([train_data['protocol_type'],train_data['service'],train_data['flag'],train_data['land'],train_data['logged_in'],
        #        train_data['is_host_login'], train_data['is_guest_login'],principalDf_train, aux_train['class']], axis = 1)
        #finalDf_test = pd.concat([test_data['protocol_type'],test_data['service'],test_data['flag'],test_data['land'],test_data['logged_in'],
        #        test_data['is_host_login'], test_data['is_guest_login'],principalDf_test, aux_test['class']], axis = 1)
    else:
        finalDf_train = pd.concat([principalDf_train, aux_train['class']], axis = 1)
        finalDf_test = pd.concat([principalDf_test, aux_test['class']], axis = 1)

    return [finalDf_train, finalDf_test]



def compute_pearson_corr(data, data_type):
    #We need to apply this only on the attributes that are numeric so we need to select those (drop protocol, service and flag) 
    #We will return the correlation matrix
    dat = copy.deepcopy(data)
    if(data_type == 'wo_out'):
        dat.drop(columns=['protocol_type', 'service', 'flag','land', 'logged_in','is_host_login','is_guest_login'],inplace = True)

    c = dat.corr(method='pearson').abs() 
    
    #plt.figure(figsize=(20,15))
    #sns.set(font_scale = 0.5)
    #hm = sns.heatmap(c, annot=True, vmin=-1, vmax=1)
    #hm.set_title('Correlation heatmap')
    #plt.savefig('figures/heatmap.png')
    #print(c)

    #Select only an interesting subset of the correlation matrix to show (from 'count' to the end) In case we want an image to show
    #subs = c.copy(deep=True)
    #shape = subs.shape
    #subs.drop(subs.loc[:,'duration':'num_outbound_cmds'], axis = 1, inplace = True)
    #red_colu = len(subs.columns)
    #rows_to_remove = shape[0]- red_colu
    #subs.drop(subs.index[:rows_to_remove], inplace = True)
    #plt.figure(figsize=(20,15))
    #sns.set(font_scale = 0.5)
    #hm_s = sns.heatmap(subs, annot=True, vmin=-1, vmax=1)
    #hm_s.set_title('Subset of correlation heatmap')
    #plt.savefig('figures/heatmap_subset.png')
    
    return c

def att_pearson_corr(train_data, test_data, cor, data_type):
    #First we need to establish a threshold 
    #If the Pearson coefficient between two attributes if above this threshold we will remove one as we will consider that it's enough information in one
    aux_train = copy.deepcopy(train_data)
    aux_test = copy.deepcopy(test_data)
    threshold = 0.7
    #Create a list to store a bool value that will say if the value we are studying is above the threshold:
    colum = np.full((cor.shape[0],), True, dtype=bool)
    #Iterate the dataframe to find the columns that have values above the threshold
    for i in range(cor.shape[0]):
        for j in range(i+1, cor.shape[0]):
            if cor.iloc[i,j] >= threshold and colum[j]:
                colum[j] = False
    #In column we have set to false the columns that we are not going to keep as they have a pearson coefficient above the threshold
    selected_columns = cor.columns[colum]
    aux_train = aux_train[selected_columns]
    aux_test = aux_test[selected_columns]
    if(data_type == 'wo_out'):
        aux_train = pd.concat([aux_train, train_data['class']], axis = 1)
        aux_test = pd.concat([aux_test, test_data['class']], axis = 1)
        #aux_train = pd.concat([train_data['protocol_type'],train_data['service'],train_data['flag'],train_data['land'],train_data['logged_in'],
        #        train_data['is_host_login'], train_data['is_guest_login'], aux_train, train_data['class'] ], axis = 1)
        #aux_test = pd.concat([test_data['protocol_type'],test_data['service'],test_data['flag'],test_data['land'],test_data['logged_in'],
        #        test_data['is_host_login'], test_data['is_guest_login'], aux_test, test_data['class'] ], axis = 1)
        
    else:
        aux_train = pd.concat([aux_train, train_data['class']], axis = 1)
        aux_test = pd.concat([aux_test, test_data['class']], axis = 1)
    return [aux_train, aux_test]

        

