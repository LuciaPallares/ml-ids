from dataset import *
from datetime import datetime
from algorithms import *



def main():
    print('-----------------------------------------------------------------------')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n \n Timestamp =", dt_string," \n \n")	

    print('-----------------------------------------------------------------------')
    train_data = pd.read_csv("data/KDDTrain+.txt")
    
    train_data.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','?']
    
    train_data.drop(columns=['?'], inplace=True)

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
    p_outl = round((outl/len(train_data.index))*100,3)
    print("Percentage of outliers over the total of samples: {}".format(p_outl), "%")
    
    ##Remove the outliers obtained in the list at_val
    data_wo_outliers = remove_outliers(train_data,at_val)
    print("Total samples once outliers are removed: {}". format(len(data_wo_outliers.index)))
    print(data_wo_outliers)
    data_one_hot_enc = pd.get_dummies(data_wo_outliers, columns=['protocol_type','service','flag', 'land', 'logged_in','is_host_login','is_guest_login'], prefix=['protocol','service','flag','land','log_in','host_login','guest_login'])
    #print(data_one_hot_enc)
    
    ##Principal component analysis PCA
    #Data types : wo_out -> Data without outliers (data_wo_outliers); hot_enc -> Hot encoding applied (data_one_hot_enc)
    data_pca_wo_out = compute_pca(data_wo_outliers,'wo_out')
    data_pca_one_hot = compute_pca(data_one_hot_enc,'hot_enc')
    print('Dataframe data_pca_wo_out after applying PCA: \n')
    print(data_pca_wo_out)
    print('Dataframe data_pca_one_hot after applying PCA: \n')
    print(data_pca_one_hot)

    ##Pearson correlation analysis
    corr_wo_out = compute_pearson_corr(data_wo_outliers, 'wo_out')
    corr_one_hot = compute_pearson_corr(data_one_hot_enc, 'hot_enc')

    data_pears_wo_out = att_pearson_corr(data_wo_outliers,corr_wo_out,'wo_out')
    data_pears_one_hot = att_pearson_corr(data_one_hot_enc,corr_one_hot,'hot_enc')
    print('Dataframe data_pears_wo_out after applying reduction based on Pearson coefficient: \n')
    print(data_pears_wo_out)
    print('Dataframe data_pears_one_hot after applying reduction based on Pearson coefficient: \n')
    print(data_pears_one_hot)



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
