from dataset import *
from datetime import datetime
from algorithms import *
from sklearn import metrics



def main():
    print('-----------------------------------------------------------------------')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n \n Timestamp =", dt_string," \n \n")	

    print('-----------------------------------------------------------------------')
    train_data = pd.read_csv("data/KDDTrain+.txt")
    test_data = pd.read_csv("data/KDDTest+.txt")
    
    train_data.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','?']
    
    test_data.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','?']

    train_data.drop(columns=['?'], inplace=True)
    test_data.drop(columns=['?'], inplace=True)

    print('-----------------------------------------------------------------------')
    print('\n\n PREPARING DATA \n\n')
    print('-----------------------------------------------------------------------')
    ##Compute the number of samples that represent attacks, the number of samples that are legitimate traffic, and how many samples are for each type of attack
    print('\n\n ----------------- INFORMATION ABOUT THE DATA -----------------\n\n')
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

    print('\n\n ----------------- REMOVING BIASED DATA -----------------\n\n')

    ##Get the number of outliers; atributes that for a value are all atacks or all normal
    train_outl, train_at_val = get_outliers(train_data)
    train_p_outl = round((train_outl/len(train_data.index))*100,3)
    print("Percentage of outliers over the total of samples on the train set: {}".format(train_p_outl), "%")
    test_outl, test_at_val = get_outliers(test_data)
    test_p_outl = round((test_outl/len(train_data.index))*100,3)
    print("Percentage of outliers over the total of samples on the test set: {}".format(test_p_outl), "%")
    
    ##Remove the outliers obtained in the list at_val
    train_data_wo_outliers = remove_outliers(train_data,train_at_val)
    print("Total samples once outliers are removed: {}". format(len(train_data_wo_outliers.index)))
    print('Biased samples removed ---> RESULT: train_data_wo_outliers')

    test_data_wo_outliers = remove_outliers(test_data,test_at_val)
    print("Total samples once outliers are removed: {}". format(len(test_data_wo_outliers.index)))
    print('Biased samples removed ---> RESULT: test_data_wo_outliers')
    
    print('\n\n ----------------- PERFORMING ONE HOT ENCODING -----------------\n\n')
    
    
    #train_data_one_hot_enc = pd.get_dummies(train_data_wo_outliers, columns=['protocol_type','service','flag', 'land', 'logged_in','is_host_login','is_guest_login'], prefix=['protocol','service','flag','land','log_in','host_login','guest_login'])
    #test_data_one_hot_enc = pd.get_dummies(test_data_wo_outliers, columns=['protocol_type','service','flag', 'land', 'logged_in','is_host_login','is_guest_login'], prefix=['protocol','service','flag','land','log_in','host_login','guest_login'])
    [train_data_one_hot_enc, test_data_one_hot_enc] = one_hot_encod(train_data_wo_outliers, test_data_wo_outliers)
    print('One hot encoded performed on train_data_wo_outliers ---> RESULT: train_data_one_hot_enc')
    print('One hot encoded performed on test_data_wo_outliers ---> RESULT: test_data_one_hot_enc')


    print('\n\n ----------------- PERFORMING PCA -----------------\n\n')

    ##Principal component analysis PCA
    #Data types : wo_out -> Data without outliers (train_data_wo_outliers); hot_enc -> Hot encoding applied (train_data_one_hot_enc)
    
    [train_data_pca_wo_out, test_data_pca_wo_out] = compute_pca(train_data_wo_outliers,test_data_wo_outliers,'wo_out')
    
    print('Reduction based on PCA to train_data_wo_outliers applied ---> RESULT: train_data_pca_wo_out')
    print('Reduction based on PCA to test_data_wo_outliers applied ---> RESULT: test_data_pca_wo_out')

    #print(train_data_pca_wo_out)
    [train_data_pca_one_hot, test_data_pca_one_hot] = compute_pca(train_data_one_hot_enc,test_data_one_hot_enc,'hot_enc')
     
    print('Reduction based on PCA to train_data_one_hot_enc applied ---> RESULT: train_data_pca_one_hot')
    print('Reduction based on PCA to test_data_one_hot_enc applied ---> RESULT: test_data_pca_one_hot')
    #print(train_data_pca_one_hot)

    print('\n\n ----------------- PERFORMING REDUCTION BASED ON PEARSON CORRELATION  -----------------\n\n')
    
    ##Pearson correlation analysis
    corr_wo_out = compute_pearson_corr(train_data_wo_outliers, 'wo_out')
    corr_one_hot = compute_pearson_corr(train_data_one_hot_enc, 'hot_enc')

    [train_data_pears_wo_out, test_data_pears_wo_out] = att_pearson_corr(train_data_wo_outliers,test_data_wo_outliers, corr_wo_out,'wo_out')
    [train_data_pears_one_hot, test_data_pears_one_hot] = att_pearson_corr(train_data_one_hot_enc,test_data_one_hot_enc, corr_one_hot,'hot_enc')
    
    print('Reduction based on Pearson coefficient to train_data_wo_out applied ---> RESULT: train_data_pears_wo_out')
    print('Reduction based on Pearson coefficient to test_data_wo_out applied ---> RESULT: test_data_pears_wo_out')
    #print(train_data_pears_wo_out)
    print('Reduction based on Pearson coefficient to train_data_one_hot_enc applied ---> RESULT: train_data_pears_one_hot')
    print('Reduction based on Pearson coefficient to test_data_one_hot_enc applied ---> RESULT: test_data_pears_one_hot')
    #print(train_data_pears_one_hot)

    print('-----------------------------------------------------------------------')
    print('\n\n APPLYING TRAINING ALGORITHMS \n\n')
    print('-----------------------------------------------------------------------')

    #We will select 3 datasets from the 5 available:
    #Choose between: train_data_wo_outliers, train_data_pca_wo_out, train_data_pca_one_hot, train_data_pears_wo_out, train_data_pears_one_hot
    #print('\n\n ----------------- DECISION TREE  -----------------\n\n')
    #dt_params = params_4_dec_tree(train_data_pca_one_hot)
    #[pred_pca_one_hot_dec_tree,true_pca_one_hot] = apply_decision_tree(dt_params, train_data_pca_one_hot, test_data_pca_one_hot)
    #print("Accuracy achieved applying decision tree to data_pca_one_hot: ", metrics.accuracy_score(true_pca_one_hot, pred_pca_one_hot_dec_tree))
    #print("\n")
    #dt_params = params_4_dec_tree(train_data_pears_one_hot)
    #[pred_pears_one_hot_dec_tree,true_pears_one_hot_dec_tree] = apply_decision_tree(dt_params, train_data_pears_one_hot, test_data_pears_one_hot)
    #print("Accuracy achieved applying decision tree to data_pears_one_hot: ", metrics.accuracy_score(true_pears_one_hot_dec_tree, pred_pears_one_hot_dec_tree))
    
    print('\n\n ----------------- SVM  -----------------\n\n')
    
    svm_params = params_4_svm(train_data_pears_one_hot)
    [pred_pears_one_hot_svm,true_pears_svm] = apply_svm(svm_params, train_data_pears_one_hot, test_data_pears_one_hot)
    print("Accuracy achieved applying SVM to data_pears_one_hot: ", metrics.accuracy_score(true_pears_svm, pred_pears_one_hot_svm))
    
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
