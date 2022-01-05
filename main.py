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

    print('\n\n ----------------- GENERATE BINARY DATASET -----------------\n\n')

    train_bin_data_one_hot, test_bin_data_one_hot = binary_datasets(train_data_one_hot_enc, test_data_one_hot_enc)
    train_bin_data_wo_out, test_bin_data_wo_out = binary_datasets(train_data_wo_outliers, test_data_wo_outliers)

    print('\n\n ----------------- PERFORMING PCA -----------------\n\n')

    ##Principal component analysis PCA
    #Data types : wo_out -> Data without outliers (train_data_wo_outliers); hot_enc -> Hot encoding applied (train_data_one_hot_enc)
    
    [train_data_pca_wo_out, test_data_pca_wo_out] = compute_pca(train_data_wo_outliers,test_data_wo_outliers,'wo_out')
    [train_bin_pca_wo_out, test_bin_pca_wo_out] = compute_pca(train_bin_data_wo_out,test_bin_data_wo_out,'wo_out')
    
    print('Reduction based on PCA to train_data_wo_outliers applied ---> RESULT: train_data_pca_wo_out')
    print('Reduction based on PCA to test_data_wo_outliers applied ---> RESULT: test_data_pca_wo_out')
    print('Reduction based on PCA to train_bin_data_wo_outliers applied ---> RESULT: train_bin_pca_wo_out')
    print('Reduction based on PCA to test_bin_data_wo_outliers applied ---> RESULT: test_bin_pca_wo_out')

    
    [train_data_pca_one_hot, test_data_pca_one_hot] = compute_pca(train_data_one_hot_enc,test_data_one_hot_enc,'hot_enc')
    [train_bin_pca_one_hot, test_bin_pca_one_hot] = compute_pca(train_bin_data_one_hot,test_bin_data_one_hot,'hot_enc')

    #Save datasets after PCA
    #train_data_pca_wo_out.to_csv("data/train_data_pca_wo_out")
    #train_data_pca_one_hot.to_csv("data/train_data_pca_one_hot")
    #test_data_pca_wo_out.to_csv("data/test_data_pca_wo_out")
    #test_data_pca_one_hot.to_csv("data/test_data_pca_one_hot")
        

    print('Reduction based on PCA to train_data_one_hot_enc applied ---> RESULT: train_data_pca_one_hot')
    print('Reduction based on PCA to test_data_one_hot_enc applied ---> RESULT: test_data_pca_one_hot')
    print('Reduction based on PCA to train_bin_data_one_hot applied ---> RESULT: train_bin_pca_one_hot')
    print('Reduction based on PCA to test_bin_data_one_hot applied ---> RESULT: test_bin_pca_one_hot')
    

    print('\n\n ----------------- PERFORMING REDUCTION BASED ON PEARSON CORRELATION  -----------------\n\n')
    
    ##Pearson correlation analysis
    corr_wo_out = compute_pearson_corr(train_data_wo_outliers, 'wo_out')
    corr_one_hot = compute_pearson_corr(train_data_one_hot_enc, 'hot_enc')
    corr_bin_wo_out = compute_pearson_corr(train_bin_data_wo_out, 'wo_out')
    corr_bin_one_hot = compute_pearson_corr(train_bin_data_one_hot, 'hot_enc')

    [train_data_pears_wo_out, test_data_pears_wo_out] = att_pearson_corr(train_data_wo_outliers,test_data_wo_outliers, corr_wo_out,'wo_out')
    [train_bin_pears_wo_out, test_bin_pears_wo_out] = att_pearson_corr(train_bin_data_wo_out,test_bin_data_wo_out,corr_bin_wo_out,'wo_out')
    [train_data_pears_one_hot, test_data_pears_one_hot] = att_pearson_corr(train_data_one_hot_enc,test_data_one_hot_enc, corr_one_hot,'hot_enc')
    [train_bin_pears_one_hot, test_bin_pears_one_hot] = att_pearson_corr(train_bin_data_one_hot,test_bin_data_one_hot,corr_bin_one_hot,'hot_enc' )

    #Save datasets after pearson reduction
    #train_data_pears_wo_out.to_csv("data/train_data_pears_wo_out")
    #train_data_pears_one_hot.to_csv("data/train_data_pears_one_hot")
    #test_data_pears_wo_out.to_csv("data/test_data_pears_wo_out")
    #test_data_pears_one_hot.to_csv("data/test_data_pears_one_hot")


    print('Reduction based on Pearson coefficient to train_data_wo_out applied ---> RESULT: train_data_pears_wo_out')
    print('Reduction based on Pearson coefficient to test_data_wo_out applied ---> RESULT: test_data_pears_wo_out')
    print('Reduction based on Pearson coefficient to train_bin_data_wo_out applied ---> RESULT: train_bin_pears_wo_out')
    print('Reduction based on Pearson coefficient to test_bin_data_wo_out applied ---> RESULT: test_bin_pears_wo_out')
    
    print('Reduction based on Pearson coefficient to train_data_one_hot_enc applied ---> RESULT: train_data_pears_one_hot')
    print('Reduction based on Pearson coefficient to test_data_one_hot_enc applied ---> RESULT: test_data_pears_one_hot')
    print('Reduction based on Pearson coefficient to train_bin_data_hot_enc applied ---> RESULT: train_bin_pears_one_hot')
    print('Reduction based on Pearson coefficient to test_bin_data_hot_enc applied ---> RESULT: test_bin_pears_one_hot')


    print('-----------------------------------------------------------------------')
    print('\n\n APPLYING TRAINING ALGORITHMS \n\n')
    print('-----------------------------------------------------------------------')

    #We will select 3 datasets from the 5 available:
    #Choose between: train_data_wo_outliers, train_data_pca_wo_out, train_data_pca_one_hot, train_data_pears_wo_out, train_data_pears_one_hot
    

    print('\n\n ----------------- DECISION TREE  -----------------\n\n')
    
    print("\nDataset used: data_pca_one_hot \n")
    ##dt_params_pca_one_hot = params_4_dec_tree(train_data_pca_one_hot)
    dt_params_pca_one_hot = {'max_features': 26, 'max_depth': 14, 'criterion': 'gini'} 
    [pred_pca_one_hot_dec_tree,true_pca_one_hot] = apply_decision_tree(dt_params_pca_one_hot, train_data_pca_one_hot, test_data_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pca_wo_out \n")
    ##dt_params_pca_wo_out = params_4_dec_tree(train_data_pca_wo_out)
    dt_params_pca_wo_out = {'max_features': 16, 'max_depth': 14, 'criterion': 'entropy'}
    [pred_pca_wo_out_dec_tree,true_pca_wo_out] = apply_decision_tree(dt_params_pca_wo_out, train_data_pca_wo_out, test_data_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_one_hot \n")
    ##dt_params_pears_one_hot = params_4_dec_tree(train_data_pears_one_hot)
    dt_params_pears_one_hot = {'max_features': 56, 'max_depth': 14, 'criterion': 'entropy'}
    [pred_pears_one_hot_dec_tree,true_pears_one_hot] = apply_decision_tree(dt_params_pears_one_hot, train_data_pears_one_hot, test_data_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_wo_out \n")
    ##dt_params_pears_wo_out = params_4_dec_tree(train_data_pears_wo_out)
    dt_params_pears_wo_out = {'max_features': 17, 'max_depth': 14, 'criterion': 'gini'}
    [pred_pears_wo_out_dec_tree,true_pears_wo_out] = apply_decision_tree(dt_params_pears_wo_out, train_data_pears_wo_out, test_data_pears_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('--- BINARY CLASSIFICATION ---')
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_pca_one_hot \n")
    ##bin_dt_params_pca_one_hot = params_4_dec_tree_bin(train_bin_pca_one_hot)
    bin_dt_params_pca_one_hot = {'max_features': 37, 'max_depth': 17, 'criterion': 'entropy'}
    [pred_bin_pca_one_hot_dec_tree,true_bin_pca_one_hot] = apply_decision_tree(bin_dt_params_pca_one_hot, train_bin_pca_one_hot, test_bin_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_pca_wo_out \n")
    ##bin_dt_params_pca_wo_out = params_4_dec_tree_bin(train_bin_pca_wo_out)
    bin_dt_params_pca_wo_out = {'max_features': 15, 'max_depth': 14, 'criterion': 'entropy'}
    [pred_bin_pca_wo_out_dec_tree,true_bin_pca_wo_out] = apply_decision_tree(bin_dt_params_pca_wo_out, train_bin_pca_wo_out, test_bin_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_pears_one_hot \n")
    ##bin_dt_params_pears_one_hot = params_4_dec_tree_bin(train_bin_pears_one_hot)
    bin_dt_params_pears_one_hot = {'max_features': 50, 'max_depth': 14, 'criterion': 'entropy'}
    [pred_bin_pears_one_hot_dec_tree,true_bin_pears_one_hot] = apply_decision_tree(bin_dt_params_pears_one_hot, train_bin_pears_one_hot, test_bin_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_pears_wo_out \n")
    ##bin_dt_params_pears_wo_out = params_4_dec_tree_bin(train_bin_pears_wo_out)
    bin_dt_params_pears_wo_out = {'max_features': 17, 'max_depth': 19, 'criterion': 'entropy'}
    [pred_bin_pears_wo_out_dec_tree,true_bin_pears_wo_out] = apply_decision_tree(bin_dt_params_pears_wo_out, train_bin_pears_wo_out, test_bin_pears_wo_out)
    
    
    print('\n\n ----------------- SVM  -----------------\n\n')
    
    print("\nDataset used: data_pca_one_hot \n")
    ##svm_params_pca_one_hot = params_4_svm(train_data_pca_one_hot)
    svm_params_pca_one_hot = {'kernel': 'rbf', 'gamma': 0.001, 'degree': 4, 'C': 250}
    [pred_pca_one_hot_svm,true_pca_one_hot_svm] = apply_svm(svm_params_pca_one_hot, train_data_pca_one_hot, test_data_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pca_wo_out \n")
    ##svm_params_pca_wo_out = params_4_svm(train_data_pca_wo_out)
    svm_params_pca_wo_out = {'kernel': 'rbf', 'gamma': 0.01, 'degree': 4, 'C': 50}
    [pred_pca_wo_out_svm,true_pca_wo_out_svm] = apply_svm(svm_params_pca_wo_out, train_data_pca_wo_out, test_data_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_one_hot \n")
    ##svm_params_pears_one_hot = params_4_svm(train_data_pears_one_hot)
    svm_params_pears_one_hot =  {'kernel': 'rbf', 'gamma': 0.001, 'degree': 3, 'C': 200}
    [pred_pears_one_hot_svm,true_pears_one_hot_svm] = apply_svm(svm_params_pears_one_hot, train_data_pears_one_hot, test_data_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_wo_out \n")
    ##svm_params_pears_wo_out = params_4_svm(train_data_pears_wo_out)
    svm_params_pears_wo_out = {'kernel': 'rbf', 'gamma': 0.001, 'degree': 3, 'C': 250}
    [pred_pears_wo_out_svm,true_pears_wo_out_svm] = apply_svm(svm_params_pears_wo_out, train_data_pears_wo_out, test_data_pears_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('--- BINARY CLASSIFICATION ---')
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pca_one_hot \n")
    ##bin_svm_params_pca_one_hot = params_4_svm_bin(train_bin_pca_one_hot)
    bin_svm_params_pca_one_hot = {'kernel': 'rbf', 'gamma': 'auto', 'degree': 3, 'C': 100}
    [pred_bin_pca_one_hot_svm,true_bin_pca_one_hot_svm] = apply_svm(bin_svm_params_pca_one_hot, train_bin_pca_one_hot, test_bin_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pca_wo_out \n")
    ##bin_svm_params_pca_wo_out = params_4_svm_bin(train_bin_pca_one_hot)
    bin_svm_params_pca_wo_out = {'kernel': 'rbf', 'gamma': 'scale', 'degree': 2, 'C': 50}
    [pred_bin_pca_wo_out_svm,true_bin_pca_wo_out_svm] = apply_svm(bin_svm_params_pca_wo_out, train_bin_pca_wo_out, test_bin_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    bin_svm_params_pears_one_hot = params_4_svm_bin(train_bin_pca_one_hot)
    print("\nDataset used: bin_train_pears_one_hot \n")
    ##bin_svm_params_pears_one_hot = params_4_svm_bin(train_bin_pears_one_hot)
    bin_svm_params_pears_one_hot = {'kernel': 'rbf', 'gamma': 0.001, 'degree': 4, 'C': 100}
    [pred_bin_pears_one_hot_svm,true_bin_pears_one_hot_svm] = apply_svm(bin_svm_params_pears_one_hot, train_bin_pears_one_hot, test_bin_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pears_wo_out \n")
    ##bin_svm_params_pears_wo_out = params_4_svm_bin(train_bin_pca_one_hot)
    bin_svm_params_pears_wo_out = {'kernel': 'rbf', 'gamma': 0.001, 'degree': 4, 'C': 200}
    [pred_bin_pears_wo_out_svm,true_bin_pears_wo_out_svm] = apply_svm(bin_svm_params_pears_wo_out, train_bin_pears_wo_out, test_bin_pears_wo_out)
    

    print('\n\n ----------------- RANDOM FOREST  -----------------\n\n')

    print("\nDataset used: data_pca_one_hot \n")
    ##rf_params_pca_one_hot = params_4_random_forest(train_data_pca_one_hot)
    rf_params_pca_one_hot = {'n_estimators': 180, 'criterion': 'gini'}
    [pred_pca_one_hot_rf,true_pca_rf] = apply_random_forest(rf_params_pca_one_hot, train_data_pca_one_hot, test_data_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pca_wo_out \n")
    ##rf_params_pca_wo_out = params_4_random_forest(train_data_pca_wo_out)
    rf_params_pca_wo_out = {'n_estimators': 189, 'criterion': 'entropy'}
    [pred_pca_wo_rf,true_pca_wo_rf] = apply_random_forest(rf_params_pca_wo_out, train_data_pca_wo_out, test_data_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_one_hot \n")
    ##rf_params_pears_one_hot = params_4_random_forest(train_data_pears_one_hot)
    rf_params_pears_one_hot = {'n_estimators': 163, 'criterion': 'gini'}
    [pred_pears_one_hot_rf,true_pears_rf] = apply_random_forest(rf_params_pears_one_hot, train_data_pears_one_hot, test_data_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_wo_out \n")
    ##rf_params_pears_wo_out = params_4_random_forest(train_data_pears_wo_out)
    rf_params_pears_wo_out = {'n_estimators': 198, 'criterion': 'gini'}
    [pred_pears_wo_out_rf,true_pears_wo_rf] = apply_random_forest(rf_params_pears_wo_out, train_data_pears_wo_out, test_data_pears_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('--- BINARY CLASSIFICATION ---')
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pca_one_hot \n")
    ##bin_rf_params_pca_one_hot = params_4_random_forest_bin(train_bin_pca_one_hot)
    bin_rf_params_pca_one_hot = {'n_estimators': 40, 'criterion': 'entropy'}
    [pred_bin_pca_one_hot_rf,true_bin_pca_one_hot_rf] = apply_random_forest(bin_rf_params_pca_one_hot, train_bin_pca_one_hot, test_bin_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pca_wo_out \n")
    ##bin_rf_params_pca_wo_out = params_4_random_forest_bin(train_bin_pca_one_hot)
    bin_rf_params_pca_wo_out = {'n_estimators': 36, 'criterion': 'entropy'}
    [pred_bin_pca_wo_out_rf,true_bin_pca_wo_out_rf] = apply_random_forest(bin_rf_params_pca_wo_out, train_bin_pca_wo_out, test_bin_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pears_one_hot \n")
    ##bin_rf_params_pears_one_hot = params_4_random_forest_bin(train_bin_pca_one_hot)
    bin_rf_params_pears_one_hot = {'n_estimators': 55, 'criterion': 'entropy'}			
    [pred_bin_pears_one_hot_rf,true_bin_pears_one_hot_rf] = apply_random_forest(bin_rf_params_pears_one_hot, train_bin_pears_one_hot, test_bin_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pears_wo_out \n")
    ##bin_rf_params_pears_wo_out = params_4_random_forest_bin(train_bin_pca_one_hot)
    bin_rf_params_pears_wo_out = {'n_estimators': 68, 'criterion': 'entropy'}
    [pred_bin_pears_wo_out_rf,true_bin_pears_wo_out_rf] = apply_random_forest(bin_rf_params_pears_wo_out, train_bin_pears_wo_out, test_bin_pears_wo_out)


    print('\n\n ----------------- MLPCLASSIFIER (NEURAL NETWORK)  -----------------\n\n')

    print("\nDataset used: data_pca_one_hot \n")
    ##nn_params_pca_one_hot = params_4_neural_network(train_data_pca_one_hot)
    nn_params_pca_one_hot =  {'solver': 'adam', 'momentum': 0.9, 'max_iter': 160, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (215, 172, 129, 86), 'batch_size': 512, 'activation': 'tanh'}
    [pred_pca_one_hot_nn,true_pca_nn] = apply_neural_network(nn_params_pca_one_hot, train_data_pca_one_hot, test_data_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pca_wo_out \n")
    ##nn_params_pca_wo_out = params_4_neural_network(train_data_pca_wo_out)
    nn_params_pca_wo_out = {'solver': 'adam', 'momentum': 0.9, 'max_iter': 140, 'learning_rate': 'invscaling', 'hidden_layer_sizes': (105, 84, 63, 42), 'batch_size': 128, 'activation': 'relu'} 
    [pred_pca_wo_nn,true_pca_wo_nn] = apply_neural_network(nn_params_pca_wo_out, train_data_pca_wo_out, test_data_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_one_hot \n")
    ##nn_params_pears_one_hot = params_4_neural_network(train_data_pears_one_hot)
    nn_params_pears_one_hot =  {'solver': 'adam', 'momentum': 0.8, 'max_iter': 120, 'learning_rate': 'constant', 'hidden_layer_sizes': (177, 118), 'batch_size': 512, 'activation': 'logistic'}
    [pred_pears_one_hot_nn,true_pears_nn] = apply_neural_network(nn_params_pears_one_hot, train_data_pears_one_hot, test_data_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: data_pears_wo_out \n")
    ##nn_params_pears_wo_out = params_4_neural_network(train_data_pears_wo_out)
    nn_params_pears_wo_out =  {'solver': 'adam', 'momentum': 0.9, 'max_iter': 120, 'learning_rate': 'invscaling', 'hidden_layer_sizes': (75, 50), 'batch_size': 512, 'activation': 'logistic'}
    [pred_pears_wo_out_nn,true_pears_wo_nn] = apply_neural_network(nn_params_pears_wo_out, train_data_pears_wo_out, test_data_pears_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('--- BINARY CLASSIFICATION ---')
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\n Dataset used: bin_train_pca_one_hot \n")
    ##bin_nn_params_pca_one_hot = params_4_neural_network_bin(train_bin_pca_one_hot)
    bin_nn_params_pca_one_hot = {'solver': 'adam', 'momentum': 0.9, 'max_iter': 120, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (129, 86), 'batch_size': 128, 'activation': 'relu'}
    [pred_bin_pca_one_hot_nn,true_bin_pca_one_hot_nn] = apply_neural_network(bin_nn_params_pca_one_hot, train_bin_pca_one_hot, test_bin_pca_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pca_wo_out \n")
    ##bin_nn_params_pca_wo_out = params_4_neural_network_bin(train_bin_pca_one_hot)
    bin_nn_params_pca_wo_out = {'solver': 'adam', 'momentum': 0.6, 'max_iter': 120, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (172, 129, 86), 'batch_size': 64, 'activation': 'relu'}
    [pred_bin_pca_wo_out_nn,true_bin_pca_wo_out_nn] = apply_neural_network(bin_nn_params_pca_wo_out, train_bin_pca_wo_out, test_bin_pca_wo_out)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pears_one_hot \n")
    ##bin_nn_params_pears_one_hot = params_4_neural_network_bin(train_bin_pca_one_hot)
    bin_nn_params_pears_one_hot = {'solver': 'adam', 'momentum': 0.8, 'max_iter': 140, 'learning_rate': 'constant', 'hidden_layer_sizes': (172, 129, 86), 'batch_size': 128, 'activation': 'relu'}	
    [pred_bin_pears_one_hot_nn,true_bin_pears_one_hot_nn] = apply_neural_network(bin_nn_params_pears_one_hot, train_bin_pears_one_hot, test_bin_pears_one_hot)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print("\nDataset used: bin_train_pears_wo_out \n")
    ##bin_nn_params_pears_wo_out = params_4_neural_network_bin(train_bin_pca_one_hot)
    bin_nn_params_pears_wo_out = {'solver': 'adam', 'momentum': 0.8, 'max_iter': 120, 'learning_rate': 'constant', 'hidden_layer_sizes': (172, 129, 86), 'batch_size': 256, 'activation': 'relu'}	
    [pred_bin_pears_wo_out_nn,true_bin_pears_wo_out_nn] = apply_neural_network(bin_nn_params_pears_wo_out, train_bin_pears_wo_out, test_bin_pears_wo_out)
    
    
    
    
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
