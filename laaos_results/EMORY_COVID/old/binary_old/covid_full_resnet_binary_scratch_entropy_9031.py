store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT ENTROPY (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 9031, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_entropy_9031', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.entropy_sampling', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_entropy_9031', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=entropy_sampling', '--acquisition_method=independent']
store['Distribution of training set classes:']={1: 1396, 0: 227}
store['Distribution of validation set classes:']={1: 198, 0: 34}
store['Distribution of test set classes:']={1: 399, 0: 65}
store['Distribution of pool classes:']={1: 1371, 0: 202}
store['Distribution of active set classes:']={0: 25, 1: 25}
store['active samples']=50
store['available samples']=1573
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[228, 320, 1518, 1151, 254, 1373, 1223, 712, 1015, 244, 419, 703, 275, 592, 40, 509, 813, 810, 761, 117, 276, 588, 1203, 6, 413, 1314, 1089, 1569, 745, 1066, 947, 356, 541, 1116, 633, 426, 1411, 611, 1324, 168, 1328, 1275, 213, 302, 806, 330, 366, 851, 1297, 1391]
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 4.474776695514548, 'f1': 0.7443910256410255, 'precision': 0.8743201966773448, 'recall': 0.693734335839599, 'ROC_AUC': 0.8197874833891784, 'PRC_AUC': 0.9701083411644296, 'specificity': 0.4}, 'chosen_targets': [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 'chosen_samples': [986, 1389, 238, 1413, 817, 1377, 643, 1012, 1432, 1019, 1182, 1338, 1082, 469, 1522, 937, 1156, 97, 1501, 624], 'chosen_samples_score': [0.6931457778115421, 0.6928987780770459, 0.6928165437589369, 0.6927930792844355, 0.6927635995039378, 0.69265860273155, 0.6926202642307344, 0.6919325381510819, 0.6919318375158434, 0.6914112391086931, 0.6898745918361597, 0.6875204186772178, 0.6868947193990904, 0.6862757605638401, 0.6862542567694043, 0.6834051717627867, 0.6757060969292824, 0.6737353081100088, 0.668979860861231, 0.6625786447595793], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.13081960985437, 'batch_acquisition_elapsed_time': 55.226990597788244})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8663793103448276, 'nll': 0.7536700676227438, 'f1': 0.7658854166666667, 'precision': 0.7362145200399373, 'recall': 0.8192789666473876, 'ROC_AUC': 0.8604740651203245, 'PRC_AUC': 0.9800129656595654, 'specificity': 0.7538461538461538}, 'chosen_targets': [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1], 'chosen_samples': [607, 1324, 1215, 238, 1434, 223, 930, 547, 309, 1250, 1353, 1111, 1252, 1185, 487, 799, 1019, 104, 1269, 752], 'chosen_samples_score': [0.6931402929807613, 0.6931208046238364, 0.6931199442570781, 0.6930406784365551, 0.6930359427371453, 0.6929756360141606, 0.6929704750144084, 0.6928816883985154, 0.6927301437226547, 0.6926749959500597, 0.6926483316872422, 0.6925737304305054, 0.6925210645251395, 0.6925019700258686, 0.6923461543965104, 0.6923460766913911, 0.6923289469618092, 0.6922098079197673, 0.6921366610594989, 0.692063724202603], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 358.164607134182, 'batch_acquisition_elapsed_time': 54.31052283756435})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8879310344827587, 'nll': 0.46751384077401, 'f1': 0.7886697263777458, 'precision': 0.7666247622731521, 'recall': 0.8189319452477347, 'ROC_AUC': 0.8765053008445761, 'PRC_AUC': 0.9840106515333327, 'specificity': 0.7230769230769231}, 'chosen_targets': [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1], 'chosen_samples': [781, 1033, 1329, 762, 438, 1151, 1027, 1123, 30, 1511, 105, 1319, 329, 1289, 987, 791, 317, 39, 521, 1522], 'chosen_samples_score': [0.693146117659156, 0.6931449521147999, 0.6931403128869937, 0.693137519414887, 0.6931320903142041, 0.6931319467748918, 0.6931121673975341, 0.6931060599446655, 0.6931019718123095, 0.6931010698046075, 0.6930755376818376, 0.6930325843520411, 0.6930096659795906, 0.6929652315094901, 0.6929172748337769, 0.6929019811042216, 0.692898073664056, 0.6928722932052609, 0.6927936057101736, 0.6926452545062565], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 398.96291614789516, 'batch_acquisition_elapsed_time': 53.71046549919993})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9245689655172413, 'nll': 1.4717450635186557, 'f1': 0.8209225137009716, 'precision': 0.8807468275431662, 'recall': 0.7822826296510508, 'ROC_AUC': 0.9029745775362374, 'PRC_AUC': 0.9809538040408046, 'specificity': 0.5846153846153846}, 'chosen_targets': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], 'chosen_samples': [109, 802, 69, 1139, 68, 965, 248, 875, 29, 781, 1322, 1427, 414, 1034, 1311, 881, 538, 826, 1188, 601], 'chosen_samples_score': [0.6931471476834231, 0.6931469219779296, 0.693146659285373, 0.6931465088586541, 0.6931354775740752, 0.6931133572025556, 0.693071606291338, 0.6927491262682048, 0.6926155242711334, 0.6921897904135764, 0.692158578436126, 0.6920453381673872, 0.6920108019954637, 0.6917082958449542, 0.6916183144264423, 0.6909580778484685, 0.6908715594525394, 0.6905927197978805, 0.6901926405199954, 0.6899938246271974], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.6999307619408, 'batch_acquisition_elapsed_time': 53.190348762087524})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8814655172413793, 'nll': 0.558358356870454, 'f1': 0.7824882593094513, 'precision': 0.756582994955088, 'recall': 0.8216117216117216, 'ROC_AUC': 0.8827467159980311, 'PRC_AUC': 0.9839298686552125, 'specificity': 0.7384615384615385}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1], 'chosen_samples': [953, 1444, 1222, 1196, 1355, 590, 249, 117, 228, 608, 1370, 19, 454, 584, 218, 1453, 41, 1367, 332, 831], 'chosen_samples_score': [0.6931429139080527, 0.6931415653180579, 0.6931397032254717, 0.6931384520976183, 0.6931362942126404, 0.6931350312751359, 0.693116639341391, 0.6931128287430587, 0.6929267298059837, 0.6927154870195893, 0.69223007647751, 0.6919054073383148, 0.6906155071114188, 0.6905824259815276, 0.6903134042741684, 0.6901873110019348, 0.6897252477144515, 0.6896807811050258, 0.6890303120376726, 0.6884212539776182], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.5052677206695, 'batch_acquisition_elapsed_time': 52.19788218429312})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.6883554129764952, 'f1': 0.8155262607240008, 'precision': 0.8469448652619015, 'recall': 0.791401580875265, 'ROC_AUC': 0.8949247716372205, 'PRC_AUC': 0.9888843338617801, 'specificity': 0.6153846153846154}, 'chosen_targets': [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], 'chosen_samples': [1210, 367, 266, 983, 1007, 813, 522, 322, 1405, 1169, 329, 435, 426, 33, 376, 115, 24, 20, 340, 473], 'chosen_samples_score': [0.6931430444093918, 0.6929862116062352, 0.6925500760648449, 0.6924629509594844, 0.6919057664917523, 0.6917729089503046, 0.6912408365623497, 0.6911054011514199, 0.6908689632327991, 0.6898593424029904, 0.6897898023761007, 0.6891984965849767, 0.6885164813799598, 0.686651789308946, 0.685855909519764, 0.6855004680569561, 0.6830072398320237, 0.6827565489824903, 0.6820911899009781, 0.6810385011059038], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 517.7644525980577, 'batch_acquisition_elapsed_time': 51.7263809540309})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9267241379310345, 'nll': 0.644412797072838, 'f1': 0.8325123152709359, 'precision': 0.8729050942410863, 'recall': 0.8028532870638134, 'ROC_AUC': 0.9184805785162302, 'PRC_AUC': 0.9920843138616056, 'specificity': 0.6307692307692307}, 'chosen_targets': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [94, 147, 355, 100, 302, 1147, 837, 600, 384, 771, 1116, 543, 1450, 542, 479, 445, 728, 1117, 662, 224], 'chosen_samples_score': [0.6931346546046446, 0.6924436388550159, 0.6923748261023439, 0.6920965513727466, 0.6895224213485057, 0.6893634791036927, 0.68904939678114, 0.6888721503889912, 0.6887908600563252, 0.688549138914238, 0.6878167709669434, 0.687278759104579, 0.6856383453020001, 0.6844093042193466, 0.6843834064296666, 0.6837986935190388, 0.6823705206308966, 0.6802181802981867, 0.6785258347552909, 0.6775587290807323], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.86919058673084, 'batch_acquisition_elapsed_time': 50.93979151081294})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.3631374753754714, 'f1': 0.8205202591625568, 'precision': 0.7829152504283828, 'recall': 0.8883362251783304, 'ROC_AUC': 0.9116261079386445, 'PRC_AUC': 0.9899651802150716, 'specificity': 0.8769230769230769}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 'chosen_samples': [117, 865, 328, 581, 724, 1040, 552, 517, 1350, 1161, 1178, 370, 1280, 579, 377, 1314, 11, 830, 28, 1295], 'chosen_samples_score': [0.6931469429185609, 0.6931152411359045, 0.6930953132020936, 0.6930630244267029, 0.6929417543014273, 0.6928883476960488, 0.6928440146336465, 0.692770301358213, 0.6926989349654135, 0.6926065427200152, 0.6924538803818081, 0.6918942362636292, 0.6917266856095656, 0.6911380311936887, 0.6910144563648495, 0.690566452017334, 0.690055110775391, 0.6891847583523036, 0.6869837029244765, 0.6867714333276624], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 520.3526563402265, 'batch_acquisition_elapsed_time': 50.15116358315572})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.9008620689655172, 'nll': 0.7502832741572939, 'f1': 0.7509335324869306, 'precision': 0.8294419306184012, 'recall': 0.7105455947561211, 'ROC_AUC': 0.8730651293908162, 'PRC_AUC': 0.9744001479573816, 'specificity': 0.4461538461538462}, 'chosen_targets': [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], 'chosen_samples': [1038, 296, 333, 1353, 1296, 538, 1214, 878, 664, 1110, 1041, 170, 1410, 519, 905, 302, 248, 409, 466, 282], 'chosen_samples_score': [0.6931411753713352, 0.6931359803953494, 0.6931193533262795, 0.6930812599857586, 0.6929934632235326, 0.6929494355591709, 0.6928999468508947, 0.6927700464772774, 0.6927232272997048, 0.6924814372791857, 0.6922102052588212, 0.6920371718387057, 0.6919418762223251, 0.6918044005867238, 0.6915543106140356, 0.6910121942348535, 0.6909524619351036, 0.690515195887309, 0.6902644184115087, 0.6893576054069765], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.0009953728877, 'batch_acquisition_elapsed_time': 49.62196237873286})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9439655172413793, 'nll': 0.33661349066372576, 'f1': 0.8806159205161699, 'precision': 0.8912256437375422, 'recall': 0.8708309234625025, 'ROC_AUC': 0.9765143688703011, 'PRC_AUC': 0.9954682011319211, 'specificity': 0.7692307692307693}, 'chosen_targets': [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1], 'chosen_samples': [1054, 1343, 1316, 1392, 1034, 807, 102, 784, 149, 1262, 482, 44, 1068, 129, 42, 775, 1007, 446, 243, 1015], 'chosen_samples_score': [0.6931385593821858, 0.6931378922293918, 0.6931071427743816, 0.69297657022132, 0.6927639697316657, 0.6926278630957637, 0.6921355024525151, 0.6920456571122462, 0.6913321574543096, 0.6911125782658645, 0.6908778540151036, 0.6905680988315488, 0.689226078189693, 0.6888035993273149, 0.6876141726006378, 0.6874541415146018, 0.6872845442050763, 0.6866212275093032, 0.6863535531672387, 0.6862330871148785], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 440.3763549500145, 'batch_acquisition_elapsed_time': 49.23475955473259})
