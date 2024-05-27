store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT ENTROPY (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 8888, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_entropy_8888', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.entropy_sampling', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_entropy_8888', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=entropy_sampling', '--acquisition_method=independent']
store['Distribution of training set classes:']={1: 1402, 0: 221}
store['Distribution of validation set classes:']={1: 192, 0: 40}
store['Distribution of test set classes:']={1: 399, 0: 65}
store['Distribution of pool classes:']={1: 1377, 0: 196}
store['Distribution of active set classes:']={1: 25, 0: 25}
store['active samples']=50
store['available samples']=1573
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[1445, 1377, 879, 983, 707, 1600, 493, 596, 264, 720, 592, 597, 1152, 714, 733, 1507, 1215, 1084, 1044, 109, 1155, 812, 306, 884, 554, 922, 606, 975, 116, 1035, 605, 178, 531, 458, 739, 1571, 1595, 811, 163, 1619, 840, 1599, 448, 219, 416, 564, 533, 502, 1578, 187]
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.875, 'nll': 1.1478931821625808, 'f1': 0.7099874994611837, 'precision': 0.7411359724612736, 'recall': 0.6890688259109312, 'ROC_AUC': 0.8140756225203065, 'PRC_AUC': 0.9757859607884949, 'specificity': 0.4307692307692308}, 'chosen_targets': [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1378, 598, 34, 341, 330, 1538, 608, 1533, 802, 857, 874, 1293, 1264, 95, 1126, 1265, 457, 1182, 1570, 658], 'chosen_samples_score': [0.6931471077581017, 0.6931467915724032, 0.6931342586598077, 0.6930811912043691, 0.693051958400889, 0.6927859202233733, 0.6926676714029827, 0.6924728707284078, 0.6923431884694926, 0.6921377591166451, 0.692006570052955, 0.691187101078426, 0.6911060198396688, 0.6908845024918583, 0.6905958251446155, 0.6904360889834907, 0.690394340654142, 0.690276302781616, 0.6900062947173172, 0.6899303271780417], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 358.48342521861196, 'batch_acquisition_elapsed_time': 54.92823580605909})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6875, 'nll': 0.7921258992162244, 'f1': 0.586871757084523, 'precision': 0.5983393357342938, 'recall': 0.6895122421438211, 'ROC_AUC': 0.7370593281860044, 'PRC_AUC': 0.9574333814426368, 'specificity': 0.6923076923076923}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1], 'chosen_samples': [282, 212, 554, 1336, 1034, 578, 1033, 644, 1269, 1038, 129, 279, 1027, 1247, 875, 374, 850, 669, 1237, 13], 'chosen_samples_score': [0.6931457649199457, 0.6931448103638425, 0.6931401521562447, 0.6931342066324437, 0.6931308689006459, 0.6930834130449122, 0.6930818405046386, 0.6930227257144672, 0.6930198061197852, 0.6930019420313325, 0.6929795406616797, 0.6929731732322961, 0.692934581767606, 0.6929320976122759, 0.6928769154711727, 0.69281507877169, 0.6927307348230378, 0.6926351343433195, 0.6925921128927929, 0.6925676117254649], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 200.67811264283955, 'batch_acquisition_elapsed_time': 54.36165348673239})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.7607758620689655, 'nll': 0.6956824598641231, 'f1': 0.6449738403953926, 'precision': 0.6323609226594301, 'recall': 0.7256795835743204, 'ROC_AUC': 0.7492757824869893, 'PRC_AUC': 0.9616278747820268, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [679, 347, 303, 1120, 842, 376, 1386, 1201, 565, 204, 808, 14, 277, 1348, 1141, 636, 519, 175, 716, 295], 'chosen_samples_score': [0.6931460929263724, 0.6931459671811937, 0.6931455538135585, 0.6931455474176738, 0.6931451862507932, 0.6931397902799525, 0.6931396260734921, 0.693137811435217, 0.6931313949850688, 0.6931184927767821, 0.6931184848057557, 0.6931155680898808, 0.6931123103687107, 0.6930946979473955, 0.6930835864672704, 0.6930834001110032, 0.6930708271442234, 0.6930551838933596, 0.6929812473112522, 0.6929705615621908], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 200.14561056066304, 'batch_acquisition_elapsed_time': 53.7907633241266})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8922413793103449, 'nll': 1.3857456075734105, 'f1': 0.7421649255390088, 'precision': 0.7915672235481305, 'recall': 0.7119722382880278, 'ROC_AUC': 0.855423087537551, 'PRC_AUC': 0.9766270219160733, 'specificity': 0.46153846153846156}, 'chosen_targets': [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1], 'chosen_samples': [962, 1403, 445, 783, 582, 1017, 1191, 28, 108, 1083, 973, 172, 772, 224, 1353, 241, 1360, 898, 1424, 1429], 'chosen_samples_score': [0.6930945233769255, 0.6930293197613558, 0.6926227343228889, 0.6924933786086918, 0.6924425574439997, 0.6912750555684071, 0.6912435567938371, 0.6896364610360066, 0.6888448531804445, 0.6876888653564428, 0.686085246827058, 0.6859928720063, 0.684753949633918, 0.6842270612258083, 0.6837181886766661, 0.6816607216798339, 0.6806564206909174, 0.6794506157429963, 0.6774924539364848, 0.6770354236934606], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 397.2640751218423, 'batch_acquisition_elapsed_time': 53.13983123237267})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 0.7351204444622171, 'f1': 0.7610059081957279, 'precision': 0.7746305418719213, 'recall': 0.7493541546173126, 'ROC_AUC': 0.9031843872301248, 'PRC_AUC': 0.9851618915192595, 'specificity': 0.5538461538461539}, 'chosen_targets': [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], 'chosen_samples': [616, 1373, 968, 890, 198, 974, 70, 832, 1136, 1355, 1137, 181, 321, 48, 58, 1442, 1198, 878, 353, 320], 'chosen_samples_score': [0.6931397141508324, 0.6931127858784736, 0.6930643666596759, 0.6929713851287224, 0.6929331125312266, 0.6929204291408562, 0.6927130279826935, 0.6926685800358618, 0.6926307107307528, 0.6926253339977693, 0.692534905688715, 0.692394039045616, 0.6922162952368356, 0.6919995492873963, 0.6919345793801497, 0.6917903677084678, 0.6914027755736396, 0.6911973729715546, 0.6908873921859238, 0.6908616894018552], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.34377868799493, 'batch_acquisition_elapsed_time': 52.21254242537543})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9137931034482759, 'nll': 0.3635230228818696, 'f1': 0.8337096369565997, 'precision': 0.8130809758716735, 'recall': 0.8597262386736071, 'ROC_AUC': 0.963886456021274, 'PRC_AUC': 0.9937537914506348, 'specificity': 0.7846153846153846}, 'chosen_targets': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [1203, 975, 902, 1191, 521, 80, 832, 1055, 14, 723, 1241, 1274, 547, 374, 801, 1394, 967, 899, 1386, 484], 'chosen_samples_score': [0.6931371053956308, 0.6931074923285283, 0.6930275633893579, 0.6930254176591382, 0.6929236291848524, 0.6928851691176694, 0.6928755138951254, 0.6928666432676464, 0.6928420041012552, 0.6928297127735462, 0.6928184843590595, 0.6926431263486177, 0.6926095114711706, 0.6925330779155934, 0.6925150731004796, 0.6923576387894179, 0.6918114621406785, 0.6914482770529262, 0.6914307336269432, 0.6905854627849459], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.5279042334296, 'batch_acquisition_elapsed_time': 51.58795955963433})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8771551724137931, 'nll': 0.9377416413405846, 'f1': 0.6385255648038051, 'precision': 0.7843480049362401, 'recall': 0.606612685560054, 'ROC_AUC': 0.8668755256734184, 'PRC_AUC': 0.9817833865552945, 'specificity': 0.23076923076923078}, 'chosen_targets': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0], 'chosen_samples': [1103, 1029, 1354, 445, 1206, 1137, 441, 872, 429, 167, 1258, 1277, 843, 505, 1329, 89, 579, 1333, 811, 791], 'chosen_samples_score': [0.6931386865907274, 0.6931247086487681, 0.6930627975943638, 0.6929769368131335, 0.6929584319999227, 0.6927218420752888, 0.6924396654924436, 0.6919099984362538, 0.6916090481903667, 0.6912508980582832, 0.6907387933322346, 0.6896175392617447, 0.6892228882988918, 0.6888819265485812, 0.6884960227970267, 0.688483671239149, 0.688418363122179, 0.6883904487412882, 0.6878979263399351, 0.6867624078727989], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 200.9074280951172, 'batch_acquisition_elapsed_time': 50.936096942052245})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9116379310344828, 'nll': 0.5036155766454237, 'f1': 0.8200784981321229, 'precision': 0.8142453951277481, 'recall': 0.8262772315403895, 'ROC_AUC': 0.8834511394664651, 'PRC_AUC': 0.9900004017812569, 'specificity': 0.7076923076923077}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 'chosen_samples': [174, 1072, 421, 1294, 1356, 833, 1198, 1413, 1086, 310, 1352, 253, 1306, 119, 621, 142, 244, 212, 614, 1236], 'chosen_samples_score': [0.6918052736809166, 0.6909292773959111, 0.6907940392736676, 0.6896621390902529, 0.6896475561098161, 0.6889990464553891, 0.6887932617065673, 0.6884698646361065, 0.6870734731428463, 0.6865798238687768, 0.6835274501682315, 0.6832788988068028, 0.6832473031134054, 0.6831233126870193, 0.6825788658352513, 0.6823566327839015, 0.6820122902951331, 0.6806307638223077, 0.6780383679730975, 0.6775029358115221], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.359571039211, 'batch_acquisition_elapsed_time': 50.08697039913386})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9008620689655172, 'nll': 0.7908735604121767, 'f1': 0.788782013220916, 'precision': 0.7968514827319693, 'recall': 0.7813765182186234, 'ROC_AUC': 0.8906340491050739, 'PRC_AUC': 0.9856304123567661, 'specificity': 0.6153846153846154}, 'chosen_targets': [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [191, 871, 616, 453, 841, 1069, 48, 455, 909, 844, 190, 224, 64, 675, 1165, 740, 425, 585, 66, 552], 'chosen_samples_score': [0.6931435853179057, 0.6931381580084242, 0.6930528495964141, 0.692951953853733, 0.6927480698013901, 0.6926257743401265, 0.6922472551228316, 0.6922447288815154, 0.6920774494045137, 0.6899162714254988, 0.689554796831423, 0.6889293090214768, 0.6886087689437526, 0.6874996366626396, 0.6874743168153514, 0.6868741317609508, 0.684898323604591, 0.6842104453343711, 0.6841496640627954, 0.6814990540216636], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 398.8583502746187, 'batch_acquisition_elapsed_time': 49.668414192739874})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9375, 'nll': 0.4931173982291386, 'f1': 0.8581892145393994, 'precision': 0.8975914861837192, 'recall': 0.8284364758048969, 'ROC_AUC': 0.9121049489835409, 'PRC_AUC': 0.993975654425335, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], 'chosen_samples': [792, 415, 1135, 328, 824, 29, 704, 1203, 859, 466, 1253, 1123, 842, 832, 997, 933, 110, 2, 915, 1091], 'chosen_samples_score': [0.6928169933104453, 0.6922289254737124, 0.6919713203991886, 0.6911421222520261, 0.6907475368621292, 0.6900447083896294, 0.6894994144872368, 0.6894394703678743, 0.6890570795270244, 0.6885378672347533, 0.6868810894405275, 0.6868202363809712, 0.6857314303161015, 0.6831640613286627, 0.6815744655889469, 0.6802847367893617, 0.6756242276935174, 0.675201611608995, 0.6732217574745297, 0.6723057272745878], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.72192145790905, 'batch_acquisition_elapsed_time': 48.824359722901136})
