store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT LEAST CONFIDENCE (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 1234, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_lc_1234', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.least_confidence', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_lc_1234', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=least_confidence', '--acquisition_method=independent']
store['Distribution of training set classes:']={1: 1397, 0: 226}
store['Distribution of validation set classes:']={1: 197, 0: 35}
store['Distribution of test set classes:']={0: 65, 1: 399}
store['Distribution of pool classes:']={1: 1372, 0: 201}
store['Distribution of active set classes:']={1: 25, 0: 25}
store['active samples']=50
store['available samples']=1573
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[915, 280, 362, 68, 1359, 1566, 800, 308, 1272, 1428, 1306, 1355, 156, 1365, 538, 21, 1320, 1388, 243, 894, 947, 1405, 1321, 222, 1301, 1426, 1110, 1093, 377, 1525, 1404, 636, 225, 252, 977, 1098, 562, 1222, 599, 1171, 702, 1474, 1014, 842, 830, 991, 138, 645, 685, 150]
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8620689655172413, 'nll': 4.196699471309267, 'f1': 0.6591837671578753, 'precision': 0.7040881047108343, 'recall': 0.6364758048968575, 'ROC_AUC': 0.8050533335347578, 'PRC_AUC': 0.9695783074177533, 'specificity': 0.3230769230769231}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1], 'chosen_samples': [1080, 204, 1073, 942, 1118, 1502, 1555, 744, 327, 1310, 21, 633, 1410, 470, 1120, 1355, 259, 1070, 359, 1374], 'chosen_samples_score': [-0.5070129188940425, -0.5161034691134676, -0.5163312873101596, -0.5321468231945922, -0.5321889289577821, -0.540071675371873, -0.543641491670172, -0.5442476885505781, -0.5470166995240704, -0.5523419069154926, -0.5720062338125985, -0.5751087662655628, -0.5769518428013476, -0.5796757037876255, -0.5878621405321925, -0.6291280330654586, -0.6361061070076405, -0.6379002385460798, -0.6387094808406432, -0.6479337196913975], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.5846542287618, 'batch_acquisition_elapsed_time': 55.22545436397195})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8081896551724138, 'nll': 0.7429670794256802, 'f1': 0.6622584259554596, 'precision': 0.6456296722254169, 'recall': 0.6952959321380374, 'ROC_AUC': 0.7529295873499187, 'PRC_AUC': 0.95536385802272, 'specificity': 0.5384615384615384}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1356, 166, 1234, 255, 1535, 929, 185, 896, 1158, 70, 908, 412, 287, 1024, 1472, 1395, 791, 691, 278, 926], 'chosen_samples_score': [-0.5008388607879821, -0.5011896862654222, -0.5019423086732683, -0.5021316690773725, -0.5040370816609213, -0.5045375966040249, -0.5045672688851949, -0.5047971367004431, -0.5089039761959168, -0.5093039416069932, -0.509805995904179, -0.5100600467207315, -0.5112395000116228, -0.5112591727631389, -0.5114665131520955, -0.512672680288029, -0.5155829410043647, -0.5159444555071395, -0.5163069691535714, -0.5169009245104754], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.50036291917786, 'batch_acquisition_elapsed_time': 54.47851443802938})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 1.6078864788186962, 'f1': 0.7766106442577032, 'precision': 0.8245318638706056, 'recall': 0.7452477347214189, 'ROC_AUC': 0.870584292061661, 'PRC_AUC': 0.982348090568214, 'specificity': 0.5230769230769231}, 'chosen_targets': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1], 'chosen_samples': [419, 766, 323, 1377, 214, 144, 921, 309, 624, 955, 1455, 986, 685, 1164, 1120, 1073, 782, 1125, 560, 1246], 'chosen_samples_score': [-0.5047412898908793, -0.5056539972796896, -0.5066522247766588, -0.5073351127536351, -0.5214325613156674, -0.5239764436855688, -0.5288539917590075, -0.5315582849124774, -0.5318189762384387, -0.5336021581965082, -0.5342840582789091, -0.5349261087138196, -0.536054290613772, -0.544751011810533, -0.5469787182285688, -0.5500322636031991, -0.5521531532280797, -0.5571526090132062, -0.5578479817728245, -0.5637302057369282], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.5115237380378, 'batch_acquisition_elapsed_time': 53.679106059949845})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.7194919586181641, 'f1': 0.8018142618882969, 'precision': 0.8201646622699255, 'recall': 0.7863890495469443, 'ROC_AUC': 0.9406249263329032, 'PRC_AUC': 0.9893609033611019, 'specificity': 0.6153846153846154}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0], 'chosen_samples': [1148, 1401, 1388, 800, 950, 1417, 397, 617, 1140, 1473, 18, 677, 1123, 429, 422, 1372, 206, 936, 423, 23], 'chosen_samples_score': [-0.5010727862109646, -0.5043831456401223, -0.5120299415792875, -0.5238911756306893, -0.5247437584651162, -0.525334020912402, -0.5256776341264247, -0.5258104314778469, -0.5271709746047394, -0.5300849950202764, -0.5328570598259575, -0.5392330133891776, -0.5423887285485095, -0.5434632764382724, -0.543851118235502, -0.556460131656314, -0.5641837366614031, -0.5665673789727088, -0.5699343796479671, -0.5719183566037288], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 280.7256123921834, 'batch_acquisition_elapsed_time': 53.03560559032485})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8943965517241379, 'nll': 0.4067877407731681, 'f1': 0.810240787881317, 'precision': 0.7784016636957813, 'recall': 0.8613263929053403, 'ROC_AUC': 0.8612301027086218, 'PRC_AUC': 0.986880875925122, 'specificity': 0.8153846153846154}, 'chosen_targets': [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 'chosen_samples': [1375, 787, 1331, 1394, 117, 1222, 1120, 1316, 1055, 127, 1410, 448, 630, 83, 1112, 166, 1158, 703, 1425, 1009], 'chosen_samples_score': [-0.5023931630909533, -0.5044360908310793, -0.5075285017705853, -0.5083337753775141, -0.5106419991109219, -0.5114485933700615, -0.5138143240065282, -0.5164297464581096, -0.5166041674471978, -0.5174136007425876, -0.5188032731106641, -0.5195209777152499, -0.5206190331412318, -0.5288530049343975, -0.5304470869653753, -0.5314448394651431, -0.5355926203289927, -0.5364788310778429, -0.538191461686663, -0.5426437881068263], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.98300487780944, 'batch_acquisition_elapsed_time': 53.76092047803104})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9073275862068966, 'nll': 1.0579368328226024, 'f1': 0.801225404732254, 'precision': 0.811963696369637, 'recall': 0.7915750915750915, 'ROC_AUC': 0.8834279831264392, 'PRC_AUC': 0.9840858828057477, 'specificity': 0.6307692307692307}, 'chosen_targets': [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1], 'chosen_samples': [1409, 1376, 674, 456, 196, 250, 947, 1326, 602, 745, 280, 866, 391, 799, 1209, 1332, 389, 236, 587, 744], 'chosen_samples_score': [-0.5034578257425725, -0.5047287858804244, -0.5067605483354594, -0.5082107905850554, -0.508981940315523, -0.5091358931273382, -0.509327479948163, -0.5095984535751722, -0.5113165539115212, -0.5128173158786173, -0.5137120400623163, -0.5137273318443019, -0.515149847215431, -0.5154972705135668, -0.5215987495446234, -0.5224151504871072, -0.5227804528706679, -0.5241261735851876, -0.5245832526102534, -0.5263166833949896], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.0028780819848, 'batch_acquisition_elapsed_time': 52.213678759057075})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9267241379310345, 'nll': 0.7037249598009833, 'f1': 0.8325123152709359, 'precision': 0.8729050942410863, 'recall': 0.8028532870638134, 'ROC_AUC': 0.9527645801148834, 'PRC_AUC': 0.9907511571888412, 'specificity': 0.6307692307692307}, 'chosen_targets': [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], 'chosen_samples': [1317, 380, 1256, 838, 379, 132, 651, 463, 748, 1319, 852, 652, 888, 1415, 212, 590, 1153, 828, 1247, 1343], 'chosen_samples_score': [-0.5001562950908688, -0.514872732727194, -0.5158720590383162, -0.5162317836499833, -0.5163291358645291, -0.5175580492471243, -0.5248231315825869, -0.5261652941009821, -0.5391358412299743, -0.5440822797645839, -0.5481698599780804, -0.5613023016408635, -0.5787559508389867, -0.5860384291967952, -0.599217643124707, -0.5994706608638455, -0.6015342960093186, -0.6027643618289388, -0.6166193945634841, -0.6262866634300637], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.9993789191358, 'batch_acquisition_elapsed_time': 50.87265223124996})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.36270220526333513, 'f1': 0.8330077120822621, 'precision': 0.8031351854726059, 'recall': 0.8765374975901292, 'ROC_AUC': 0.9198201121696331, 'PRC_AUC': 0.9915526690127783, 'specificity': 0.8307692307692308}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 'chosen_samples': [534, 1404, 426, 1188, 313, 689, 899, 435, 1081, 931, 1191, 896, 1386, 1403, 892, 325, 260, 1420, 50, 653], 'chosen_samples_score': [-0.5009969087612568, -0.5037108614587819, -0.5041505209657419, -0.5080315791763914, -0.5086432544551827, -0.5098783009438536, -0.5234255475125698, -0.5238273795165569, -0.5238632024112294, -0.5249503400812899, -0.5268402021085727, -0.5296481485940763, -0.5326543698932624, -0.532969981646541, -0.5399754832031077, -0.5400440450828771, -0.5403084604685567, -0.5482970152553991, -0.5501581026126499, -0.5534544083893586], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.1684110998176, 'batch_acquisition_elapsed_time': 50.21304801013321})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8771551724137931, 'nll': 0.7603347383696457, 'f1': 0.5963801181192486, 'precision': 0.8558259587020649, 'recall': 0.5744168112589165, 'ROC_AUC': 0.8265225853714542, 'PRC_AUC': 0.9673143083150796, 'specificity': 0.15384615384615385}, 'chosen_targets': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 'chosen_samples': [67, 291, 1198, 714, 746, 1356, 619, 828, 1020, 675, 138, 73, 734, 1018, 267, 1377, 844, 800, 1360, 128], 'chosen_samples_score': [-0.5000039799515782, -0.503809256894666, -0.5117624601185488, -0.5133391741909488, -0.5144348764405169, -0.516107557669832, -0.520122494955955, -0.527801247112647, -0.5323448694233639, -0.5384240667900732, -0.5404133863285149, -0.544974956186849, -0.545736247028491, -0.5573242112343814, -0.5610513306107912, -0.5626179157117573, -0.5626583222892785, -0.565779518418069, -0.5661689263727095, -0.5666876236663295], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.02951286686584, 'batch_acquisition_elapsed_time': 49.47481590975076})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.915948275862069, 'nll': 1.4918322069891568, 'f1': 0.7795079869868771, 'precision': 0.9004787961696306, 'recall': 0.7257566994409099, 'ROC_AUC': 0.8716433409943232, 'PRC_AUC': 0.9875224979421636, 'specificity': 0.46153846153846156}, 'chosen_targets': [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'chosen_samples': [696, 852, 1197, 1173, 499, 1157, 1034, 380, 684, 589, 493, 597, 917, 75, 1156, 1369, 474, 307, 356, 139], 'chosen_samples_score': [-0.5026604269097066, -0.5081754924591471, -0.512252085199439, -0.521576520808522, -0.5303951841390304, -0.5414765303451774, -0.5615663305412476, -0.5647130978167114, -0.6089154321512512, -0.6160717739308085, -0.6302449641876635, -0.6374320925885648, -0.6399120577174566, -0.6463838831988558, -0.6512767102153019, -0.6515426821020199, -0.659512544219893, -0.6616665249082653, -0.6617715019392411, -0.6810215166390768], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.73307843180373, 'batch_acquisition_elapsed_time': 49.01870014797896})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8620689655172413, 'nll': 1.1093646082384834, 'f1': 0.4780285453139281, 'precision': 0.9308855291576674, 'recall': 0.5076923076923077, 'ROC_AUC': 0.5741361445173919, 'PRC_AUC': 0.8567313217830617, 'specificity': 0.015384615384615385}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [397, 847, 218, 817, 523, 905, 684, 711, 80, 619, 1108, 93, 1257, 107, 915, 1243, 919, 334, 1032, 493], 'chosen_samples_score': [-0.5079441078331016, -0.5186708449509734, -0.5202437245549549, -0.526843033691117, -0.5386573730260021, -0.5441637548927974, -0.5466757451026572, -0.5478720645457301, -0.564603153326169, -0.5748022543715007, -0.5795619261354882, -0.5815369218405377, -0.5839837496532697, -0.5841806274183262, -0.6038566901885305, -0.6058717941778685, -0.6140016104429861, -0.6140995071542567, -0.616455696661927, -0.618175346836629], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.2185093718581, 'batch_acquisition_elapsed_time': 48.44166102493182})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.915948275862069, 'nll': 0.3512179604892073, 'f1': 0.8457644020557928, 'precision': 0.8136766334440753, 'recall': 0.8931752458068247, 'ROC_AUC': 0.9135060393439657, 'PRC_AUC': 0.9894244934465343, 'specificity': 0.8615384615384616}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [407, 65, 1092, 627, 979, 1334, 837, 149, 432, 1133, 1187, 1319, 201, 696, 463, 166, 751, 314, 299, 548], 'chosen_samples_score': [-0.5027096229258652, -0.507518397968416, -0.5129194252658387, -0.5131533897831334, -0.5203675806319563, -0.5363249093297543, -0.5389880437917456, -0.5457374283931806, -0.5533684095718384, -0.5545683292644716, -0.5546797970039617, -0.5584926710331239, -0.5590072880419751, -0.5607997861351692, -0.5615766565170757, -0.5621501671613085, -0.5622732872163615, -0.5637473389924967, -0.5665344825710138, -0.5669100373046498], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 439.3008418371901, 'batch_acquisition_elapsed_time': 47.71563162095845})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9073275862068966, 'nll': 0.5656456782900053, 'f1': 0.79567216606757, 'precision': 0.8163515406162465, 'recall': 0.7786967418546366, 'ROC_AUC': 0.8826629181003034, 'PRC_AUC': 0.9861577549758116, 'specificity': 0.6}, 'chosen_targets': [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 'chosen_samples': [276, 934, 379, 1314, 1006, 703, 1274, 1245, 517, 993, 775, 744, 18, 1062, 1090, 363, 1253, 715, 1323, 1185], 'chosen_samples_score': [-0.5018628165703432, -0.5112425170079333, -0.5128572677693319, -0.5134764102659575, -0.5175174808727937, -0.5176872143552411, -0.5214907313469679, -0.5407525277941888, -0.5409903160427125, -0.54758987273202, -0.5507993238309056, -0.5576733043462813, -0.560372754504886, -0.5642436474219342, -0.565703563529777, -0.5662094874727726, -0.5667556085158494, -0.5709759498884703, -0.5719529654692717, -0.5734658174508996], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.3561838809401, 'batch_acquisition_elapsed_time': 48.26163875032216})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9073275862068966, 'nll': 0.2723629556853196, 'f1': 0.8384283365860374, 'precision': 0.7991394927536233, 'recall': 0.9074802390591864, 'ROC_AUC': 0.9154630546323131, 'PRC_AUC': 0.9929779396270119, 'specificity': 0.9076923076923077}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [164, 662, 397, 1289, 345, 159, 786, 1284, 1069, 972, 746, 284, 468, 446, 1218, 105, 73, 823, 141, 790], 'chosen_samples_score': [-0.508994890992042, -0.5103574379241517, -0.5110305955865536, -0.512955206875674, -0.514104012491393, -0.5174985590949346, -0.5179830321980464, -0.5234777488754404, -0.5242607084933317, -0.5250843050514785, -0.5262474653243632, -0.5266320227818347, -0.5279375539514297, -0.529116545352829, -0.5301650776059188, -0.5325139751878936, -0.535540712891883, -0.5371361068191833, -0.5385375390307989, -0.5424640050602553], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 321.20398249384016, 'batch_acquisition_elapsed_time': 46.4973883619532})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7478448275862069, 'nll': 0.5644597020642511, 'f1': 0.6546894380307223, 'precision': 0.648113675850087, 'recall': 0.7761133603238866, 'ROC_AUC': 0.8248671121329347, 'PRC_AUC': 0.9707224740635487, 'specificity': 0.8153846153846154}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1077, 161, 913, 842, 868, 452, 1231, 306, 1084, 813, 221, 56, 753, 1183, 420, 1212, 100, 1291, 210, 633], 'chosen_samples_score': [-0.5000009140588523, -0.500354117701986, -0.5004551409202191, -0.5005203173735444, -0.5006025568320074, -0.5006463973939852, -0.500751981421224, -0.5008177666366289, -0.5008714373909893, -0.5009524566657508, -0.5013874804203978, -0.5016347080647305, -0.5017540109240675, -0.5020529466456557, -0.5021444303055933, -0.5022537296784938, -0.5022704882603244, -0.5022941063540988, -0.5022972852744928, -0.5024243045957035], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.95773602789268, 'batch_acquisition_elapsed_time': 46.01061386382207})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9288793103448276, 'nll': 0.3002696859425512, 'f1': 0.8551851326429283, 'precision': 0.8487076648841355, 'recall': 0.862058993637941, 'ROC_AUC': 0.9177608607412591, 'PRC_AUC': 0.9924943127976624, 'specificity': 0.7692307692307693}, 'chosen_targets': [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], 'chosen_samples': [141, 728, 526, 984, 874, 136, 1017, 569, 293, 980, 511, 1157, 31, 1018, 535, 712, 787, 1044, 1032, 1139], 'chosen_samples_score': [-0.5062991147192546, -0.5149837331838644, -0.5157454946845436, -0.5168237388214725, -0.5206143632503799, -0.5294905268291753, -0.5311375414322045, -0.5444695411378099, -0.545844046532405, -0.5458665670106345, -0.5485174347010265, -0.5495061476378873, -0.5528549364936771, -0.5561521836675725, -0.5583905952836716, -0.5632989686860587, -0.5661685308606208, -0.5744709594410642, -0.5745674216817964, -0.5783735443203396], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.1458091121167, 'batch_acquisition_elapsed_time': 45.45787907065824})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 0.9905718441667228, 'f1': 0.66259356954445, 'precision': 0.8895117090184355, 'recall': 0.6205706574127627, 'ROC_AUC': 0.8869492747161425, 'PRC_AUC': 0.979717297278698, 'specificity': 0.24615384615384617}, 'chosen_targets': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'chosen_samples': [1169, 353, 220, 290, 715, 17, 553, 1227, 369, 57, 811, 236, 201, 75, 303, 141, 364, 928, 588, 108], 'chosen_samples_score': [-0.5064941693397907, -0.5100956802131321, -0.5168397221959968, -0.521875646615986, -0.5241607059300876, -0.5265472939992988, -0.5471423751956392, -0.5507169699501336, -0.559207924941548, -0.5717994215863476, -0.5750441492531205, -0.5793923705205151, -0.5815264363448037, -0.5865574821901741, -0.5960837439596235, -0.6035507817699466, -0.6093419595309957, -0.6146499624670747, -0.6191870248440795, -0.6201849952829478], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.3573432583362, 'batch_acquisition_elapsed_time': 44.48472524480894})
store['iterations'].append({'num_epochs': 15, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.1999512212029819, 'f1': 0.857911871837024, 'precision': 0.8155004354054889, 'recall': 0.9330634278002699, 'ROC_AUC': 0.9343612696358539, 'PRC_AUC': 0.9961081595613074, 'specificity': 0.9538461538461539}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [548, 277, 788, 945, 360, 55, 25, 266, 1197, 811, 138, 1095, 729, 283, 853, 630, 608, 530, 1041, 446], 'chosen_samples_score': [-0.5007279579794921, -0.5070979536158357, -0.507111916297079, -0.50770580865869, -0.5129235034022336, -0.5148119521905964, -0.5175366826714449, -0.5215758347401844, -0.5312537380186456, -0.5378853834124656, -0.5384463605258392, -0.5555236706512959, -0.556816171610038, -0.5571644230374392, -0.5592912032645726, -0.5646804578628037, -0.5741934769447237, -0.574676116648133, -0.5756808466174468, -0.5772925107602312], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 597.7403492182493, 'batch_acquisition_elapsed_time': 44.11219820380211})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.299696396137106, 'f1': 0.8382682076683177, 'precision': 0.8232841677469082, 'recall': 0.8557933294775399, 'ROC_AUC': 0.9053988116358841, 'PRC_AUC': 0.992826937495006, 'specificity': 0.7692307692307693}, 'chosen_targets': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], 'chosen_samples': [612, 285, 804, 1169, 908, 852, 833, 218, 199, 963, 565, 742, 843, 168, 408, 1161, 651, 165, 67, 231], 'chosen_samples_score': [-0.500355819181618, -0.5066851950867522, -0.5075905322654817, -0.5116190390151237, -0.5130632234415224, -0.516927408703846, -0.51852838916312, -0.5195379601841299, -0.5200009581793681, -0.5216471156815767, -0.5219171409084497, -0.5220319688180975, -0.5229122221819824, -0.52529890632572, -0.5256925723871263, -0.5307352202360383, -0.5333437776916742, -0.5425540700329148, -0.5441992432461866, -0.545983980925839], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.3741283789277, 'batch_acquisition_elapsed_time': 43.398947739973664})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9375, 'nll': 0.19881973595454774, 'f1': 0.8727384498983308, 'precision': 0.8659387997623291, 'recall': 0.8799498746867168, 'ROC_AUC': 0.9775914719351294, 'PRC_AUC': 0.9972325567803366, 'specificity': 0.8}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], 'chosen_samples': [1171, 701, 1117, 741, 294, 139, 92, 277, 251, 680, 542, 697, 155, 856, 286, 3, 781, 606, 101, 1114], 'chosen_samples_score': [-0.5041416907340482, -0.5061482968505001, -0.5169940253016903, -0.5239244179235528, -0.5296636273940905, -0.5309571779141222, -0.5318555512614151, -0.5347076343517214, -0.5349409771050202, -0.5439333275663302, -0.5449812260192539, -0.5489205622260362, -0.5526227792481081, -0.5628679848527657, -0.5752855399570104, -0.5872237291584194, -0.5935444807396334, -0.5950323605595615, -0.5961358687111007, -0.6096686495375868], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 519.3981211492792, 'batch_acquisition_elapsed_time': 42.41508554900065})
