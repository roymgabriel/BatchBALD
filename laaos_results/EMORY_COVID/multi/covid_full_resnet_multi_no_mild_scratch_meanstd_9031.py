store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT MEAN STD (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 9031, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_meanstd_9031', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.mean_stddev', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_meanstd_9031', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=mean_stddev', '--acquisition_method=independent']
store['Distribution of training set classes:']={2: 812, 1: 586, 0: 225}
store['Distribution of validation set classes:']={2: 113, 1: 83, 0: 36}
store['Distribution of test set classes:']={1: 167, 2: 232, 0: 65}
store['Distribution of pool classes:']={2: 787, 1: 561, 0: 200}
store['Distribution of active set classes:']={0: 25, 2: 25, 1: 25}
store['active samples']=75
store['available samples']=1548
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[228, 1066, 633, 320, 426, 1411, 1324, 168, 1328, 330, 1391, 390, 105, 28, 1389, 254, 1138, 949, 537, 1223, 1435, 449, 712, 321, 1383, 1314, 1089, 1569, 745, 947, 356, 541, 611, 1275, 302, 806, 366, 851, 1297, 980, 1377, 630, 613, 556, 941, 229, 967, 76, 1596, 131, 1116, 213, 976, 1518, 1151, 448, 24, 1373, 1595, 1209, 65, 1285, 182, 1604, 244, 714, 419, 629, 509, 813, 465, 1510, 798, 484, 718]
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.584051724137931, 'nll': 2.2700818818191, 'f1': 0.5428362918969691, 'precision': 0.5871670885339385, 'recall': 0.5298571292428406, 'ROC_AUC': 0.74993896484375, 'PRC_AUC': 0.6112047662281473, 'specificity': 0.7697421162333443}, 'chosen_targets': [1, 1, 2, 1, 0, 1, 2, 0, 1, 1, 2, 2, 1, 2, 2, 2, 0, 0, 1, 1], 'chosen_samples': [255, 640, 324, 1440, 809, 1081, 421, 1157, 66, 180, 490, 726, 383, 1340, 989, 334, 1456, 308, 425, 1137], 'chosen_samples_score': [0.1796923460896616, 0.17427079736494555, 0.17126027191898974, 0.1707131066400019, 0.16719405743267046, 0.16577735564646182, 0.16522716878484917, 0.16046542176849643, 0.15282011569533846, 0.14442933472669223, 0.13577149809836583, 0.1333443867369867, 0.1325159322603957, 0.13219213440615868, 0.13126438885045116, 0.13122679303387588, 0.12959366138068568, 0.1295021601943221, 0.1255759412316832, 0.12491361542360746], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.4943758677691, 'batch_acquisition_elapsed_time': 54.66706607490778})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5560344827586207, 'nll': 1.2771193405677532, 'f1': 0.40856251483253114, 'precision': 0.6966540880503146, 'recall': 0.42952132869539433, 'ROC_AUC': 0.752685546875, 'PRC_AUC': 0.5974943447739814, 'specificity': 0.7417566469290607}, 'chosen_targets': [2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1], 'chosen_samples': [1331, 64, 766, 862, 600, 1236, 716, 1149, 223, 609, 621, 1298, 1235, 902, 1221, 1325, 399, 1013, 739, 248], 'chosen_samples_score': [0.08995682029274783, 0.08967716379246612, 0.08940344184881074, 0.08661705838250844, 0.08539698110388175, 0.08399709515406258, 0.08339410104139314, 0.08219126285346434, 0.08209610330174429, 0.08108146468248785, 0.0804827530403623, 0.08005334604540584, 0.07955196725023896, 0.0780479823088859, 0.07788687500452607, 0.07770099220704335, 0.0774550538017824, 0.07742931615626084, 0.07694990005403249, 0.07644216316629675], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 200.98022895585746, 'batch_acquisition_elapsed_time': 53.91972678294405})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5991379310344828, 'nll': 1.2311572370858028, 'f1': 0.5367123714467882, 'precision': 0.5921575938354461, 'recall': 0.5206777426579202, 'ROC_AUC': 0.77850341796875, 'PRC_AUC': 0.6444294144021064, 'specificity': 0.761015362043795}, 'chosen_targets': [1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 0], 'chosen_samples': [1102, 594, 393, 796, 1026, 1462, 364, 306, 1319, 1463, 24, 1494, 1002, 299, 1139, 864, 1089, 797, 593, 198], 'chosen_samples_score': [0.1614037228524749, 0.09102377948654031, 0.08571686542216433, 0.08420520196174158, 0.08006916482762705, 0.07864004696571542, 0.0779024401364915, 0.0777375358085676, 0.07284877141395514, 0.07281412132651782, 0.0727329931797312, 0.0726517590777383, 0.07251406593352853, 0.07220450893149323, 0.07150509666325051, 0.07100831774567627, 0.07026571433132622, 0.07023990542129274, 0.06998109613430832, 0.06985909270956327], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.44916202314198, 'batch_acquisition_elapsed_time': 53.33644391829148})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.6185344827586207, 'nll': 1.4684360109526535, 'f1': 0.5488624560833085, 'precision': 0.5884488318726707, 'recall': 0.5346143257252074, 'ROC_AUC': 0.7962646484375, 'PRC_AUC': 0.6671253443482642, 'specificity': 0.7798382886858386}, 'chosen_targets': [2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 0, 0, 1, 1, 2, 2, 1, 0, 2], 'chosen_samples': [1068, 311, 854, 1320, 1475, 912, 833, 985, 143, 1389, 1446, 1124, 1476, 1379, 138, 1239, 1487, 195, 1076, 903], 'chosen_samples_score': [0.16916465773554182, 0.16227645147669342, 0.15571005850189867, 0.15416583694985708, 0.15207607695615777, 0.14967469818667534, 0.1485650800430609, 0.14502280771254206, 0.1392810400099314, 0.13869884345252576, 0.13714966501445963, 0.1358441451055586, 0.13575849491827713, 0.13362332139991667, 0.1324444269219289, 0.1293393423640956, 0.12660079805292312, 0.12652863999758668, 0.12591016997381635, 0.12461071383190636], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 478.5186395868659, 'batch_acquisition_elapsed_time': 52.46326674101874})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 1.107708371918777, 'f1': 0.6063577473931697, 'precision': 0.637481717856668, 'recall': 0.5898969964580124, 'ROC_AUC': 0.80841064453125, 'PRC_AUC': 0.6989726294110855, 'specificity': 0.7933386743882811}, 'chosen_targets': [2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 1, 2, 0, 1, 1, 2, 0, 0, 1, 1], 'chosen_samples': [1030, 107, 1443, 311, 67, 24, 160, 960, 1026, 1113, 1169, 732, 803, 577, 756, 1274, 771, 672, 450, 78], 'chosen_samples_score': [0.09683648418833979, 0.09602163989815678, 0.09561877352564081, 0.09083399179666304, 0.09010350923990207, 0.08998949475404812, 0.08663171199365834, 0.08604776372870293, 0.0826554698303148, 0.08138346708618865, 0.08100451869889279, 0.08078311349014697, 0.08016414982035802, 0.07903031452555724, 0.07791087138077082, 0.07707956795489299, 0.076865164081442, 0.0767997639847762, 0.07614476341374526, 0.07559062833509678], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.16607360588387, 'batch_acquisition_elapsed_time': 52.15714174322784})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.584051724137931, 'nll': 0.8256603109425512, 'f1': 0.5235289880207586, 'precision': 0.5338520175476696, 'recall': 0.5619667826151411, 'ROC_AUC': 0.78875732421875, 'PRC_AUC': 0.6525997003893544, 'specificity': 0.7632284953065352}, 'chosen_targets': [2, 2, 1, 2, 2, 1, 2, 0, 2, 1, 1, 2, 2, 2, 0, 0, 2, 2, 0, 1], 'chosen_samples': [442, 51, 1114, 1387, 596, 339, 354, 407, 920, 1047, 504, 283, 577, 523, 798, 18, 374, 1424, 1049, 923], 'chosen_samples_score': [0.07939754911015798, 0.06968930381291644, 0.06921801371184907, 0.06709546220971904, 0.0657563401949323, 0.0649751699865679, 0.0646387679926087, 0.06415099082113362, 0.06407350405378348, 0.0630370133403943, 0.06301461376360973, 0.0619761038177193, 0.06166680201316298, 0.06086910547418087, 0.06084016494736362, 0.060589380945204566, 0.06050021746158039, 0.06037597594787017, 0.060203532965410825, 0.060195897355580716], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.20115944603458, 'batch_acquisition_elapsed_time': 51.204223359934986})
store['iterations'].append({'num_epochs': 18, 'test_metrics': {'accuracy': 0.6379310344827587, 'nll': 1.9691913210112473, 'f1': 0.639436163096503, 'precision': 0.6795821279331404, 'recall': 0.6323325497546022, 'ROC_AUC': 0.8060302734375, 'PRC_AUC': 0.6806163234253938, 'specificity': 0.803543130146276}, 'chosen_targets': [2, 0, 2, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 1, 2, 0, 1], 'chosen_samples': [967, 1301, 1039, 1257, 772, 775, 210, 720, 79, 761, 1350, 1352, 647, 408, 154, 140, 163, 1400, 93, 1361], 'chosen_samples_score': [0.14306044527938144, 0.13450672623076665, 0.1344304943185685, 0.1324329727323095, 0.13114597057706556, 0.12834032384108737, 0.12737742595654533, 0.1236853645857671, 0.12198554554551022, 0.12002239407684602, 0.11928237627155361, 0.1173745186095039, 0.11718288122213694, 0.11695748307418692, 0.11693576349055657, 0.11654216304183573, 0.11548589714755453, 0.11546043652342805, 0.1149643969057972, 0.11478219994617878], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 715.6728710928, 'batch_acquisition_elapsed_time': 50.46272854274139})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6443965517241379, 'nll': 0.8256699792270003, 'f1': 0.6116877291882274, 'precision': 0.618303445824219, 'recall': 0.6412349041969113, 'ROC_AUC': 0.83636474609375, 'PRC_AUC': 0.7244830689269803, 'specificity': 0.7956407039891613}, 'chosen_targets': [1, 1, 0, 2, 2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 2], 'chosen_samples': [1123, 320, 1351, 113, 435, 239, 313, 1356, 141, 251, 324, 1314, 873, 582, 902, 16, 813, 704, 72, 415], 'chosen_samples_score': [0.1625480346612488, 0.1482427307172408, 0.13064978574393912, 0.12437755190199334, 0.12359372056004761, 0.12171009953099737, 0.11787053105727677, 0.11654237179210487, 0.11404289054380826, 0.1100200773050165, 0.10972793396840921, 0.10867377005799102, 0.10624951188455796, 0.10479790794200496, 0.10463256461929722, 0.10226619449803771, 0.09885248857564333, 0.09826715133151731, 0.09807195633535244, 0.09753454186249161], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.61583775561303, 'batch_acquisition_elapsed_time': 49.85648330487311})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6551724137931034, 'nll': 0.8017598513899178, 'f1': 0.616519687327908, 'precision': 0.6657905790737085, 'recall': 0.6135183214472911, 'ROC_AUC': 0.82574462890625, 'PRC_AUC': 0.7141782291168455, 'specificity': 0.7895671635841025}, 'chosen_targets': [1, 1, 0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1], 'chosen_samples': [732, 1162, 1060, 466, 493, 1103, 1087, 1122, 479, 549, 975, 598, 1143, 29, 702, 1321, 664, 1062, 850, 234], 'chosen_samples_score': [0.1143572729963229, 0.10549704785272747, 0.10172284262047045, 0.09958458362823315, 0.09644105255972119, 0.09494039421233014, 0.09373666383470804, 0.09293889480116849, 0.0908930012480499, 0.0886601720045429, 0.08630707596643522, 0.08587194873313049, 0.08480556146316023, 0.08474523507455126, 0.08382525535586918, 0.08310713068904235, 0.08227442905870397, 0.08148463919262762, 0.08119220052615655, 0.08074215991215944], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.6638170229271, 'batch_acquisition_elapsed_time': 48.87034571217373})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6745689655172413, 'nll': 0.6318312020137392, 'f1': 0.6271098490922499, 'precision': 0.6452808768839303, 'recall': 0.6164908908972505, 'ROC_AUC': 0.837646484375, 'PRC_AUC': 0.7333577840743712, 'specificity': 0.8148833421065725}, 'chosen_targets': [0, 1, 1, 0, 2, 2, 1, 2, 1, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 0], 'chosen_samples': [197, 673, 1143, 382, 1067, 987, 1292, 741, 696, 1269, 772, 1029, 861, 1056, 990, 1251, 479, 507, 761, 173], 'chosen_samples_score': [0.12658934826860915, 0.09021847145670903, 0.09018888061844106, 0.08276435052698303, 0.08236136216643636, 0.07910203484960696, 0.07707736471605098, 0.07622449526115957, 0.07604977287747144, 0.07569711828129486, 0.07411244050400148, 0.07398894473622591, 0.07112937080649948, 0.07070459435406468, 0.06967785988245508, 0.06943300604690945, 0.06938982113446136, 0.06927587645469063, 0.06914339089632263, 0.06786891777677526], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.7091029030271, 'batch_acquisition_elapsed_time': 48.25294457515702})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6961206896551724, 'nll': 0.6073053294214709, 'f1': 0.6922261661613027, 'precision': 0.6794974491763828, 'recall': 0.710615506387755, 'ROC_AUC': 0.8687744140625, 'PRC_AUC': 0.8100113551209935, 'specificity': 0.8319159750647955}, 'chosen_targets': [2, 1, 1, 1, 2, 1, 0, 1, 2, 2, 0, 2, 1, 0, 1, 1, 2, 0, 1, 1], 'chosen_samples': [1292, 134, 1298, 496, 207, 458, 709, 66, 853, 485, 1172, 339, 114, 955, 249, 52, 152, 971, 415, 1311], 'chosen_samples_score': [0.07805676056143357, 0.07567715568329147, 0.07493224231375185, 0.07413187249955527, 0.07160195364653492, 0.0703612249219488, 0.0702957362308705, 0.07011950464925748, 0.06970192653026926, 0.06964635433205335, 0.06852698798970919, 0.06382780940742289, 0.06335500119589782, 0.06270347722435494, 0.06250434798945532, 0.062326373590219475, 0.06155789964797749, 0.061037613418313416, 0.06071767990760671, 0.060292941882035725], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.00693141203374, 'batch_acquisition_elapsed_time': 47.75542923109606})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6616379310344828, 'nll': 0.6775755388983364, 'f1': 0.6580110550210218, 'precision': 0.644255410065357, 'recall': 0.6809162841425902, 'ROC_AUC': 0.8355712890625, 'PRC_AUC': 0.7229003363860635, 'specificity': 0.8113604427881499}, 'chosen_targets': [2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1], 'chosen_samples': [44, 337, 149, 510, 875, 182, 709, 558, 1156, 507, 992, 707, 902, 739, 653, 890, 1219, 338, 73, 392], 'chosen_samples_score': [0.1020984611167147, 0.09831832487625045, 0.09717257574899477, 0.09624270609620036, 0.09612422409934798, 0.09421847060135033, 0.09375566846157724, 0.09346002796690474, 0.09277033337256742, 0.09261627233171132, 0.09183103329505729, 0.09130238358870799, 0.09084706789495392, 0.09030702826741066, 0.09025915300430104, 0.0888777167608612, 0.08821965979127988, 0.08814532358202468, 0.08759957649095747, 0.08693414234082653], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.51907581370324, 'batch_acquisition_elapsed_time': 47.34613571688533})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6099137931034483, 'nll': 0.8008977298078865, 'f1': 0.6080888086029473, 'precision': 0.6191165152359183, 'recall': 0.6135441318953605, 'ROC_AUC': 0.8048095703125, 'PRC_AUC': 0.672899966526053, 'specificity': 0.7926551110156675}, 'chosen_targets': [2, 1, 2, 0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 0, 2], 'chosen_samples': [182, 841, 868, 824, 1192, 1251, 267, 123, 142, 1172, 1036, 468, 930, 110, 443, 909, 1282, 692, 511, 244], 'chosen_samples_score': [0.06386123868952572, 0.06071143064347909, 0.05980048203210786, 0.05885117192384498, 0.0584127444629325, 0.05821850477171695, 0.057856026700843444, 0.057731077805733055, 0.057594245172033744, 0.056312435473207315, 0.05619232678832304, 0.0561869048394387, 0.05609283138521036, 0.056010136298613164, 0.05599520728022831, 0.0559077693888998, 0.055665174558895386, 0.05554723568332187, 0.05538028154013752, 0.055356687744612124], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.0754313180223, 'batch_acquisition_elapsed_time': 46.50460576778278})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6616379310344828, 'nll': 0.752487643011685, 'f1': 0.6608610719459321, 'precision': 0.6515199409936252, 'recall': 0.6723020801897531, 'ROC_AUC': 0.8389892578125, 'PRC_AUC': 0.7315350569449625, 'specificity': 0.8120994754388583}, 'chosen_targets': [1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 2], 'chosen_samples': [758, 647, 422, 513, 769, 1029, 983, 1229, 433, 354, 700, 218, 759, 482, 105, 830, 710, 807, 393, 479], 'chosen_samples_score': [0.12075184540351841, 0.11063396467417333, 0.09546062013008755, 0.09352498745456794, 0.09047001916869188, 0.08685603257191024, 0.08179345962395898, 0.08148520305799024, 0.07888292202872767, 0.07680505177429835, 0.07557782836481221, 0.07503130463552787, 0.07376007258168742, 0.07257074352835811, 0.07060618950194149, 0.07042322499660464, 0.06856375330815363, 0.06796767697040881, 0.06706034436928213, 0.0656121531364256], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.3682602341287, 'batch_acquisition_elapsed_time': 45.73832229990512})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 0.5615656622524919, 'f1': 0.5610943883285019, 'precision': 0.6692088127731693, 'recall': 0.5420240421014735, 'ROC_AUC': 0.81378173828125, 'PRC_AUC': 0.6816944863054486, 'specificity': 0.7706593780398983}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 'chosen_samples': [950, 343, 925, 400, 371, 1260, 209, 580, 456, 839, 602, 557, 637, 160, 736, 512, 1181, 1228, 835, 717], 'chosen_samples_score': [0.08913932152797557, 0.08899517090026661, 0.08876154295216478, 0.08134876889818128, 0.08036054594288608, 0.0668095157173136, 0.06559105782062875, 0.06529727787795794, 0.06485170479475794, 0.06457321064896032, 0.06392759723650615, 0.06358134444268959, 0.06244523916773177, 0.059348068688952926, 0.05931604562438254, 0.05926295211287394, 0.058055600189981904, 0.05709938971144307, 0.056419742986271074, 0.055876668827081824], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.12376976571977, 'batch_acquisition_elapsed_time': 45.05611291807145})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6443965517241379, 'nll': 0.796004920170225, 'f1': 0.6550284852824978, 'precision': 0.663372506917978, 'recall': 0.6620365370055645, 'ROC_AUC': 0.830322265625, 'PRC_AUC': 0.6988289845445823, 'specificity': 0.8092054340541939}, 'chosen_targets': [2, 2, 2, 2, 2, 0, 2, 0, 1, 0, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1], 'chosen_samples': [1167, 799, 385, 850, 1062, 236, 578, 959, 321, 293, 394, 782, 924, 119, 717, 532, 102, 351, 234, 1042], 'chosen_samples_score': [0.06941808952636622, 0.0674654952631536, 0.06344549639152701, 0.05482372477355627, 0.05473945392630348, 0.05382642262314198, 0.05376095956699828, 0.05351817284049168, 0.05344750291612855, 0.05325685031923151, 0.05295681076565528, 0.052512183823272385, 0.05239193378306023, 0.051821980573294554, 0.05159318201912825, 0.051000611972678814, 0.04976000245364326, 0.049752906799739795, 0.049651279137311724, 0.049601412698773], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.17610610928386, 'batch_acquisition_elapsed_time': 44.30936286877841})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.6551724137931034, 'nll': 1.1011697177229256, 'f1': 0.6584127860846999, 'precision': 0.6457930367504835, 'recall': 0.6883496931865712, 'ROC_AUC': 0.79852294921875, 'PRC_AUC': 0.6916024618147227, 'specificity': 0.8188338822791333}, 'chosen_targets': [1, 0, 2, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 1, 1, 1, 1], 'chosen_samples': [844, 1141, 4, 135, 470, 358, 96, 1144, 31, 55, 978, 793, 426, 900, 840, 243, 267, 984, 1021, 488], 'chosen_samples_score': [0.07186112670921027, 0.068989311468238, 0.06883471317508064, 0.06863274829353784, 0.06854002222006327, 0.06836182437755967, 0.06796223027761636, 0.06756882636268434, 0.06634348755840276, 0.06628170147535985, 0.06615936702251138, 0.06611293591439289, 0.06585511630010914, 0.06579872115036745, 0.06477851801603954, 0.0644664148172393, 0.06441321953067271, 0.06429882384308433, 0.06363289624731816, 0.06354071045853389], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 439.7304504159838, 'batch_acquisition_elapsed_time': 43.67602203134447})
store['iterations'].append({'num_epochs': 17, 'test_metrics': {'accuracy': 0.7219827586206896, 'nll': 0.8898795555377829, 'f1': 0.7209016414003292, 'precision': 0.723509703332538, 'recall': 0.7215924914097535, 'ROC_AUC': 0.8641357421875, 'PRC_AUC': 0.7796480607130553, 'specificity': 0.8397123730608304}, 'chosen_targets': [0, 1, 2, 2, 1, 2, 2, 2, 1, 0, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1], 'chosen_samples': [185, 309, 358, 1100, 1143, 471, 1142, 182, 494, 79, 113, 918, 931, 261, 75, 820, 598, 1030, 1150, 776], 'chosen_samples_score': [0.08157921438997381, 0.07814962179920204, 0.0779104342648037, 0.07771879208053911, 0.07733749541927895, 0.07710211639750006, 0.07542473758140511, 0.07532022364328406, 0.07495275100690155, 0.07482305940002601, 0.07449463227643462, 0.07424287316050494, 0.07351983027172475, 0.07218227918224497, 0.07168570741296999, 0.07133624798865637, 0.07128482189146712, 0.07123721739477995, 0.07091348038137832, 0.0708523288120936], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 677.8309153728187, 'batch_acquisition_elapsed_time': 42.916782261803746})
