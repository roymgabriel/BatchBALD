store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT VARIATIONAL RATIOS (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 8888, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_vr_8888', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.variation_ratios', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_vr_8888', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=variation_ratios', '--acquisition_method=independent']
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
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8556034482758621, 'nll': 1.0140346658640895, 'f1': 0.7198446384961295, 'precision': 0.7073385784047749, 'recall': 0.7357432041642568, 'ROC_AUC': 0.8338697542612772, 'PRC_AUC': 0.9783307258426995, 'specificity': 0.5692307692307692}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1455, 1543, 340, 960, 1165, 900, 104, 1535, 954, 1030, 1415, 1124, 191, 329, 381, 1314, 1486, 215, 668, 772], 'chosen_samples_score': [0.4986583076042638, 0.4974184496321634, 0.4959343292488626, 0.4954543104027007, 0.49325163323752597, 0.49323493586737577, 0.49170367027571604, 0.49025935225744344, 0.4834732453307895, 0.48281274498636395, 0.4811678799265703, 0.4802378059628082, 0.479371625807828, 0.4743129292061733, 0.47247011645256765, 0.4723949699209681, 0.46892935975990246, 0.4688204978564925, 0.4678885429325066, 0.467327770379217], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.9872734239325, 'batch_acquisition_elapsed_time': 55.12374006072059})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8448275862068966, 'nll': 1.504620124553812, 'f1': 0.682046596619461, 'precision': 0.6797999924809204, 'recall': 0.6844033159822633, 'ROC_AUC': 0.7905475878775687, 'PRC_AUC': 0.9657668445807507, 'specificity': 0.46153846153846156}, 'chosen_targets': [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [963, 851, 1076, 664, 96, 881, 1424, 26, 1452, 932, 711, 18, 975, 1403, 351, 1162, 685, 654, 1201, 238], 'chosen_samples_score': [0.4986850598292004, 0.49746161203731276, 0.49180677366383496, 0.4909670613136178, 0.48943034152471143, 0.48381087167141945, 0.47474821641228204, 0.471386219857725, 0.46307341588497475, 0.4629800816141608, 0.4621494177094472, 0.4575556512940143, 0.45663080914426646, 0.4540296173085657, 0.45316450644115924, 0.451216526144659, 0.4509377458894088, 0.4488916662545762, 0.44721442590939553, 0.444910849982317], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.8136841878295, 'batch_acquisition_elapsed_time': 54.14679687982425})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.8556034482758621, 'nll': 8.348652280610183, 'f1': 0.536491180987312, 'precision': 0.6487301587301587, 'recall': 0.5361287834972046, 'ROC_AUC': 0.8856115065490064, 'PRC_AUC': 0.9539991681873805, 'specificity': 0.09230769230769231}, 'chosen_targets': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0], 'chosen_samples': [1206, 1084, 1277, 1284, 53, 454, 1319, 1187, 1524, 742, 1448, 363, 809, 61, 359, 718, 448, 785, 501, 296], 'chosen_samples_score': [0.48574581080127777, 0.4398567110388881, 0.4368357127091107, 0.43673703717476275, 0.43620249289581703, 0.4299203643417875, 0.41479476514664404, 0.41280312018512, 0.40536220693028613, 0.39716281057628744, 0.35506181955629057, 0.3302040064878069, 0.3257364413999688, 0.32423145257441754, 0.3108239478993229, 0.3041218780466076, 0.2987188422354802, 0.2560030370825993, 0.24813357773521738, 0.22967924154554853], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 556.424498883076, 'batch_acquisition_elapsed_time': 53.72873981716111})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8620689655172413, 'nll': 1.4137158887139682, 'f1': 0.6349151708876322, 'precision': 0.7018913028193771, 'recall': 0.6107191054559475, 'ROC_AUC': 0.7767156080757613, 'PRC_AUC': 0.956510632209834, 'specificity': 0.26153846153846155}, 'chosen_targets': [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], 'chosen_samples': [688, 266, 536, 263, 1185, 1097, 293, 366, 732, 30, 1150, 38, 277, 1181, 242, 1163, 200, 1371, 638, 393], 'chosen_samples_score': [0.4996700378715162, 0.4993316349511878, 0.4966263081516761, 0.48891542864059945, 0.4874511799911707, 0.4845705138569525, 0.4829765789117627, 0.48216223282232284, 0.4818541931212189, 0.4817748153907907, 0.4816745067632856, 0.47957119767932777, 0.47777772638360194, 0.4769204637094643, 0.47272376220847456, 0.4726017500108941, 0.4696244887540417, 0.4631256563164218, 0.4593544082356661, 0.45638262407657393], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.60661242203787, 'batch_acquisition_elapsed_time': 52.9256577049382})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8706896551724138, 'nll': 1.2972905389193832, 'f1': 0.6577329727071553, 'precision': 0.7345145187372566, 'recall': 0.6286099865047233, 'ROC_AUC': 0.8142238047003373, 'PRC_AUC': 0.9691807847470526, 'specificity': 0.2923076923076923}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [1365, 1007, 1289, 980, 784, 726, 27, 866, 103, 262, 737, 863, 277, 1231, 1304, 667, 232, 117, 124, 1458], 'chosen_samples_score': [0.498367428749088, 0.49273962226900336, 0.49239564166845595, 0.4915919735072143, 0.48900459445957334, 0.4870137784112055, 0.48563142506369195, 0.4852853952651598, 0.48388528678492915, 0.4827293902405827, 0.48100009512700626, 0.4801752091894198, 0.47946365795593004, 0.47930781008902, 0.4790609659953414, 0.4750036524692557, 0.4734050180721363, 0.46344792128803725, 0.4630955175448952, 0.46116701264050397], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.64614167204127, 'batch_acquisition_elapsed_time': 52.21440218668431})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 1.0814612816120017, 'f1': 0.7832512315270936, 'precision': 0.8178322176328159, 'recall': 0.7581260844418739, 'ROC_AUC': 0.8839394718645197, 'PRC_AUC': 0.97964411538027, 'specificity': 0.5538461538461539}, 'chosen_targets': [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0], 'chosen_samples': [305, 1437, 869, 1261, 1239, 430, 629, 1278, 1212, 943, 232, 503, 934, 121, 16, 1118, 1189, 624, 636, 1182], 'chosen_samples_score': [0.499368319467807, 0.49419260460093206, 0.4798951670652093, 0.47549093572217616, 0.4750911389345376, 0.4687705497675273, 0.46811302666573484, 0.4594875561045929, 0.4572286386548522, 0.4562922016544827, 0.4554996611831925, 0.45527533533942044, 0.45417306678363, 0.45397112629826675, 0.43887465906738965, 0.43867745089832366, 0.4310603725996984, 0.42105174864523875, 0.41877708198757324, 0.4119490241757017], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.90135116223246, 'batch_acquisition_elapsed_time': 51.62592206662521})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8814655172413793, 'nll': 1.2570074673356681, 'f1': 0.6944700516000815, 'precision': 0.7703790238836967, 'recall': 0.6606323501060343, 'ROC_AUC': 0.8257423635237333, 'PRC_AUC': 0.9709480539737219, 'specificity': 0.35384615384615387}, 'chosen_targets': [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [1329, 1145, 1383, 954, 2, 1104, 448, 33, 1395, 311, 1445, 964, 1138, 635, 20, 823, 875, 659, 818, 1021], 'chosen_samples_score': [0.4980423737047599, 0.49625770956150894, 0.49558658160804936, 0.4929573385985737, 0.49274656958117513, 0.4917179573445237, 0.4910297779944659, 0.4816776417281532, 0.4796222394196351, 0.4784463009694294, 0.47647869915381613, 0.47467795600140117, 0.4727036435782743, 0.4721933566251051, 0.47206502551814955, 0.4640227118212673, 0.46308959529979166, 0.4560585450399465, 0.45095025436062, 0.4489901626938827], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.74797974713147, 'batch_acquisition_elapsed_time': 51.17157932603732})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9288793103448276, 'nll': 0.6970194454850822, 'f1': 0.8337369021119496, 'precision': 0.8866185897435898, 'recall': 0.7976672450356661, 'ROC_AUC': 0.9139560003197456, 'PRC_AUC': 0.9899576044607384, 'specificity': 0.6153846153846154}, 'chosen_targets': [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0], 'chosen_samples': [36, 307, 1089, 549, 745, 817, 1279, 1393, 1287, 108, 60, 377, 63, 138, 1081, 733, 58, 1322, 848, 1186], 'chosen_samples_score': [0.4905088660196115, 0.4904932860199894, 0.48918840217252335, 0.4864516911917647, 0.4804684077128609, 0.4764380145374463, 0.46608472879720997, 0.4646711550960897, 0.45661020253576745, 0.45310277296644275, 0.4437287955955487, 0.4280844669240631, 0.42704227262653327, 0.41526677864882333, 0.41093442130146385, 0.40970185957937044, 0.4070659897475175, 0.4041629481274355, 0.40047777545680585, 0.3986593295565577], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 445.8058688431047, 'batch_acquisition_elapsed_time': 51.45742873288691})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.698376228069437, 'f1': 0.7725914861837193, 'precision': 0.8574358974358974, 'recall': 0.7284364758048969, 'ROC_AUC': 0.9153296056504869, 'PRC_AUC': 0.9862282722610063, 'specificity': 0.47692307692307695}, 'chosen_targets': [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1], 'chosen_samples': [901, 902, 244, 451, 111, 912, 120, 282, 838, 329, 866, 934, 1312, 874, 1366, 1008, 1072, 79, 1078, 409], 'chosen_samples_score': [0.4968058251813585, 0.4906073677904239, 0.4899632561492654, 0.4865034855879363, 0.4855212014917025, 0.48386963998660026, 0.4837003472239463, 0.4812500986027254, 0.47937188159975164, 0.47883939741686743, 0.47437814844067705, 0.4740495248872275, 0.47300546508310726, 0.4728697702431768, 0.47272045253361306, 0.46792508367184926, 0.46568381572179707, 0.46377484233729926, 0.463236862812086, 0.45690587245578496], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 322.3862708359957, 'batch_acquisition_elapsed_time': 50.81230527209118})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 0.4662194087587554, 'f1': 0.7867474113925761, 'precision': 0.7702794357026587, 'recall': 0.8073067283593599, 'ROC_AUC': 0.8693143023518981, 'PRC_AUC': 0.9883377918699264, 'specificity': 0.6923076923076923}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 'chosen_samples': [569, 1285, 917, 583, 355, 697, 1322, 828, 282, 481, 704, 1287, 72, 973, 418, 676, 822, 442, 19, 655], 'chosen_samples_score': [0.498645469156731, 0.4962724569148014, 0.4958862766921228, 0.4937005163052547, 0.4925089170801362, 0.48967332852398926, 0.4873902415599545, 0.4873043987683713, 0.4840870838122978, 0.4832933038159978, 0.4827653696126246, 0.4811513714924227, 0.4808187582067128, 0.4776101367342409, 0.4773389734603378, 0.47053428016212495, 0.4688476121280091, 0.4685634618529194, 0.46801587547268786, 0.46614963532449627], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.42911080503836, 'batch_acquisition_elapsed_time': 49.17977477423847})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.8943965517241379, 'nll': 0.47086807777141704, 'f1': 0.7822368232015096, 'precision': 0.7804553068372164, 'recall': 0.7840562945826104, 'ROC_AUC': 0.9202863254419767, 'PRC_AUC': 0.992112914192537, 'specificity': 0.6307692307692307}, 'chosen_targets': [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [654, 601, 537, 257, 1064, 745, 1286, 671, 835, 895, 878, 761, 168, 437, 1118, 459, 918, 1068, 948, 1326], 'chosen_samples_score': [0.49809464850187257, 0.49805139445972124, 0.4952380500642021, 0.49413714576629975, 0.4909756500856731, 0.48977839590802863, 0.4866874900032879, 0.484774322108829, 0.4843767355173323, 0.4839984795261363, 0.47879900802356934, 0.47727385829113667, 0.46643250382471124, 0.4654739339298959, 0.4646001363413488, 0.4642922454345859, 0.46246570175444923, 0.46122548626033033, 0.4469977910460776, 0.43505486968619433], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 559.624979857821, 'batch_acquisition_elapsed_time': 48.543447374831885})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9073275862068966, 'nll': 1.010230623442551, 'f1': 0.7652015298617241, 'precision': 0.8536570298986904, 'recall': 0.7207441681125892, 'ROC_AUC': 0.8408955516510593, 'PRC_AUC': 0.9829938317742193, 'specificity': 0.46153846153846156}, 'chosen_targets': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0], 'chosen_samples': [698, 719, 860, 978, 273, 431, 309, 169, 878, 161, 831, 911, 1193, 425, 925, 522, 183, 1055, 1233, 530], 'chosen_samples_score': [0.49925978402131943, 0.48713181772482206, 0.48323547476289563, 0.47834217489822894, 0.47806792850972935, 0.47754074693204407, 0.47666068606987866, 0.469113906930679, 0.4589355761914673, 0.45091711418643343, 0.4484564889966588, 0.44801902642546165, 0.4445630143537518, 0.43837715217273643, 0.43033002659383024, 0.42158842304874367, 0.42116655353730403, 0.4197087205040768, 0.4119365835607508, 0.40688686296645893], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 521.0699110860005, 'batch_acquisition_elapsed_time': 50.78006498422474})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.853448275862069, 'nll': 0.4607364720311658, 'f1': 0.7528822055137845, 'precision': 0.7219428233332437, 'recall': 0.818199344515134, 'ROC_AUC': 0.8787596182375876, 'PRC_AUC': 0.9828586111255839, 'specificity': 0.7692307692307693}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 'chosen_samples': [1139, 863, 1140, 516, 34, 427, 266, 173, 1262, 193, 1244, 75, 671, 281, 793, 594, 487, 1240, 221, 541], 'chosen_samples_score': [0.49937974837361465, 0.49700357809967977, 0.49572410278650814, 0.49075397788163166, 0.48993621218917727, 0.4893966939699125, 0.4889247982719205, 0.48380874104529314, 0.4810931601984191, 0.4796141654246475, 0.4794494941285713, 0.4786776704627238, 0.47670773909912945, 0.47524530721869984, 0.47294955312908393, 0.4729486046858946, 0.4712753434274012, 0.47051695392054116, 0.47011060555250506, 0.46913415787984425], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 203.06553183868527, 'batch_acquisition_elapsed_time': 48.477360770106316})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.3733449146665376, 'f1': 0.7906688221361705, 'precision': 0.783746101632728, 'recall': 0.7981877771351455, 'ROC_AUC': 0.8704113163248699, 'PRC_AUC': 0.9877873598072945, 'specificity': 0.6615384615384615}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [488, 839, 698, 86, 1173, 1086, 438, 868, 1157, 550, 1021, 89, 1218, 1041, 180, 473, 310, 892, 929, 648], 'chosen_samples_score': [0.4990124213048368, 0.4988150025931174, 0.4964777494618028, 0.4952303389801289, 0.4943684778311076, 0.49130814431320324, 0.4909212491146695, 0.4863919243758197, 0.4860738575256296, 0.4852178944994001, 0.4852061825527485, 0.4850927516435325, 0.48443884846211893, 0.48417729821244204, 0.48402771427953706, 0.4837912566426099, 0.48254666446795147, 0.4819299023961454, 0.47936340746886175, 0.4791816368880669], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.3016689219512, 'batch_acquisition_elapsed_time': 46.40825063502416})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.9257646757980873, 'f1': 0.7686939182452641, 'precision': 0.8644217988480284, 'recall': 0.7219973009446694, 'ROC_AUC': 0.9178098328696987, 'PRC_AUC': 0.9904228186615247, 'specificity': 0.46153846153846156}, 'chosen_targets': [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 'chosen_samples': [682, 924, 1261, 843, 989, 647, 205, 1133, 125, 301, 1289, 1007, 135, 486, 462, 119, 563, 263, 931, 124], 'chosen_samples_score': [0.4980764338394644, 0.49685640165718636, 0.49509701154486363, 0.4641720629559495, 0.450236410210186, 0.4496943957789006, 0.42528612396321885, 0.41477338725767643, 0.402652282298662, 0.3836786959082876, 0.3675843059021393, 0.34915539488389546, 0.3449343923440267, 0.3297582843198643, 0.3243318365834116, 0.3129118215086214, 0.31236321259986544, 0.3118002416056622, 0.2918648747763739, 0.29015573827504326], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 520.1554407561198, 'batch_acquisition_elapsed_time': 45.68367902375758})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 0.6174968851023707, 'f1': 0.7731051344743276, 'precision': 0.8284805091487668, 'recall': 0.7388085598611914, 'ROC_AUC': 0.9155049491890968, 'PRC_AUC': 0.9927704994597198, 'specificity': 0.5076923076923077}, 'chosen_targets': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'chosen_samples': [199, 293, 924, 377, 255, 692, 535, 643, 390, 42, 23, 833, 92, 969, 657, 294, 179, 668, 127, 710], 'chosen_samples_score': [0.49857857179387277, 0.49651837585762193, 0.4949131092820588, 0.49146757754955717, 0.48460413513480294, 0.44847843604430837, 0.44140842160422133, 0.43722706448179516, 0.42503036804408867, 0.40589805696839043, 0.403127200591465, 0.3911540818584621, 0.384569337208557, 0.3802189471310343, 0.36642065062937834, 0.35502672539951463, 0.35117658769084126, 0.3508706418356987, 0.3453223317170706, 0.33976034982399805], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 322.18410273082554, 'batch_acquisition_elapsed_time': 44.71901190187782})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.915948275862069, 'nll': 0.5787328522780846, 'f1': 0.8266782878542627, 'precision': 0.8246155017511801, 'recall': 0.8287834972045498, 'ROC_AUC': 0.9118180687182125, 'PRC_AUC': 0.989185221588014, 'specificity': 0.7076923076923077}, 'chosen_targets': [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], 'chosen_samples': [1188, 1193, 1221, 671, 1180, 1192, 965, 99, 942, 998, 772, 118, 1044, 580, 987, 109, 780, 734, 13, 1042], 'chosen_samples_score': [0.4991849911691171, 0.4960579648321729, 0.49059365805529576, 0.4874237902971562, 0.4869405691262778, 0.485202891254389, 0.4827341009760856, 0.47968001433000107, 0.47537849303521396, 0.47059972061711597, 0.4681893124923381, 0.4650471611697521, 0.4625615452205726, 0.4603573395761994, 0.45639429449998015, 0.45040169196446733, 0.4454018790363027, 0.4432644965738609, 0.4413749357141792, 0.4329949039284977], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 440.0753006739542, 'batch_acquisition_elapsed_time': 44.33182989200577})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 0.42954481059107286, 'f1': 0.8324506499759268, 'precision': 0.8469554300062775, 'recall': 0.8196645459803354, 'ROC_AUC': 0.9125182752804879, 'PRC_AUC': 0.9945475308352154, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 'chosen_samples': [491, 1088, 162, 323, 206, 405, 24, 675, 187, 450, 213, 942, 788, 1188, 507, 54, 1067, 744, 692, 562], 'chosen_samples_score': [0.4914086936827792, 0.48738403894524795, 0.48449541125325457, 0.4844879055531389, 0.48324312535880265, 0.4446549872161847, 0.43989147341826884, 0.43930959281412807, 0.4327187332827146, 0.427544366690745, 0.42263840814496945, 0.41423953088658394, 0.3912727108655758, 0.3883702852214832, 0.38058170475649256, 0.3793806510942761, 0.3736455039755119, 0.37315437826816833, 0.3626891073811379, 0.35331297471423684], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.6565525587648, 'batch_acquisition_elapsed_time': 43.54895749129355})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9418103448275862, 'nll': 0.3464458728658742, 'f1': 0.8768468548173052, 'precision': 0.8845690900337024, 'recall': 0.8695777906304223, 'ROC_AUC': 0.9292222701202587, 'PRC_AUC': 0.9952879609767035, 'specificity': 0.7692307692307693}, 'chosen_targets': [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1], 'chosen_samples': [324, 941, 978, 656, 973, 746, 140, 894, 113, 382, 164, 133, 6, 1145, 980, 444, 116, 253, 73, 262], 'chosen_samples_score': [0.4928677791545083, 0.4789355694770222, 0.46441348993338893, 0.4621473810104334, 0.4563605402069948, 0.44858804243630435, 0.43576357127959453, 0.4317657653531569, 0.43038920176199313, 0.4266733746635031, 0.3918360068439183, 0.3870420652322131, 0.3787181452168886, 0.36256993038565444, 0.35677159315281415, 0.3523308954560683, 0.3241160548318205, 0.317437573639599, 0.31213602216461755, 0.308107471735781], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 519.8801693269052, 'batch_acquisition_elapsed_time': 43.05539240408689})
