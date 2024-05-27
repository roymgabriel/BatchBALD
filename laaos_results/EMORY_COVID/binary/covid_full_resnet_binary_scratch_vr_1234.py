store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT VARIATIONAL RATIOS (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 1234, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_vr_1234', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.variation_ratios', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_vr_1234', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=variation_ratios', '--acquisition_method=independent']
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
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8491379310344828, 'nll': 0.9614184807086813, 'f1': 0.7407738475290512, 'precision': 0.712716203127162, 'recall': 0.796375554270291, 'ROC_AUC': 0.8609393706112144, 'PRC_AUC': 0.9778801435679292, 'specificity': 0.7230769230769231}, 'chosen_targets': [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0], 'chosen_samples': [905, 127, 820, 1337, 1409, 4, 70, 764, 1158, 847, 234, 1286, 890, 1322, 897, 505, 183, 1421, 1422, 573], 'chosen_samples_score': [0.49856183804533805, 0.4965796653623433, 0.49583468855987045, 0.49392006960312806, 0.48772890470816277, 0.4830617009182, 0.4817946058525776, 0.47743401466010804, 0.47726638575292624, 0.47519282328159973, 0.472524919046574, 0.47052184083348714, 0.46862646919798445, 0.4683373152314879, 0.467099698499699, 0.46302350237961143, 0.46164839343494724, 0.4587303324637573, 0.4565966784349116, 0.4557568371597689], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.0758252609521, 'batch_acquisition_elapsed_time': 55.29483807692304})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8663793103448276, 'nll': 0.984945165699926, 'f1': 0.7296138952592204, 'precision': 0.7241607044578977, 'recall': 0.7355696934644302, 'ROC_AUC': 0.7783105036601077, 'PRC_AUC': 0.9665322042883461, 'specificity': 0.5538461538461539}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], 'chosen_samples': [143, 523, 1485, 1477, 777, 979, 1366, 858, 1222, 1035, 429, 1158, 279, 743, 328, 818, 44, 971, 1171, 638], 'chosen_samples_score': [0.4999604958525937, 0.4996975934819743, 0.49794761773178364, 0.4974479264307865, 0.49633066144305515, 0.4951223560720521, 0.49504946776950065, 0.4925008929712815, 0.49235627057155895, 0.49032852727206144, 0.4890995687119287, 0.48890291507967787, 0.48780670145407545, 0.4854738341179421, 0.4839263551483546, 0.4838234370797214, 0.48358739880137414, 0.4835303056965442, 0.48256768323888266, 0.48111925274786926], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.4727907590568, 'batch_acquisition_elapsed_time': 57.92852030089125})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8943965517241379, 'nll': 1.0858602194950497, 'f1': 0.7901944318842452, 'precision': 0.7787698412698413, 'recall': 0.8033738191632929, 'ROC_AUC': 0.8874139131690689, 'PRC_AUC': 0.98348645630569, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1], 'chosen_samples': [1221, 1014, 66, 321, 1386, 1093, 927, 1164, 1523, 670, 1469, 396, 194, 1475, 1203, 259, 955, 326, 1494, 21], 'chosen_samples_score': [0.4969601401689554, 0.4952034180214373, 0.4939240936910747, 0.4852903249041701, 0.48506829870680146, 0.4816211413028715, 0.48102585869438985, 0.48063997798206415, 0.47641522744876996, 0.4740807552631966, 0.4737656953355093, 0.4730256851729643, 0.47288103019411654, 0.47247157679235163, 0.4713963342324057, 0.47055242721081914, 0.469628251386387, 0.46850046669212975, 0.46661648085728835, 0.46640129316061885], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 403.0818691761233, 'batch_acquisition_elapsed_time': 56.85589390108362})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.875, 'nll': 2.8529394741716056, 'f1': 0.6500936134803412, 'precision': 0.7598949063479956, 'recall': 0.6182379024484288, 'ROC_AUC': 0.8402969723226053, 'PRC_AUC': 0.9699717033378397, 'specificity': 0.26153846153846155}, 'chosen_targets': [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], 'chosen_samples': [1387, 864, 1406, 12, 546, 351, 1194, 1127, 305, 894, 847, 747, 761, 1389, 882, 595, 61, 121, 1359, 398], 'chosen_samples_score': [0.4950361560097363, 0.4942029686044924, 0.49412306328595157, 0.49387384784263466, 0.48716394683589836, 0.48363350318769815, 0.47632574575773523, 0.47188986571954783, 0.4685655674458562, 0.46782626422751616, 0.46230322204544694, 0.4516231394816852, 0.4316503497719103, 0.42483732372285354, 0.40466527176871814, 0.3989302372800607, 0.395803959392669, 0.3906849858611845, 0.3860740803278524, 0.38263813427785043], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 243.80410549324006, 'batch_acquisition_elapsed_time': 56.72610912518576})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 1.8835659684806034, 'f1': 0.6952675294572146, 'precision': 0.8245412844036697, 'recall': 0.6527665317139001, 'ROC_AUC': 0.8596592575089631, 'PRC_AUC': 0.9794493669957315, 'specificity': 0.3230769230769231}, 'chosen_targets': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], 'chosen_samples': [178, 1285, 336, 1028, 1070, 1247, 752, 232, 1135, 1230, 109, 1000, 183, 736, 339, 767, 276, 452, 858, 1467], 'chosen_samples_score': [0.49401943260294634, 0.4876700459103014, 0.47432522079744754, 0.46809453426732794, 0.4644203510328476, 0.44715924395927964, 0.4445623013494354, 0.4318642263973693, 0.4299223891475884, 0.4203001665281417, 0.41914351428210994, 0.41657810328763667, 0.41269144354623466, 0.406855156563401, 0.39813341835455096, 0.39070462417577057, 0.3670477225850163, 0.3422337834359689, 0.34203672580465994, 0.33983137157829746], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.5489550521597, 'batch_acquisition_elapsed_time': 52.94514717813581})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8685344827586207, 'nll': 4.328509363634833, 'f1': 0.604996162165934, 'precision': 0.7403153153153152, 'recall': 0.5822826296510507, 'ROC_AUC': 0.7751822902012604, 'PRC_AUC': 0.9552611204619753, 'specificity': 0.18461538461538463}, 'chosen_targets': [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0], 'chosen_samples': [903, 957, 1358, 313, 1445, 614, 129, 1080, 412, 722, 923, 298, 709, 658, 1420, 1438, 186, 515, 252, 142], 'chosen_samples_score': [0.49377179285610673, 0.4906572644083902, 0.4902343140382549, 0.4824203201110987, 0.4793000495017067, 0.45344896720071537, 0.434327619092701, 0.4333994978638922, 0.4306690225705625, 0.4242933120007314, 0.4129433240064606, 0.4124131986708903, 0.4088143064128087, 0.40684181806398767, 0.3959359101161277, 0.3860210013276699, 0.3811048659134977, 0.36891620736035224, 0.3677317814299219, 0.35344304565676166], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.9917370369658, 'batch_acquisition_elapsed_time': 51.89193049212918})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.806634574100889, 'f1': 0.7635467980295566, 'precision': 0.7958030669895076, 'recall': 0.7402352033930981, 'ROC_AUC': 0.8937736696569686, 'PRC_AUC': 0.9873643738727529, 'specificity': 0.5230769230769231}, 'chosen_targets': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1], 'chosen_samples': [1286, 85, 895, 1274, 1255, 1321, 1348, 272, 967, 346, 929, 293, 327, 1382, 1386, 946, 1356, 17, 728, 239], 'chosen_samples_score': [0.49929088679403155, 0.49815369055065317, 0.4969419544205864, 0.4957209837911195, 0.4929789024779415, 0.49167878174740265, 0.4788732110918219, 0.4728273287792917, 0.46974155586339483, 0.46972993119510886, 0.4678135256647923, 0.46770646035746877, 0.45792825624521916, 0.4552515988985314, 0.4532231925843778, 0.44812715360405164, 0.4394197642305989, 0.43899571149594796, 0.4365453706742175, 0.4359759256414497], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 319.9508672906086, 'batch_acquisition_elapsed_time': 51.17173204990104})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9353448275862069, 'nll': 0.5331016080132847, 'f1': 0.8603755416466057, 'precision': 0.8760828625235405, 'recall': 0.8465008675534991, 'ROC_AUC': 0.9358486647221509, 'PRC_AUC': 0.9902624565020972, 'specificity': 0.7230769230769231}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], 'chosen_samples': [327, 655, 582, 632, 833, 756, 505, 1030, 1150, 640, 1345, 651, 1167, 570, 365, 639, 333, 888, 273, 876], 'chosen_samples_score': [0.4954509876931663, 0.49439180713335074, 0.49073393998366843, 0.4896952813408014, 0.4842209150224761, 0.4841301862176458, 0.48396353070772, 0.4726036725612839, 0.46516873666424574, 0.45813753812352576, 0.4580614495237004, 0.4522923297811856, 0.4518918468899622, 0.45059789267695294, 0.44515564357303494, 0.4421997378815231, 0.4403446714313387, 0.4398864179105555, 0.4387757099249363, 0.437422370889598], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.53975280793384, 'batch_acquisition_elapsed_time': 50.35194666311145})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.7527036338016905, 'f1': 0.800903342366757, 'precision': 0.8713472905043362, 'recall': 0.7592057065741276, 'ROC_AUC': 0.9293461489343906, 'PRC_AUC': 0.9875170354533931, 'specificity': 0.5384615384615384}, 'chosen_targets': [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 'chosen_samples': [70, 31, 221, 554, 21, 351, 172, 321, 1377, 1392, 257, 540, 326, 605, 962, 989, 1391, 827, 635, 1386], 'chosen_samples_score': [0.4998221560889733, 0.4995202137690865, 0.49722115137888756, 0.49601022844903453, 0.4937859180480746, 0.4882808480501162, 0.48303577906552, 0.48112844280718103, 0.47782534587123715, 0.4723120721574029, 0.47203780474183576, 0.4715317258481264, 0.46564595128951025, 0.4655287433562222, 0.4635799740294869, 0.4631147002684466, 0.46121347868676577, 0.4496458750748652, 0.444014478328013, 0.4418964907864201], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.07000137958676, 'batch_acquisition_elapsed_time': 49.70897252764553})
store['iterations'].append({'num_epochs': 16, 'test_metrics': {'accuracy': 0.9288793103448276, 'nll': 0.5765991868643925, 'f1': 0.8474520547945205, 'precision': 0.8598184818481849, 'recall': 0.8363022941970311, 'ROC_AUC': 0.8959919000244263, 'PRC_AUC': 0.9881847993973054, 'specificity': 0.7076923076923077}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [970, 1017, 1049, 1119, 974, 85, 1288, 109, 1046, 16, 992, 1296, 438, 621, 1259, 651, 220, 976, 546, 1167], 'chosen_samples_score': [0.4930875209738582, 0.4916094574204901, 0.4629543214452432, 0.45579933678133977, 0.4535951337994508, 0.4388120184048031, 0.4381726385365866, 0.4186134336476173, 0.412737939118338, 0.4122247703891242, 0.4076422558791545, 0.40688027426548345, 0.400310244661515, 0.39870164661429364, 0.3897054021644659, 0.38341202139091035, 0.3736575332550619, 0.36999484365116964, 0.36841872476401993, 0.3665843876432159], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 636.746442545671, 'batch_acquisition_elapsed_time': 48.96464736200869})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9288793103448276, 'nll': 0.5956557701373922, 'f1': 0.8431902669820885, 'precision': 0.8671218487394958, 'recall': 0.8234239444765761, 'ROC_AUC': 0.9266314031719114, 'PRC_AUC': 0.9909289801680774, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0], 'chosen_samples': [1021, 1044, 1221, 366, 181, 1361, 316, 1342, 1110, 192, 583, 140, 837, 285, 850, 160, 166, 1345, 1118, 416], 'chosen_samples_score': [0.4959862181338802, 0.4957951778270453, 0.49085503001594744, 0.48468525327573664, 0.4826077829218728, 0.48035728242264275, 0.47451226827521176, 0.47395941746954995, 0.470139259532945, 0.46622840356312734, 0.46554955718824065, 0.4645037315082253, 0.4621314987975119, 0.45747842375969083, 0.4549302874165593, 0.452264825081939, 0.43919414622938924, 0.43490105015187974, 0.43203228069909805, 0.43026901471957224], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.07902988418937, 'batch_acquisition_elapsed_time': 48.34282526280731})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9202586206896551, 'nll': 0.5240092770806675, 'f1': 0.8334352048587867, 'precision': 0.8356250000000001, 'recall': 0.8312897628687103, 'ROC_AUC': 0.9305937164818737, 'PRC_AUC': 0.9911831662431076, 'specificity': 0.7076923076923077}, 'chosen_targets': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1020, 395, 431, 724, 1060, 928, 214, 1120, 1225, 603, 390, 34, 377, 106, 1328, 605, 8, 281, 289, 308], 'chosen_samples_score': [0.497371159645425, 0.4960360596874125, 0.49478120649842605, 0.4929162651540636, 0.48864178613045306, 0.48861694192609173, 0.48809663643059265, 0.48793627876456847, 0.4863246489682099, 0.48452603929558113, 0.4719266341280536, 0.470916662787498, 0.4607718799562809, 0.45763831256128507, 0.45679307679077397, 0.45508726774386843, 0.4530080312693082, 0.45269013806544833, 0.4331071116809273, 0.43239524620121994], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 398.8079708190635, 'batch_acquisition_elapsed_time': 47.70723904389888})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8987068965517241, 'nll': 0.4113956977581156, 'f1': 0.7987579244603985, 'precision': 0.7869897959183674, 'recall': 0.8123192596876807, 'ROC_AUC': 0.8765796580216796, 'PRC_AUC': 0.9865544489758706, 'specificity': 0.6923076923076923}, 'chosen_targets': [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1261, 318, 664, 923, 269, 1039, 1103, 1122, 18, 958, 939, 191, 972, 1037, 676, 144, 73, 1060, 1083, 1169], 'chosen_samples_score': [0.4998255821593258, 0.4973536847002421, 0.4966724292703819, 0.4962193962662933, 0.4958029075097087, 0.4953982229969661, 0.4925624340578757, 0.4910436533103477, 0.48930765733454584, 0.48640635627528894, 0.48406498583354907, 0.48390317390409243, 0.48359576105163915, 0.4800011020888838, 0.47989230088135004, 0.4766347523824992, 0.47534578442541386, 0.4740307094788325, 0.4730646749921681, 0.4681754575967342], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.2979510417208, 'batch_acquisition_elapsed_time': 47.991312825120986})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9116379310344828, 'nll': 0.6363962107691271, 'f1': 0.7868960804740621, 'precision': 0.8494588744588745, 'recall': 0.7490071332176595, 'ROC_AUC': 0.9094948347868084, 'PRC_AUC': 0.9857109067713574, 'specificity': 0.5230769230769231}, 'chosen_targets': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0], 'chosen_samples': [1126, 105, 1015, 456, 705, 418, 862, 490, 942, 1043, 965, 385, 160, 604, 960, 997, 67, 1008, 175, 717], 'chosen_samples_score': [0.49988103335028466, 0.4970216025971481, 0.4935025786316475, 0.4887515037861596, 0.4877113255718242, 0.47991433868991196, 0.47991084827568775, 0.47975241184585327, 0.47941344369365857, 0.47906370116092933, 0.47847931174783276, 0.47413585325972585, 0.4732613422115671, 0.47281209409489755, 0.46810975338504857, 0.46690431564749035, 0.4606443230476789, 0.4579412153133732, 0.4566024814897538, 0.45493361432490276], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 321.8089647158049, 'batch_acquisition_elapsed_time': 46.365416556131095})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9331896551724138, 'nll': 0.37476207470071726, 'f1': 0.8688440882275168, 'precision': 0.8507623007623007, 'recall': 0.8903219587430113, 'ROC_AUC': 0.9087520095320771, 'PRC_AUC': 0.9941155408883711, 'specificity': 0.8307692307692308}, 'chosen_targets': [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [529, 1044, 26, 1025, 931, 970, 659, 738, 1049, 1175, 1078, 1205, 1156, 1212, 643, 763, 654, 1009, 1122, 1224], 'chosen_samples_score': [0.49973720336882055, 0.4969094089367463, 0.4892925971042704, 0.48429934765559446, 0.47403661095292937, 0.47081709095108004, 0.47068960931936954, 0.46929714642780773, 0.4687125282386113, 0.46475517510566744, 0.4612590765552014, 0.45937189127139655, 0.4519184593318776, 0.4404965860076253, 0.43508662959845856, 0.4296798718418072, 0.42897774160442526, 0.41900434271686593, 0.4165169112772903, 0.4075533105228941], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 519.2450237451121, 'batch_acquisition_elapsed_time': 45.486773781012744})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9202586206896551, 'nll': 1.1030198458967537, 'f1': 0.7908152697054989, 'precision': 0.9163474692202462, 'recall': 0.7347021399652979, 'ROC_AUC': 0.9114345459077665, 'PRC_AUC': 0.9905554533915789, 'specificity': 0.47692307692307695}, 'chosen_targets': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0], 'chosen_samples': [25, 1112, 142, 832, 57, 418, 663, 333, 1262, 488, 583, 520, 534, 226, 659, 242, 354, 153, 441, 897], 'chosen_samples_score': [0.4924504265382472, 0.4831041972407051, 0.48155330611369573, 0.4679106298598763, 0.4646534348784187, 0.44791569052891766, 0.4396385478482333, 0.42189842838594793, 0.4163193577026786, 0.41554389021938176, 0.40536401323789806, 0.40345494024131257, 0.38828918515663624, 0.36317184096555866, 0.3466420113926769, 0.34471158200281515, 0.34208014211100635, 0.28523988498824804, 0.27434699806976803, 0.25386652620675454], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.55146698933095, 'batch_acquisition_elapsed_time': 44.94925590278581})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6724137931034483, 'nll': 0.5882161239097858, 'f1': 0.5919935207682517, 'precision': 0.6151654780796513, 'recall': 0.7322537112010796, 'ROC_AUC': 0.7794704002825399, 'PRC_AUC': 0.9574495680191598, 'specificity': 0.8153846153846154}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], 'chosen_samples': [75, 527, 367, 285, 43, 328, 687, 529, 492, 1061, 708, 186, 1086, 351, 791, 78, 324, 374, 1109, 221], 'chosen_samples_score': [0.4986743891775718, 0.4958935780973347, 0.4908331110053116, 0.48977808894257935, 0.4888464458234002, 0.48823519319939024, 0.4880132439558236, 0.4868466266916529, 0.4864976630338823, 0.48534237963180826, 0.48116388078580885, 0.4811598384993647, 0.4802405355932845, 0.47760667746190244, 0.47690082189060823, 0.4765050968138276, 0.47639641008999534, 0.4762179132291343, 0.47435586784906003, 0.4742854878243675], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.30480602011085, 'batch_acquisition_elapsed_time': 44.36417329963297})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9030172413793104, 'nll': 0.5519464427027209, 'f1': 0.7919800747198007, 'precision': 0.8023927392739274, 'recall': 0.7826296510507037, 'ROC_AUC': 0.8582813929879375, 'PRC_AUC': 0.9807407853435185, 'specificity': 0.6153846153846154}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], 'chosen_samples': [230, 415, 1076, 1014, 1171, 1083, 729, 831, 804, 1144, 927, 339, 685, 798, 684, 1098, 238, 312, 1126, 1205], 'chosen_samples_score': [0.494519296954549, 0.48483640611097323, 0.48327276778438666, 0.4832542349430158, 0.4744162745637941, 0.47425253404339807, 0.4734496413434074, 0.47153450108871053, 0.4641130154696368, 0.46383179715003564, 0.4630010966260927, 0.4628606431461312, 0.46153629143357566, 0.45665987227733307, 0.45384671527940434, 0.4525554159956774, 0.4508897502143874, 0.4505105816129171, 0.44259159241782897, 0.4384316332901308], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 477.81308751925826, 'batch_acquisition_elapsed_time': 43.49983600107953})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8642241379310345, 'nll': 0.4412386992882038, 'f1': 0.7609245037662242, 'precision': 0.7323461759631973, 'recall': 0.81158665895508, 'ROC_AUC': 0.8436875861008588, 'PRC_AUC': 0.9749352037369272, 'specificity': 0.7384615384615385}, 'chosen_targets': [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [706, 6, 953, 813, 660, 554, 925, 322, 1045, 167, 177, 381, 413, 321, 1008, 74, 375, 977, 1133, 133], 'chosen_samples_score': [0.4993452961502628, 0.49898023629576327, 0.498848848594849, 0.4987350234722724, 0.4965743124275236, 0.4960564888347222, 0.4953752326538615, 0.49448558006670706, 0.49367667024296524, 0.4908290559363503, 0.49037065943420655, 0.486823706680511, 0.486477217876223, 0.481634446100774, 0.48062646699139777, 0.48041755765721517, 0.48014941555467217, 0.4790925321689292, 0.47539132357802105, 0.47456414961588644], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.5638350979425, 'batch_acquisition_elapsed_time': 42.89565011905506})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.3406026116732893, 'f1': 0.8311850311850313, 'precision': 0.803529076937672, 'recall': 0.8700983227299017, 'ROC_AUC': 0.918123299944932, 'PRC_AUC': 0.9902579617918535, 'specificity': 0.8153846153846154}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], 'chosen_samples': [325, 796, 997, 694, 771, 1077, 328, 443, 435, 327, 190, 610, 136, 890, 924, 989, 1141, 403, 1075, 42], 'chosen_samples_score': [0.4972153567314511, 0.4956236530401593, 0.4892691866048191, 0.4852502937162815, 0.4849754506801902, 0.48147652452778056, 0.47991722453150365, 0.4689549478257198, 0.46856989066430543, 0.46394825179194, 0.4596880199649277, 0.45556735708917173, 0.4534461498630036, 0.45222300104115787, 0.4515834320673844, 0.4509788018748798, 0.44745534680453625, 0.4462927320381658, 0.44325600709941515, 0.43665660289489694], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 479.9169523459859, 'batch_acquisition_elapsed_time': 42.102044503204525})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8254310344827587, 'nll': 0.3905571575822501, 'f1': 0.7243261084827812, 'precision': 0.6958704810424219, 'recall': 0.8083477925583189, 'ROC_AUC': 0.8662754267558417, 'PRC_AUC': 0.9827142478969985, 'specificity': 0.7846153846153846}, 'chosen_targets': [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [152, 924, 1080, 270, 808, 584, 328, 535, 202, 397, 223, 26, 1018, 1167, 1036, 178, 404, 999, 285, 125], 'chosen_samples_score': [0.49989435065413457, 0.49808266273387103, 0.4973374253669718, 0.49485308626785307, 0.49421497491897415, 0.4935720713059095, 0.49314634922059275, 0.49196453921437966, 0.48986241818348986, 0.4892703199264957, 0.48822189784311243, 0.48762232062491073, 0.48705347426970547, 0.48644209200593547, 0.4863977240333788, 0.4855199204220989, 0.4850573347027449, 0.4793636063119153, 0.47778941485107995, 0.4753418797148793], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.4548530508764, 'batch_acquisition_elapsed_time': 41.39180531864986})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9202586206896551, 'nll': 0.44199815289727573, 'f1': 0.8190689978606132, 'precision': 0.8542755787901419, 'recall': 0.7926547137073453, 'ROC_AUC': 0.8922035439851818, 'PRC_AUC': 0.983331693059033, 'specificity': 0.6153846153846154}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1], 'chosen_samples': [150, 44, 728, 1, 1144, 101, 66, 675, 273, 298, 186, 617, 325, 118, 715, 530, 404, 766, 452, 8], 'chosen_samples_score': [0.4969515382380181, 0.49180801145105946, 0.4804485967702192, 0.4752148998249196, 0.47370227944722343, 0.46352731752742093, 0.46009792961538465, 0.4578124041358044, 0.4535585727486058, 0.4375157967897665, 0.43657461015335375, 0.4339332875861289, 0.4293251307695184, 0.4258059728471445, 0.4221794024119715, 0.41039226775374715, 0.4088197838464044, 0.4080956903644548, 0.40256296868906105, 0.3999356671187485], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.0288885878399, 'batch_acquisition_elapsed_time': 40.71637128479779})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9331896551724138, 'nll': 0.554476836632038, 'f1': 0.8547290814523052, 'precision': 0.8731527093596059, 'recall': 0.8388085598611914, 'ROC_AUC': 0.8953780723785327, 'PRC_AUC': 0.9923517378792426, 'specificity': 0.7076923076923077}, 'chosen_targets': [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], 'chosen_samples': [533, 729, 8, 455, 376, 907, 58, 46, 542, 624, 57, 991, 657, 484, 213, 272, 1083, 640, 1025, 82], 'chosen_samples_score': [0.48640852576925864, 0.46789830104939023, 0.4629735789605849, 0.42992739173631167, 0.408482519728957, 0.3987839682016141, 0.3965484932080704, 0.3768911311258182, 0.3671048120547178, 0.3451318507409923, 0.32219598351749434, 0.31213921616296925, 0.296037519658251, 0.28780778285185504, 0.28400534978903225, 0.28364124952609704, 0.2830257718461672, 0.280959880640943, 0.2588013343579504, 0.2426873141190281], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.07843338511884, 'batch_acquisition_elapsed_time': 39.9093252918683})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8577586206896551, 'nll': 1.9972684136752425, 'f1': 0.9234338747099767, 'precision': 0.4298056155507559, 'recall': 0.49874686716791977, 'ROC_AUC': 0.4103082227841569, 'PRC_AUC': 0.8128392830384861, 'specificity': 0.0}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [956, 784, 276, 915, 398, 346, 805, 1059, 167, 120, 766, 748, 287, 706, 251, 215, 532, 410, 749, 1085], 'chosen_samples_score': [0.48563819889793647, 0.4814247081504972, 0.47724247561784594, 0.46614239320803275, 0.46601076184335966, 0.46174450512580556, 0.46088700788984605, 0.4543552278636721, 0.4437575064263306, 0.4433935269393271, 0.4197948895631147, 0.40831439486089727, 0.39560407257063157, 0.3938013023028294, 0.3914332924562405, 0.38102238356125995, 0.3795565371159313, 0.3772801958404636, 0.3747396530031821, 0.36495605186225877], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.9605937981978, 'batch_acquisition_elapsed_time': 39.25572548201308})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8448275862068966, 'nll': 0.29286029421049975, 'f1': 0.7559607293127628, 'precision': 0.7222506393861893, 'recall': 0.8518218623481781, 'ROC_AUC': 0.8886464803848267, 'PRC_AUC': 0.9889126650745624, 'specificity': 0.8615384615384616}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [616, 621, 842, 436, 1088, 447, 740, 1013, 574, 166, 655, 424, 309, 959, 45, 1026, 418, 409, 269, 572], 'chosen_samples_score': [0.4983074951345323, 0.4970803698419539, 0.49086474134324576, 0.48825781643933963, 0.48678195969796967, 0.48372469473953983, 0.48298209827105254, 0.4752096371418524, 0.47468675464470456, 0.4733167506443654, 0.4726077241310509, 0.4715324881819414, 0.4711285211667331, 0.46977155898034695, 0.4615853476147507, 0.4614942620289424, 0.46111843894545423, 0.4598913058445434, 0.4586202403813969, 0.4581642172473843], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.37494312226772, 'batch_acquisition_elapsed_time': 38.535093632061034})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.9137931034482759, 'nll': 0.22199096350834288, 'f1': 0.8409597257926307, 'precision': 0.8103367996275027, 'recall': 0.8854829381145171, 'ROC_AUC': 0.9321494029568307, 'PRC_AUC': 0.9939710166893512, 'specificity': 0.8461538461538461}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [537, 853, 671, 1002, 988, 723, 20, 118, 500, 557, 799, 101, 905, 432, 356, 306, 778, 153, 802, 655], 'chosen_samples_score': [0.49669358826503907, 0.4888263047537842, 0.4830724522527745, 0.4812883950362242, 0.48052504416476294, 0.4788989762339998, 0.4759693786854332, 0.47038139844426696, 0.468030445802478, 0.4663814608271676, 0.4640271415910143, 0.46224195665744094, 0.44520976298225956, 0.43922652901463755, 0.4386970014457493, 0.4380972431110768, 0.42902609332704655, 0.42444238789977773, 0.4179246476985663, 0.4148520065874747], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 519.2301409370266, 'batch_acquisition_elapsed_time': 37.83558101719245})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.28902001216493806, 'f1': 0.8472626472626472, 'precision': 0.8182019416247668, 'recall': 0.8879892037786774, 'ROC_AUC': 0.9048680306671668, 'PRC_AUC': 0.9922889799288857, 'specificity': 0.8461538461538461}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [661, 475, 212, 215, 45, 939, 806, 604, 269, 500, 259, 533, 441, 263, 2, 195, 859, 85, 536, 932], 'chosen_samples_score': [0.49575035724211847, 0.47633341079605573, 0.4710021647151069, 0.47080313414866315, 0.4671733388406284, 0.46573574377311666, 0.45953402547327293, 0.44695183238521663, 0.4450579227621808, 0.4336442965080063, 0.43153694373046037, 0.4255893425572287, 0.422669335618156, 0.41622693736281824, 0.40377713124410064, 0.39954683574687766, 0.3955980522647262, 0.3945285702529796, 0.3927009266898108, 0.39246602137717024], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 479.7775936778635, 'batch_acquisition_elapsed_time': 37.165108104236424})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8814655172413793, 'nll': 0.568191067925815, 'f1': 0.7141288884408151, 'precision': 0.7615800865800866, 'recall': 0.6863890495469442, 'ROC_AUC': 0.8336176521855441, 'PRC_AUC': 0.9710094150049767, 'specificity': 0.4153846153846154}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], 'chosen_samples': [923, 594, 610, 579, 713, 491, 821, 187, 525, 1013, 876, 59, 192, 866, 666, 312, 435, 231, 825, 29], 'chosen_samples_score': [0.4983690706648136, 0.49806398216353454, 0.49781710720150585, 0.4970485493756961, 0.49682873777643455, 0.4955273143987442, 0.49549610471286587, 0.4954325010505741, 0.49472496351933726, 0.49469589724531693, 0.49447046321229193, 0.4928812316394112, 0.49251542014094485, 0.49134109009936, 0.4901397354206021, 0.4897155416104645, 0.4895214408731249, 0.48946884026426885, 0.48944388344021805, 0.4891132817549477], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.46653817920014, 'batch_acquisition_elapsed_time': 36.51356986304745})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8663793103448276, 'nll': 0.9119712566507274, 'f1': 0.5332294911734164, 'precision': 0.7914973429196623, 'recall': 0.5359552727973781, 'ROC_AUC': 0.7971933420374626, 'PRC_AUC': 0.9679833452683879, 'specificity': 0.07692307692307693}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], 'chosen_samples': [280, 923, 59, 752, 275, 1003, 385, 187, 434, 334, 877, 983, 974, 971, 999, 696, 528, 115, 963, 119], 'chosen_samples_score': [0.48559440654234587, 0.47692123143890686, 0.45908922737990165, 0.4061855827424773, 0.40314916214110763, 0.3972637814425406, 0.382448777226851, 0.37736030719759894, 0.37166316801645116, 0.3445394569875665, 0.3434532511136711, 0.3422829879901639, 0.33787533783958545, 0.33530813093031353, 0.3231438118573364, 0.3147651740848486, 0.3129571240845823, 0.2957463815212452, 0.29099062808119625, 0.28922135086316036], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.54406247194856, 'batch_acquisition_elapsed_time': 35.70624004304409})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9116379310344828, 'nll': 1.3992220122238685, 'f1': 0.7596006823782144, 'precision': 0.9062211981566821, 'recall': 0.7039329091960671, 'ROC_AUC': 0.9316337111742785, 'PRC_AUC': 0.9910407040967989, 'specificity': 0.4153846153846154}, 'chosen_targets': [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [558, 93, 952, 924, 971, 323, 262, 972, 669, 305, 653, 734, 761, 9, 631, 325, 648, 279, 496, 985], 'chosen_samples_score': [0.4529913409394748, 0.40378672617365063, 0.3663963441012478, 0.24520492826497275, 0.1571296763794171, 0.13244808155099397, 0.12856494842099664, 0.10168221297959834, 0.06796897071689367, 0.04496201497265373, 0.039103430412611884, 0.02844651750687832, 0.02539466262919121, 0.02073237960488694, 0.01381298483085569, 0.013795541862717475, 0.01263544603322131, 0.012448511175883525, 0.011934301720392115, 0.010206260627118602], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.22136062802747, 'batch_acquisition_elapsed_time': 35.111861614044756})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8125, 'nll': 0.61886609833816, 'f1': 0.6698481242485953, 'precision': 0.6523001725129385, 'recall': 0.7042413726624253, 'ROC_AUC': 0.7943600461739522, 'PRC_AUC': 0.9326870725797621, 'specificity': 0.5538461538461539}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [747, 220, 36, 266, 946, 843, 49, 667, 421, 907, 252, 840, 368, 20, 85, 871, 734, 430, 448, 242], 'chosen_samples_score': [0.49903580672591563, 0.49818449968902334, 0.49817532710991597, 0.4974174809918106, 0.4954822182830818, 0.49472897029928053, 0.4942118093382045, 0.4934880452859304, 0.4905939873687827, 0.4880459925944207, 0.4860333506664879, 0.4854489586131798, 0.48444009326381265, 0.48392344975569934, 0.4838569644499072, 0.48297682971778844, 0.4828873946232861, 0.482325813399597, 0.48107573185976626, 0.4789909455089538], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.12497331993654, 'batch_acquisition_elapsed_time': 34.53986346721649})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9202586206896551, 'nll': 0.564418990036537, 'f1': 0.8106895144838842, 'precision': 0.868681090076971, 'recall': 0.7733371891266627, 'ROC_AUC': 0.9086415733886198, 'PRC_AUC': 0.9853095192377195, 'specificity': 0.5692307692307692}, 'chosen_targets': [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [58, 455, 558, 520, 463, 41, 385, 791, 472, 579, 634, 156, 55, 114, 195, 299, 528, 149, 898, 649], 'chosen_samples_score': [0.44849828709186124, 0.3173307923615992, 0.28437435107381526, 0.2653694489331573, 0.23839234886380356, 0.23594155326747235, 0.23078111913738286, 0.22923320716590556, 0.21103720279816163, 0.18823073147845681, 0.1856562481543782, 0.17562356942621737, 0.16708386439081402, 0.1639116930911333, 0.15533374670721567, 0.14508101293015563, 0.1423982064432776, 0.13577110178145746, 0.13430576064781297, 0.12912963656837884], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.20075844880193, 'batch_acquisition_elapsed_time': 33.998610948212445})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9439655172413793, 'nll': 0.3966816211568898, 'f1': 0.8719211822660099, 'precision': 0.9169633955277026, 'recall': 0.838635049161365, 'ROC_AUC': 0.9676191275328576, 'PRC_AUC': 0.9945020850156787, 'specificity': 0.6923076923076923}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [247, 391, 850, 801, 846, 380, 908, 592, 424, 30, 241, 399, 472, 872, 700, 279, 143, 688, 389, 196], 'chosen_samples_score': [0.3480185040691687, 0.20398292016164876, 0.14980191614253213, 0.14530689052858314, 0.1321943057915974, 0.12319062379072543, 0.12261467596565334, 0.09186370239733133, 0.07592758650753073, 0.06510201005896232, 0.05995039148357084, 0.05309446860253009, 0.049094245743595666, 0.0360770108002203, 0.03562953499107979, 0.034474902015290576, 0.03396213241023216, 0.03254641843768957, 0.03240744096297876, 0.03235865930012005], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.826365896035, 'batch_acquisition_elapsed_time': 33.205754672177136})
