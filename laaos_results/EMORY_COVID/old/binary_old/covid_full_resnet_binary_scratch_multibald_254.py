store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT MULTI BALD (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 254, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_multibald_254', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_multibald_254', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=bald', '--acquisition_method=multibald']
store['Distribution of training set classes:']={1: 1400, 0: 223}
store['Distribution of validation set classes:']={1: 194, 0: 38}
store['Distribution of test set classes:']={1: 399, 0: 65}
store['Distribution of pool classes:']={1: 1375, 0: 198}
store['Distribution of active set classes:']={1: 25, 0: 25}
store['active samples']=50
store['available samples']=1573
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[298, 805, 1312, 1018, 966, 1290, 964, 1260, 1475, 1583, 949, 631, 503, 394, 1442, 1164, 492, 942, 1365, 853, 710, 1120, 1382, 352, 1381, 696, 913, 1353, 130, 1429, 390, 1499, 73, 1109, 79, 1487, 567, 316, 416, 1601, 925, 1009, 1160, 211, 1576, 1390, 662, 1096, 1110, 813]
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8275862068965517, 'nll': 0.5708010443325701, 'f1': 0.7243018419489007, 'precision': 0.6959013330011212, 'recall': 0.8031617505301716, 'ROC_AUC': 0.8841449262210757, 'PRC_AUC': 0.9716436954782286, 'specificity': 0.7692307692307693}, 'chosen_targets': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], 'chosen_samples': [1332, 1562, 382, 835, 747, 369, 904, 40, 816, 1494, 635, 572, 679, 575, 1134, 105, 86, 81, 1047, 585], 'chosen_samples_score': [0.03851631899511476, 0.06973732481110873, 0.09883744871418365, 0.12437403309724848, 0.14897723121174078, 0.1723109368061686, 0.19439660755335186, 0.2155133535024647, 0.23566204268106272, 0.2550059594958096, 0.27388170361372755, 0.2920174709864476, 0.3099017121845957, 0.327165821311759, 0.34422311126229665, 0.3606819286353229, 0.36416704506902775, 0.4010928433457792, 0.39420520031908346, 0.4248998030811766], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.8096596850082, 'batch_acquisition_elapsed_time': 76.33081366494298})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8879310344827587, 'nll': 0.968072693923424, 'f1': 0.7732245573141847, 'precision': 0.7667217024399193, 'recall': 0.7802968960863698, 'ROC_AUC': 0.8581612485180493, 'PRC_AUC': 0.9665931945065237, 'specificity': 0.6307692307692307}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], 'chosen_samples': [756, 771, 633, 1413, 1305, 691, 1502, 483, 454, 1405, 1523, 418, 859, 848, 1262, 108, 1363, 865, 128, 593], 'chosen_samples_score': [0.05326777298381846, 0.09665541903708452, 0.1347001336809941, 0.16835441203230106, 0.19993311840328865, 0.22805243433794553, 0.25556799291759447, 0.28208438823732873, 0.3076585112767338, 0.33191773855753937, 0.35554703403907606, 0.37898588730676686, 0.40144155240531987, 0.4235416185769134, 0.4453273170206016, 0.46622708165937077, 0.5085948916004668, 0.5031580781274112, 0.525022937525339, 0.5464297522237267], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 397.2863653367385, 'batch_acquisition_elapsed_time': 75.37610769923776})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.8603136786099138, 'f1': 0.7932263814616756, 'precision': 0.783141597677669, 'recall': 0.804626951995373, 'ROC_AUC': 0.8740701030380148, 'PRC_AUC': 0.9777228851369333, 'specificity': 0.676923076923077}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], 'chosen_samples': [645, 530, 1235, 1088, 1480, 337, 1409, 102, 1255, 668, 339, 451, 1496, 232, 967, 191, 604, 1136, 1027, 1389], 'chosen_samples_score': [0.04682964258731148, 0.08971886045821953, 0.12875410068546733, 0.16417092220880858, 0.19678381551967483, 0.22720592454222466, 0.2565799108205211, 0.2830129495283833, 0.30902696033776156, 0.3334401724279381, 0.3574485074056275, 0.3803937190003577, 0.4022920909346892, 0.4236373813424219, 0.4444342553237055, 0.4646277082244996, 0.4871568217298208, 0.49652985848786635, 0.511116660441429, 0.5408467658293148], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 398.87204432394356, 'batch_acquisition_elapsed_time': 74.22986599896103})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.8470697731807314, 'f1': 0.7635467980295566, 'precision': 0.7958030669895076, 'recall': 0.7402352033930981, 'ROC_AUC': 0.8524477589779313, 'PRC_AUC': 0.9810584161782582, 'specificity': 0.5230769230769231}, 'chosen_targets': [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 'chosen_samples': [760, 507, 1504, 985, 979, 171, 914, 915, 1181, 697, 168, 1348, 419, 602, 648, 1445, 1069, 1217, 1243, 875], 'chosen_samples_score': [0.03436909287105705, 0.06429610835922062, 0.09079907383180963, 0.1120956004735505, 0.13265841759435615, 0.1518000053813835, 0.17021918492689592, 0.18805224729743664, 0.20536038469637408, 0.22211796224041613, 0.23876470023084728, 0.2545048853210252, 0.2694929884620221, 0.2840236671856724, 0.29801688261049186, 0.3115967250188376, 0.3276113644839693, 0.33842703308422273, 0.3529293396748088, 0.3730049445296686], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.18599477084354, 'batch_acquisition_elapsed_time': 73.24980747373775})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9030172413793104, 'nll': 0.380217091790561, 'f1': 0.8181295565601401, 'precision': 0.7926510024262546, 'recall': 0.853460574513206, 'ROC_AUC': 0.881641662801864, 'PRC_AUC': 0.9858682357790246, 'specificity': 0.7846153846153846}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [1067, 709, 124, 679, 1453, 301, 731, 665, 244, 186, 1344, 1465, 1265, 325, 108, 1117, 1380, 1219, 275, 2], 'chosen_samples_score': [0.021056987770498137, 0.041106908179340595, 0.0595502309177931, 0.07736411393225717, 0.09450839161395619, 0.1110124624815132, 0.12694093070653611, 0.14231430154918323, 0.15718924251305655, 0.17163681527046215, 0.18569954001647115, 0.19938253035226072, 0.212302044454451, 0.22491222694968727, 0.23710797986997534, 0.248837757982745, 0.2580163118525345, 0.25819125524083475, 0.28131757012089054, 0.285815502550296], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 280.1395168961026, 'batch_acquisition_elapsed_time': 72.22555021103472})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.9073275862068966, 'nll': 0.9543712221342942, 'f1': 0.7478738863966639, 'precision': 0.8884024577572964, 'recall': 0.6949874686716793, 'ROC_AUC': 0.8628987726389546, 'PRC_AUC': 0.9805261745789646, 'specificity': 0.4}, 'chosen_targets': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 'chosen_samples': [409, 126, 1458, 1010, 1153, 873, 68, 54, 426, 1282, 644, 702, 1373, 993, 1065, 1242, 1092, 667, 666, 1403], 'chosen_samples_score': [0.010077305247759627, 0.019177760486171203, 0.028029927371768126, 0.03645649221006941, 0.043923300061377235, 0.05126071953426248, 0.05846386329088027, 0.06542634511807632, 0.07226826884008108, 0.07898261597417644, 0.08557955240644777, 0.09207772845688833, 0.09842125636075849, 0.10467454440212443, 0.11076845262276169, 0.11678308431244133, 0.11429743789706492, 0.11637600711561547, 0.13000601132661238, 0.1345720709881295], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.45071494812146, 'batch_acquisition_elapsed_time': 71.50461822003126})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8793103448275862, 'nll': 2.232233507879849, 'f1': 0.6334913112164298, 'precision': 0.811117681845062, 'recall': 0.6014266435319067, 'ROC_AUC': 0.8222683719714371, 'PRC_AUC': 0.9643899405186794, 'specificity': 0.2153846153846154}, 'chosen_targets': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'chosen_samples': [1446, 372, 826, 511, 1410, 1076, 769, 274, 1174, 27, 720, 1355, 1226, 644, 958, 1327, 520, 665, 1195, 929], 'chosen_samples_score': [0.006458796975387426, 0.012531294627104339, 0.01836760662664849, 0.02404269388696867, 0.02952753312255396, 0.034895150008770415, 0.04010603855676731, 0.04527656428069626, 0.05035571323007382, 0.05534570541963646, 0.060105835772255034, 0.06466612885763023, 0.0691680082378987, 0.07354872384925404, 0.07786992328824383, 0.08211225143814893, 0.09109427967537442, 0.07913089547392183, 0.09757967978736559, 0.0992931512480748], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.52600717823952, 'batch_acquisition_elapsed_time': 70.30495579401031})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9245689655172413, 'nll': 0.5625526165140087, 'f1': 0.8263008717043692, 'precision': 0.8698067632850242, 'recall': 0.7951609793715058, 'ROC_AUC': 0.9515768564307836, 'PRC_AUC': 0.9922666187778326, 'specificity': 0.6153846153846154}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], 'chosen_samples': [649, 1248, 1366, 730, 1358, 1239, 562, 1157, 64, 1171, 653, 700, 657, 373, 1251, 112, 1156, 813, 462, 991], 'chosen_samples_score': [0.036196866506801095, 0.06070547353633393, 0.08364558640772546, 0.1052885566761157, 0.12457045047696269, 0.1425179270254313, 0.15952001702022933, 0.17324431949819807, 0.18627791423488294, 0.19883668056527792, 0.21133634232409726, 0.2234549098498837, 0.2351239582929594, 0.245919459939822, 0.25645984060061444, 0.2668476477315256, 0.2714170827643656, 0.2880021035243612, 0.30365256157732645, 0.30131967307611696], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 319.80636710487306, 'batch_acquisition_elapsed_time': 69.60292680282146})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8943965517241379, 'nll': 0.5305599673040982, 'f1': 0.7876231843444959, 'precision': 0.7792240754169688, 'recall': 0.7969346443030654, 'ROC_AUC': 0.9127844710543465, 'PRC_AUC': 0.9854990246349467, 'specificity': 0.6615384615384615}, 'chosen_targets': [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1], 'chosen_samples': [1083, 110, 474, 518, 1243, 525, 494, 1007, 276, 183, 1371, 1275, 131, 172, 1171, 25, 1382, 21, 336, 178], 'chosen_samples_score': [0.02878375527667021, 0.055463580713004945, 0.07659432267321498, 0.09603815180725706, 0.11086853322918033, 0.12426569498378814, 0.13700398025718163, 0.1491341900694172, 0.16097025660264386, 0.17128104415701717, 0.18124321025850332, 0.1889160923683315, 0.19621664892556634, 0.20331406636334215, 0.21022984871123818, 0.21700621738158787, 0.22888131333718498, 0.226430934622325, 0.22798311699636997, 0.23032805182928762], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.89740341575816, 'batch_acquisition_elapsed_time': 68.54867841303349})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9353448275862069, 'nll': 0.48309418250774516, 'f1': 0.8499935342040605, 'precision': 0.9008605851979345, 'recall': 0.8143049932523616, 'ROC_AUC': 0.9634869892772191, 'PRC_AUC': 0.9921197725837836, 'specificity': 0.6461538461538462}, 'chosen_targets': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1], 'chosen_samples': [1291, 444, 1092, 603, 251, 1093, 121, 1179, 630, 599, 675, 589, 654, 996, 226, 496, 547, 1120, 1285, 380], 'chosen_samples_score': [0.022081413757413793, 0.04178490053015005, 0.06054200686867883, 0.07733769506197552, 0.09340615541428665, 0.10771700905375114, 0.12146077551109391, 0.13433280029649408, 0.14682495198574141, 0.1589532143102499, 0.17038516726185104, 0.1813948389959652, 0.19138109689182592, 0.20127558936051848, 0.21087089385524038, 0.22014873402739, 0.235561809633829, 0.22715566064685966, 0.2666293759568994, 0.2595681800082197], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.38163335295394, 'batch_acquisition_elapsed_time': 67.24778163898736})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.9181034482758621, 'nll': 0.7851859783304149, 'f1': 0.7792467948717949, 'precision': 0.9261714966847947, 'recall': 0.7205706574127627, 'ROC_AUC': 0.9130071187769944, 'PRC_AUC': 0.9893134497952459, 'specificity': 0.4461538461538462}, 'chosen_targets': [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], 'chosen_samples': [361, 30, 953, 1316, 8, 437, 541, 1222, 1239, 1055, 1088, 765, 1118, 100, 286, 828, 672, 374, 668, 258], 'chosen_samples_score': [0.007925511168795696, 0.015634180798994257, 0.023036399914345473, 0.03023859334403145, 0.03716721510828247, 0.04389512687751829, 0.050507013473776396, 0.05701788420922904, 0.06344319994726533, 0.06977322226500338, 0.07600538032771897, 0.08211492049052005, 0.08812879751928371, 0.09404807682485483, 0.0999085982681196, 0.10572059300445869, 0.12349796848630135, 0.1126144210154294, 0.12624845289180087, 0.11406154369389299], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.97962163574994, 'batch_acquisition_elapsed_time': 66.54819675302133})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9310344827586207, 'nll': 0.33526095028581293, 'f1': 0.8684807256235828, 'precision': 0.8427420680585238, 'recall': 0.9019471756313862, 'ROC_AUC': 0.9108445617724833, 'PRC_AUC': 0.9890577229015193, 'specificity': 0.8615384615384616}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [199, 1079, 1100, 1020, 38, 816, 214, 1015, 560, 907, 257, 773, 1207, 1180, 949, 1245, 663, 223, 1095, 331], 'chosen_samples_score': [0.013016381479236061, 0.02475561111582547, 0.03614812464642214, 0.04705785550584274, 0.057505985826183714, 0.06701670361594836, 0.07631189634132562, 0.0853524449812273, 0.09379233034870182, 0.1019442132596593, 0.10987701883534484, 0.11730635613602747, 0.1246119756737416, 0.13164085112751334, 0.13857347099963313, 0.1454536840750027, 0.16468142089335913, 0.15734631899069385, 0.151259467973464, 0.18785792839276994], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 318.64700076403096, 'batch_acquisition_elapsed_time': 65.53601199481636})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 0.3317053235810378, 'f1': 0.8410232983097304, 'precision': 0.8367983758787925, 'recall': 0.8454212454212454, 'ROC_AUC': 0.9008612286234413, 'PRC_AUC': 0.990495997554838, 'specificity': 0.7384615384615385}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 'chosen_samples': [1150, 946, 1152, 60, 727, 82, 338, 1135, 110, 105, 814, 539, 1184, 628, 594, 549, 302, 1015, 154, 850], 'chosen_samples_score': [0.013712469405275152, 0.024379228957296273, 0.03475117758446422, 0.044555519187485126, 0.053372160317280315, 0.06192270917490372, 0.07022889071328375, 0.07843719710925168, 0.08642945639514465, 0.09431839622925953, 0.10186449346505988, 0.10906623390383618, 0.11612468463556258, 0.12305546634776032, 0.12995896868375212, 0.13674478571563498, 0.14092738871442734, 0.14527587802436415, 0.15265122073381754, 0.15994550161060772], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 280.0585759473033, 'batch_acquisition_elapsed_time': 66.67581770615652})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.8987068965517241, 'nll': 0.25948072301930397, 'f1': 0.8141263306826221, 'precision': 0.7851298141995816, 'recall': 0.8573934837092732, 'ROC_AUC': 0.9164833272304537, 'PRC_AUC': 0.9902380809471313, 'specificity': 0.8}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [37, 268, 1267, 431, 676, 1210, 28, 1046, 910, 972, 1024, 551, 9, 1290, 665, 1020, 115, 929, 574, 433], 'chosen_samples_score': [0.010828591592920267, 0.02139745535373938, 0.03166968969650785, 0.0415076571210502, 0.050568382474871054, 0.0594573542665251, 0.06814024065392665, 0.0765870635923811, 0.08480520744079989, 0.09261496427694649, 0.10018931177738999, 0.10761203441682632, 0.11477940684412147, 0.12178955223148957, 0.12866363583820828, 0.13542281058091632, 0.1383778946033729, 0.1304142429995938, 0.14250462127978203, 0.14633944796638154], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.53863043524325, 'batch_acquisition_elapsed_time': 64.84732563327998})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8620689655172413, 'nll': 0.57304388901283, 'f1': 0.7208917628482274, 'precision': 0.7156485048614933, 'recall': 0.7266242529400424, 'ROC_AUC': 0.8379958689405529, 'PRC_AUC': 0.9661295579209701, 'specificity': 0.5384615384615384}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0], 'chosen_samples': [824, 442, 1249, 1165, 94, 187, 1154, 1074, 1093, 56, 716, 581, 447, 885, 1035, 28, 1271, 990, 770, 250], 'chosen_samples_score': [0.008826794360737078, 0.015973533435807208, 0.022879592480749444, 0.029643239931401277, 0.03624681134925334, 0.042685532531217873, 0.04880017940611037, 0.05483421582042425, 0.06076177703115615, 0.06661664500035425, 0.07238264912933623, 0.07806518559100972, 0.08367805651954896, 0.0891670399252007, 0.09455391156874882, 0.09984600271003963, 0.09882476620439462, 0.10498944050315018, 0.11355256662508495, 0.12681798187590587], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.98136887699366, 'batch_acquisition_elapsed_time': 63.75118619995192})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.9439655172413793, 'nll': 0.5090052834872542, 'f1': 0.8755775577557755, 'precision': 0.9052678372971772, 'recall': 0.85151339888182, 'ROC_AUC': 0.882345079489476, 'PRC_AUC': 0.98633871272758, 'specificity': 0.7230769230769231}, 'chosen_targets': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0], 'chosen_samples': [832, 390, 287, 929, 137, 325, 103, 1041, 1003, 1230, 47, 874, 1237, 114, 1128, 1259, 278, 59, 180, 911], 'chosen_samples_score': [0.010460871219716061, 0.019140219614252163, 0.027581964731009823, 0.035676125477682064, 0.043013233831775466, 0.050175344533486044, 0.056752049461476606, 0.06268302456985575, 0.06845413016432556, 0.07412424157829456, 0.07972416017696471, 0.08527749709561938, 0.09074630838010922, 0.09617976326043465, 0.10154570715874911, 0.10685099513837315, 0.11364009234628192, 0.11638823496435968, 0.1239060852685192, 0.1252190759119607], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.56682382104918, 'batch_acquisition_elapsed_time': 62.041234641335905})
