store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT VARIATIONAL RATIOS (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 1234, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_vr_1234', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.variation_ratios', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_vr_1234', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=variation_ratios', '--acquisition_method=independent']
store['Distribution of training set classes:']={2: 809, 1: 582, 0: 232}
store['Distribution of validation set classes:']={1: 87, 0: 29, 2: 116}
store['Distribution of test set classes:']={2: 232, 0: 65, 1: 167}
store['Distribution of pool classes:']={2: 784, 1: 557, 0: 207}
store['Distribution of active set classes:']={1: 25, 0: 25, 2: 25}
store['active samples']=75
store['available samples']=1548
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[915, 280, 362, 68, 1359, 1566, 800, 1428, 1355, 156, 1093, 1388, 243, 222, 1525, 1301, 953, 1404, 1420, 492, 741, 160, 119, 51, 1293, 1426, 1272, 1365, 538, 21, 1321, 392, 451, 636, 864, 1546, 664, 252, 1425, 477, 1098, 801, 562, 290, 871, 780, 1458, 368, 835, 324, 308, 1110, 1306, 1320, 894, 947, 377, 1405, 225, 1052, 977, 1241, 82, 24, 1285, 1222, 1234, 1198, 537, 155, 999, 1014, 842, 1219, 830]
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.5172413793103449, 'nll': 2.4385946865739494, 'f1': 0.30629835376933745, 'precision': 0.5649164380144674, 'recall': 0.3648494787613103, 'ROC_AUC': 0.7210693359375, 'PRC_AUC': 0.5872361038043976, 'specificity': 0.6856801166389793}, 'chosen_targets': [1, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 0, 0, 1, 2, 0, 2], 'chosen_samples': [574, 1176, 1203, 695, 1185, 909, 579, 1148, 1074, 9, 520, 1100, 689, 156, 61, 25, 1143, 1012, 727, 51], 'chosen_samples_score': [0.6309458636527153, 0.5594248206896877, 0.5533524166319703, 0.5285667400587057, 0.5229295707412973, 0.5183103972328094, 0.5075491783044637, 0.5004682626958556, 0.4997865460782762, 0.49471265183537827, 0.49267345513499783, 0.48915360458783763, 0.48873977265104473, 0.4886756378317415, 0.48627541070409785, 0.4861285494381741, 0.4844894604880864, 0.48154893309942837, 0.48143772960373143, 0.4809789948694436], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.5386144546792, 'batch_acquisition_elapsed_time': 54.05206833872944})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.5775862068965517, 'nll': 1.517419354668979, 'f1': 0.5224873739565074, 'precision': 0.6126080892608089, 'recall': 0.5007150155921579, 'ROC_AUC': 0.7738037109375, 'PRC_AUC': 0.5984017257852244, 'specificity': 0.7578184038407874}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 0, 1, 2, 2, 1, 0, 0, 0, 2, 1, 1, 1, 2, 2], 'chosen_samples': [1169, 808, 30, 313, 593, 1327, 1262, 1524, 1220, 173, 761, 461, 870, 1305, 1322, 775, 307, 1316, 1160, 141], 'chosen_samples_score': [0.6411557844185518, 0.6192331828815606, 0.5898087138061798, 0.5580087758046348, 0.5576526415990204, 0.5554321919566176, 0.5426170591079478, 0.5350824724017826, 0.5267759915500301, 0.525564118341838, 0.5162789201918935, 0.5154311245019828, 0.5067887674925293, 0.5000489073158503, 0.4997808340132056, 0.4988171694342264, 0.4969452451505624, 0.49595602289660257, 0.4891577605273466, 0.48565664316409696], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 477.38871777430177, 'batch_acquisition_elapsed_time': 52.91951444372535})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5538793103448276, 'nll': 1.0410986933214912, 'f1': 0.4082296178607538, 'precision': 0.5254618254618254, 'recall': 0.4204811861687765, 'ROC_AUC': 0.75177001953125, 'PRC_AUC': 0.6182946732442245, 'specificity': 0.7269754119421391}, 'chosen_targets': [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 2, 1, 2, 1, 2, 2], 'chosen_samples': [1225, 1355, 747, 1, 865, 337, 971, 739, 787, 294, 1018, 828, 713, 1122, 750, 561, 1352, 956, 459, 1277], 'chosen_samples_score': [0.6516600388284812, 0.6302798059379269, 0.6258636408218884, 0.6111338494423375, 0.601877055138808, 0.6013543001448379, 0.5990170161873851, 0.5969312877921007, 0.5926646116858377, 0.588529811346352, 0.5884774263743879, 0.5871311572974238, 0.5821093023536803, 0.579000139606406, 0.5777825307146802, 0.5761188756321817, 0.5760989127971665, 0.5757548864719899, 0.5712271482461795, 0.5701705780707147], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.39639465278015, 'batch_acquisition_elapsed_time': 53.875283032190055})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6099137931034483, 'nll': 0.9035895117397966, 'f1': 0.5360540603689666, 'precision': 0.5721470342522975, 'recall': 0.5519136792727543, 'ROC_AUC': 0.7904052734375, 'PRC_AUC': 0.67689267832011, 'specificity': 0.7686745235898291}, 'chosen_targets': [2, 1, 1, 0, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1], 'chosen_samples': [350, 1050, 720, 104, 290, 998, 187, 1186, 372, 160, 1211, 1030, 2, 117, 107, 1113, 1299, 298, 321, 75], 'chosen_samples_score': [0.6418101031221664, 0.6408137274032832, 0.6334486459825441, 0.6237395128290535, 0.6181736095951187, 0.6126775570210079, 0.6125445450076301, 0.611086673048703, 0.6060216121944811, 0.6040994532259443, 0.5983793975929166, 0.5980454238917529, 0.5920964740719715, 0.591303806113854, 0.5885463355858254, 0.5852593846388934, 0.5832541079092393, 0.5794494738996747, 0.5792232623000362, 0.5764834629287716], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.06668108236045, 'batch_acquisition_elapsed_time': 52.987542249727994})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.6185344827586207, 'nll': 0.7561863866345636, 'f1': 0.5480042119386382, 'precision': 0.606658913892868, 'recall': 0.5602337764788725, 'ROC_AUC': 0.797119140625, 'PRC_AUC': 0.6820802122087432, 'specificity': 0.7680110019039238}, 'chosen_targets': [1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0, 2, 2, 0, 1], 'chosen_samples': [100, 760, 1224, 432, 282, 803, 878, 982, 704, 1217, 1464, 1441, 1218, 596, 514, 539, 1201, 463, 66, 335], 'chosen_samples_score': [0.6507155159764364, 0.6408494205062092, 0.638657422946062, 0.6361086666205769, 0.6279516527709668, 0.6244599596565634, 0.6238874714523439, 0.6235856693183599, 0.6225114857256997, 0.6219834912404696, 0.6208233064167934, 0.6108079339081354, 0.6089018622784013, 0.6054674515161416, 0.6044351065305283, 0.603552402241039, 0.6032085417263969, 0.6024882760618979, 0.6017077376931492, 0.6014565019342981], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.0137775549665, 'batch_acquisition_elapsed_time': 51.20252227364108})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6012931034482759, 'nll': 0.6869183244376347, 'f1': 0.5150747248025249, 'precision': 0.5962079819341168, 'recall': 0.5135029675397216, 'ROC_AUC': 0.7841796875, 'PRC_AUC': 0.658576173148434, 'specificity': 0.7535925541823908}, 'chosen_targets': [0, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 0, 1, 0], 'chosen_samples': [1046, 925, 417, 792, 908, 475, 471, 301, 1398, 1082, 1181, 506, 596, 872, 1425, 1216, 1318, 92, 655, 876], 'chosen_samples_score': [0.6625166289731956, 0.6500727191688677, 0.6482926483416681, 0.6459971961443378, 0.6423076378648023, 0.640629501523365, 0.6261863440653612, 0.6259869183692586, 0.6258904985603506, 0.6236242475325342, 0.6233142439541309, 0.6206373714782929, 0.6185209301543628, 0.6173226671393819, 0.6158908258305795, 0.6121532195016813, 0.6101696297451655, 0.6099031727143808, 0.6059795956125149, 0.6048884866262291], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.0785509608686, 'batch_acquisition_elapsed_time': 50.54689015308395})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6379310344827587, 'nll': 0.6746393401047279, 'f1': 0.5889298678273188, 'precision': 0.6051417572747191, 'recall': 0.6290953901216135, 'ROC_AUC': 0.81103515625, 'PRC_AUC': 0.6992811213674128, 'specificity': 0.7894436762404095}, 'chosen_targets': [2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2], 'chosen_samples': [701, 1129, 1331, 443, 294, 840, 289, 1406, 1338, 1238, 38, 1289, 1204, 575, 788, 548, 611, 212, 737, 683], 'chosen_samples_score': [0.651597959381471, 0.6467194127137925, 0.6378779653689366, 0.6365986522602767, 0.6271590446010726, 0.6194101801322374, 0.6143143749379787, 0.6136729762169042, 0.6128985109995453, 0.6063576414252466, 0.6028868464412698, 0.6027714929189838, 0.6006919906923989, 0.5979340404091945, 0.596079773724534, 0.5956657646547363, 0.595397292470514, 0.592828802757059, 0.5915083047221095, 0.5894542097729426], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.504667583853, 'batch_acquisition_elapsed_time': 49.85255237715319})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6357758620689655, 'nll': 0.9484601514092807, 'f1': 0.5946082125335138, 'precision': 0.6210470085470085, 'recall': 0.6162482726853984, 'ROC_AUC': 0.7989501953125, 'PRC_AUC': 0.6813829897281938, 'specificity': 0.7837973765832205}, 'chosen_targets': [0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1], 'chosen_samples': [163, 856, 793, 1323, 928, 1036, 654, 319, 796, 1356, 901, 1299, 1124, 458, 1043, 1193, 849, 1042, 932, 579], 'chosen_samples_score': [0.6504730312752686, 0.6336699160290138, 0.6195675309735909, 0.6165991174227992, 0.6094961933787769, 0.6066640986809286, 0.6054670658472383, 0.6053841607514425, 0.6029697958036202, 0.599678196450208, 0.5937311088558127, 0.589971310056295, 0.5888984567576218, 0.5877699874418525, 0.5860135899412666, 0.5857824405054143, 0.5830371092252725, 0.5799502082472068, 0.5795865681194505, 0.5793347259266772], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.75374582596123, 'batch_acquisition_elapsed_time': 48.85454964824021})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6228448275862069, 'nll': 0.6339064630968817, 'f1': 0.5505763240808911, 'precision': 0.5932539682539683, 'recall': 0.5876056904758123, 'ROC_AUC': 0.8045654296875, 'PRC_AUC': 0.7087155507199255, 'specificity': 0.7744651888632531}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 2, 1, 1, 2, 0, 1, 1, 1, 2, 1], 'chosen_samples': [35, 1250, 498, 922, 586, 550, 1242, 673, 923, 1149, 1385, 1136, 852, 174, 1375, 957, 565, 640, 866, 736], 'chosen_samples_score': [0.6589079094079286, 0.6522250889469927, 0.6477670968387923, 0.6470519275948552, 0.6420262316047106, 0.6409604242782403, 0.6403266315942995, 0.6381986524578094, 0.6381710782896797, 0.6376742084246421, 0.6376540486596862, 0.6372637031451467, 0.6320503694154442, 0.6255888956537354, 0.6246004282792843, 0.6237349953638682, 0.6219625981624729, 0.6177990216015983, 0.614092197509708, 0.6140282369778022], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.70587676530704, 'batch_acquisition_elapsed_time': 48.28168735979125})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5366379310344828, 'nll': 0.9587452329438308, 'f1': 0.4620096331752774, 'precision': 0.7312409812409811, 'recall': 0.4718460956071941, 'ROC_AUC': 0.7442626953125, 'PRC_AUC': 0.5636279760397617, 'specificity': 0.7489502302720693}, 'chosen_targets': [0, 1, 0, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2], 'chosen_samples': [1245, 745, 1032, 860, 771, 561, 531, 653, 470, 600, 689, 1353, 398, 367, 357, 1013, 417, 154, 97, 317], 'chosen_samples_score': [0.5832434606246353, 0.5758906037841856, 0.5754053029910979, 0.5712043279321244, 0.5641273675338938, 0.5539352123522227, 0.5538159920083069, 0.5512825263414367, 0.5491882037529374, 0.5464796717443309, 0.5449409422980765, 0.5370438340337214, 0.536669663242076, 0.5356694709965564, 0.5343360783314006, 0.5315268440276205, 0.5284510864425722, 0.5271116811143957, 0.5266530254591607, 0.5243451252244968], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.06077708490193, 'batch_acquisition_elapsed_time': 47.48079716600478})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6336206896551724, 'nll': 0.7067039095122238, 'f1': 0.6111205397017254, 'precision': 0.5967785967785968, 'recall': 0.6700699396962045, 'ROC_AUC': 0.818115234375, 'PRC_AUC': 0.7414736318192391, 'specificity': 0.8036559964872124}, 'chosen_targets': [1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 2], 'chosen_samples': [733, 113, 963, 275, 277, 1056, 617, 1035, 1249, 516, 596, 243, 1139, 504, 673, 98, 954, 798, 848, 271], 'chosen_samples_score': [0.5956054395293481, 0.5867061630722243, 0.5835391903045057, 0.5825947360557879, 0.582060138755149, 0.5768955395416437, 0.5679573865150456, 0.567335699424812, 0.5671578654550379, 0.564885807378106, 0.5637992668870686, 0.5608907945005653, 0.5608382176572313, 0.5600939318580429, 0.558389794645942, 0.5581555744199922, 0.5556164799056034, 0.5525788770888054, 0.5522867879831788, 0.549300725037377], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.44864355493337, 'batch_acquisition_elapsed_time': 46.891325471922755})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.7025862068965517, 'nll': 0.9287283009496229, 'f1': 0.7077678653264489, 'precision': 0.7458654650190467, 'recall': 0.6843862672532919, 'ROC_AUC': 0.8585205078125, 'PRC_AUC': 0.7475958267807883, 'specificity': 0.8295722616654255}, 'chosen_targets': [1, 2, 2, 0, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 0], 'chosen_samples': [1241, 946, 1166, 1266, 436, 958, 1027, 189, 801, 1112, 380, 597, 376, 1263, 1039, 1253, 512, 617, 408, 863], 'chosen_samples_score': [0.5712382161080581, 0.5651073939324875, 0.5457213123793763, 0.5360180410517019, 0.5332927212784545, 0.5263822921257062, 0.5194862538607451, 0.5082381837968855, 0.5050738664411083, 0.5010239850195228, 0.4973170692960004, 0.49586904144776733, 0.4950366219213741, 0.493057686181291, 0.49251529427882357, 0.4910531694941155, 0.49068159813044043, 0.48890325824590986, 0.48882181565807503, 0.4883902114897436], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 560.8050792668946, 'batch_acquisition_elapsed_time': 46.10800743382424})
