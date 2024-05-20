store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT MULTI BALD (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 1234, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_multibald_1234', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_multibald_1234', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=bald', '--acquisition_method=multibald']
# store['Distribution of training set classes:']={1: 1397, 0: 226}
# store['Distribution of validation set classes:']={1: 197, 0: 35}
# store['Distribution of test set classes:']={0: 65, 1: 399}
# store['Distribution of pool classes:']={1: 1372, 0: 201}
# store['Distribution of active set classes:']={1: 25, 0: 25}
# store['active samples']=50
# store['available samples']=1573
# store['validation samples']=232
# store['test samples']=464
store['iterations']=[]
store['initial_samples']=[915, 280, 362, 68, 1359, 1566, 800, 308, 1272, 1428, 1306, 1355, 156, 1365, 538, 21, 1320, 1388, 243, 894, 947, 1405, 1321, 222, 1301, 1426, 1110, 1093, 377, 1525, 1404, 636, 225, 252, 977, 1098, 562, 1222, 599, 1171, 702, 1474, 1014, 842, 830, 991, 138, 645, 685, 150]
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 1.4403998276283, 'f1': 0.773502013586625, 'precision': 0.7901633691107375, 'recall': 0.7595527279737806, 'ROC_AUC': 0.8910118886089249, 'PRC_AUC': 0.9699486139498539, 'specificity': 0.5692307692307692}, 'chosen_targets': [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], 'chosen_samples': [18, 1456, 1040, 205, 1310, 1170, 911, 291, 598, 1192, 194, 1412, 317, 708, 250, 1383, 136, 602, 242, 1166], 'chosen_samples_score': [0.040228967041998964, 0.07639880835494806, 0.11156416399786417, 0.14446626698761778, 0.17548645040409738, 0.20408650783324767, 0.23153076007126305, 0.2583008724406657, 0.28448456987736215, 0.3082976182567947, 0.33089397292267453, 0.3525702253686056, 0.3731757623637888, 0.3932607371296344, 0.4126525862579715, 0.4317407901951906, 0.4460041535437611, 0.46028347209272624, 0.5010446199204655, 0.4923911691709506], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 319.65929602179676, 'batch_acquisition_elapsed_time': 75.80029454408213})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8556034482758621, 'nll': 1.6530036926269531, 'f1': 0.6071523346180578, 'precision': 0.6745775729646697, 'recall': 0.5876421823790245, 'ROC_AUC': 0.7627803295001139, 'PRC_AUC': 0.9458024953463195, 'specificity': 0.2153846153846154}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0], 'chosen_samples': [738, 919, 98, 850, 1472, 1519, 0, 78, 833, 353, 1454, 608, 243, 1354, 438, 1464, 1233, 544, 1392, 996], 'chosen_samples_score': [0.08850256310607019, 0.1612871950414505, 0.21947276114703995, 0.26279132443616526, 0.30313790723370637, 0.33832555622779825, 0.37032784697473975, 0.3987489349387374, 0.4253376568407763, 0.44998542937600394, 0.46935484160531526, 0.48791807016791733, 0.5053626101742905, 0.5221138664790512, 0.5381648238083194, 0.5536004860076371, 0.5689003911326527, 0.5854588781218357, 0.6030248041398139, 0.6186228408081647], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 200.41113299131393, 'batch_acquisition_elapsed_time': 74.81255299877375})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9008620689655172, 'nll': 0.8521466748467807, 'f1': 0.7968631033957667, 'precision': 0.7931877138238279, 'recall': 0.8006940427993059, 'ROC_AUC': 0.8847825512179742, 'PRC_AUC': 0.9866704191029055, 'specificity': 0.6615384615384615}, 'chosen_targets': [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], 'chosen_samples': [1282, 1431, 20, 1002, 17, 1517, 1150, 1212, 1436, 321, 176, 681, 393, 1361, 427, 572, 1491, 831, 1365, 256], 'chosen_samples_score': [0.09424300342064584, 0.17148945837657315, 0.23521945232727481, 0.28422374944582396, 0.32504685535094735, 0.363642437561952, 0.39327160083190904, 0.4213846645467645, 0.4441073996944951, 0.4655402974325593, 0.48624952347917105, 0.5062331100530768, 0.5257350802024954, 0.5447019024649178, 0.5633256289526862, 0.5812244079885449, 0.5833451957462898, 0.6171788306394692, 0.6353513235620554, 0.6593929602806785], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 556.0507016866468, 'batch_acquisition_elapsed_time': 73.49872585618868})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8879310344827587, 'nll': 2.18237462537042, 'f1': 0.7318515225605691, 'precision': 0.779262795014585, 'recall': 0.7030267977636399, 'ROC_AUC': 0.8372618542903629, 'PRC_AUC': 0.9666254724510123, 'specificity': 0.4461538461538462}, 'chosen_targets': [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1], 'chosen_samples': [1293, 650, 719, 1405, 31, 1342, 1476, 658, 414, 757, 767, 676, 524, 116, 51, 12, 146, 793, 313, 1035], 'chosen_samples_score': [0.045386669744786556, 0.07588757450771189, 0.10373395631703564, 0.12427145941027229, 0.14469337047238984, 0.1640887731412799, 0.18274499300106983, 0.20117227669621185, 0.21885395342703173, 0.23618942217593997, 0.2531247786427162, 0.2698877193686604, 0.2859335681982609, 0.3016481320401372, 0.3169750045872135, 0.33214621055723725, 0.33937186178565426, 0.3610936518657919, 0.3626131890085045, 0.39486391834146595], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.82057481911033, 'batch_acquisition_elapsed_time': 72.81919417902827})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 0.5673269074538658, 'f1': 0.7983072949960368, 'precision': 0.7708564045773348, 'recall': 0.8395026026604975, 'ROC_AUC': 0.8921109052598984, 'PRC_AUC': 0.9783213807027927, 'specificity': 0.7692307692307693}, 'chosen_targets': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [442, 1207, 298, 1113, 106, 582, 1398, 1483, 211, 834, 1358, 64, 920, 506, 1125, 980, 335, 1196, 1176, 317], 'chosen_samples_score': [0.021428525768447337, 0.03719457794411851, 0.052270398443601485, 0.06563984877243612, 0.07789076122732164, 0.08960374442574448, 0.10094194188289629, 0.1119244184323045, 0.12282796961153863, 0.13345511152474465, 0.14398296857351767, 0.154349432685418, 0.16447909984101194, 0.17443397243617742, 0.1842840619145436, 0.19388739770017338, 0.20677584218560163, 0.20871413480509204, 0.2225287384310537, 0.2206137087936355], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.00662197871134, 'batch_acquisition_elapsed_time': 72.17235620273277})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8814655172413793, 'nll': 0.7429626070219895, 'f1': 0.7185925215300981, 'precision': 0.760089452881215, 'recall': 0.6928282244071717, 'ROC_AUC': 0.8722369496269949, 'PRC_AUC': 0.9721395097210769, 'specificity': 0.4307692307692308}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 'chosen_samples': [1080, 1097, 224, 247, 401, 1110, 199, 1036, 628, 1365, 315, 1099, 741, 351, 1284, 1230, 155, 304, 963, 426], 'chosen_samples_score': [0.01071508329111015, 0.020936701985194706, 0.030901122345277532, 0.040372666278019276, 0.04955941841781342, 0.05835468158385115, 0.06693923278484881, 0.07525449743741452, 0.08331557104362375, 0.09124155400720024, 0.09897698339152772, 0.10649581178001277, 0.11389818533109253, 0.12116976818988778, 0.1280729718801359, 0.134858353948486, 0.145900795562957, 0.16476704338296422, 0.1500160096028189, 0.1545308586627776], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 280.4590677060187, 'batch_acquisition_elapsed_time': 70.75415875995532})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 0.621148471174569, 'f1': 0.834698966868543, 'precision': 0.8440385632347558, 'recall': 0.8261037208405629, 'ROC_AUC': 0.9516447819077836, 'PRC_AUC': 0.9866503218784392, 'specificity': 0.6923076923076923}, 'chosen_targets': [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], 'chosen_samples': [506, 769, 1167, 586, 906, 1173, 1107, 1415, 1434, 484, 626, 1433, 429, 427, 1182, 882, 1423, 1236, 897, 608], 'chosen_samples_score': [0.03314754643983886, 0.0588636435719323, 0.08105327825118858, 0.10250487480679338, 0.1229779065797727, 0.14256190863102347, 0.16121192089268765, 0.1793853913481085, 0.19693424583158325, 0.21410501909371416, 0.23082294150829252, 0.24696790158378334, 0.2625590695793978, 0.27792216358218713, 0.29269092012968745, 0.3068274729409932, 0.31809089714563576, 0.3385292643064961, 0.35280178918056926, 0.36418619627915305], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 479.2714581582695, 'batch_acquisition_elapsed_time': 70.35649997601286})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.8857758620689655, 'nll': 0.8648876321726832, 'f1': 0.7644602373404081, 'precision': 0.762791228871631, 'recall': 0.7661654135338345, 'ROC_AUC': 0.8843073218995303, 'PRC_AUC': 0.9831495053648427, 'specificity': 0.6}, 'chosen_targets': [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 'chosen_samples': [282, 752, 679, 28, 1319, 1074, 1386, 785, 933, 1363, 812, 40, 602, 1250, 584, 1399, 637, 823, 540, 374], 'chosen_samples_score': [0.02270883456348105, 0.043355777886055824, 0.06321042600378224, 0.08176059239555977, 0.09977952668898471, 0.11681329470164714, 0.1332963536874181, 0.14925544890783726, 0.1646194307521398, 0.1794313934318872, 0.19402303835348, 0.20814894346138413, 0.22181751610535017, 0.23517996765572935, 0.2480866083584079, 0.26066166738159247, 0.2739559180813309, 0.2840340090669464, 0.3009024755866765, 0.29765003542443047], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.25779774505645, 'batch_acquisition_elapsed_time': 68.85777195682749})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8900862068965517, 'nll': 1.181495271880051, 'f1': 0.7306225596776212, 'precision': 0.7894944707740916, 'recall': 0.6978407557354926, 'ROC_AUC': 0.8190579220060666, 'PRC_AUC': 0.9728075084643399, 'specificity': 0.4307692307692308}, 'chosen_targets': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0], 'chosen_samples': [302, 171, 330, 5, 1343, 746, 321, 172, 1236, 1222, 828, 555, 446, 1224, 978, 1276, 1313, 114, 974, 1390], 'chosen_samples_score': [0.009972771435510563, 0.01948475651097692, 0.027986537330467964, 0.03612156867839933, 0.04411137970484713, 0.051770741731193226, 0.05924549133656676, 0.06658654742629144, 0.0738101047289641, 0.08091121080949915, 0.08790411816273203, 0.09477214973730241, 0.10152871294308419, 0.10810258890185853, 0.11457964967389245, 0.12096026525710535, 0.12588265524037645, 0.14229059572127767, 0.1515344872912241, 0.14540503235503266], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.03547971788794, 'batch_acquisition_elapsed_time': 68.13242319598794})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.9396551724137931, 'nll': 0.4170712109269767, 'f1': 0.8763514542409014, 'precision': 0.8716869055227641, 'recall': 0.881203007518797, 'ROC_AUC': 0.9461035163469643, 'PRC_AUC': 0.9934567399364922, 'specificity': 0.8}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1], 'chosen_samples': [710, 1263, 700, 697, 559, 1040, 928, 712, 1306, 1187, 1305, 383, 414, 143, 85, 705, 1112, 1311, 1333, 1267], 'chosen_samples_score': [0.03218189100914104, 0.06178086110771486, 0.08844082351228977, 0.11444765733300466, 0.1387430155989895, 0.1618177466534605, 0.18443355438793851, 0.20497149004763848, 0.22433430119939413, 0.24306201051248255, 0.2612629750525217, 0.27876187887831705, 0.2954262448247702, 0.31195127123529875, 0.32765088899299233, 0.3426036929006706, 0.35689797703858694, 0.3798786546814217, 0.3848868735230866, 0.4098712922793126], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 557.8292878931388, 'batch_acquisition_elapsed_time': 67.00818512635306})
