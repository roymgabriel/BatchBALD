store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT MULTI BALD (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 8888, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_multibald_8888', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.bald', 'acquisition_method': 'AcquisitionMethod.multibald', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_multibald_8888', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=bald', '--acquisition_method=multibald']
store['Distribution of training set classes:']={2: 802, 1: 588, 0: 233}
store['Distribution of validation set classes:']={2: 123, 1: 81, 0: 28}
store['Distribution of test set classes:']={2: 232, 1: 167, 0: 65}
store['Distribution of pool classes:']={2: 777, 1: 563, 0: 208}
store['Distribution of active set classes:']={1: 25, 0: 25, 2: 25}
store['active samples']=75
store['available samples']=1548
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[1445, 922, 879, 983, 1600, 592, 597, 1152, 714, 733, 1507, 109, 1155, 975, 884, 560, 535, 116, 865, 1035, 532, 1565, 611, 1059, 1563, 1377, 707, 493, 596, 606, 1084, 306, 951, 10, 821, 721, 670, 1498, 1429, 369, 934, 629, 1123, 224, 458, 243, 711, 148, 497, 1389, 264, 720, 1215, 1044, 812, 554, 814, 791, 1141, 19, 612, 605, 457, 1159, 832, 1484, 1502, 696, 1513, 1341, 1279, 1110, 579, 400, 1046]
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5926724137931034, 'nll': 1.4807288071204876, 'f1': 0.5227775320654579, 'precision': 0.5911854204537131, 'recall': 0.5122296521016323, 'ROC_AUC': 0.77606201171875, 'PRC_AUC': 0.6420719308821492, 'specificity': 0.7513594156062395}, 'chosen_targets': [2, 0, 1, 1, 0, 0, 2, 1, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 1], 'chosen_samples': [1084, 1332, 1214, 1274, 1354, 473, 1400, 336, 596, 491, 117, 999, 913, 1341, 697, 1025, 1070, 1315, 153, 897], 'chosen_samples_score': [0.11831749900988753, 0.1992626559197721, 0.27754503282614995, 0.34583926648504826, 0.4088330883478086, 0.4633320683321931, 0.5154555971583625, 0.5637163890524102, 0.6108594203831714, 0.656224474268841, 0.6982032500621829, 0.7412587387487859, 0.7827187955114843, 0.8322937333422278, 0.844281812578723, 0.8912753560557025, 0.9156480694051456, 0.9618268367162592, 0.9820199304381365, 1.012628550471483], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.05351857095957, 'batch_acquisition_elapsed_time': 87.30033273994923})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.5905172413793104, 'nll': 1.783123410981277, 'f1': 0.5068096857059158, 'precision': 0.5408984855446498, 'recall': 0.5179525564256103, 'ROC_AUC': 0.73968505859375, 'PRC_AUC': 0.579908286116024, 'specificity': 0.7582039971634648}, 'chosen_targets': [2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 1, 2, 1, 2, 2, 1, 0, 0, 0], 'chosen_samples': [848, 746, 323, 954, 1414, 294, 176, 988, 553, 254, 734, 1373, 767, 1294, 981, 1493, 870, 693, 978, 1413], 'chosen_samples_score': [0.1453557764137574, 0.2524041288366038, 0.3420182667108843, 0.41416575898436636, 0.48088052822258565, 0.5409070141578649, 0.5962207417640482, 0.6441988176342228, 0.690825548618407, 0.7353691557722399, 0.7775353778718461, 0.8195022590785461, 0.8526463982765016, 0.9112827596394037, 0.9411820653681708, 0.9695479319673286, 1.0070014459628478, 1.049828967257584, 1.0722903481747732, 1.1053118777997266], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 358.95660790381953, 'batch_acquisition_elapsed_time': 86.03698159800842})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.6012931034482759, 'nll': 1.404125082081762, 'f1': 0.5308404832031417, 'precision': 0.6152256311235071, 'recall': 0.5130539981045866, 'ROC_AUC': 0.79193115234375, 'PRC_AUC': 0.6726654098775615, 'specificity': 0.7572166803866743}, 'chosen_targets': [1, 0, 1, 2, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 0, 2, 2, 2, 0, 1], 'chosen_samples': [1053, 1239, 779, 291, 72, 916, 1152, 221, 1234, 1380, 1293, 1135, 97, 678, 78, 1392, 567, 1485, 290, 477], 'chosen_samples_score': [0.07278416312164226, 0.13865121375731326, 0.1967622500906654, 0.25002219685921867, 0.2981975386363578, 0.3437433126782752, 0.38779060563242673, 0.428207342979948, 0.4670906251512523, 0.504146216639942, 0.5420154122184044, 0.5798867354469461, 0.6072015309310448, 0.6445073855205088, 0.6688569152867707, 0.6966179706580089, 0.7085101607241455, 0.737859571673571, 0.7761021437715385, 0.8027363645797383], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 438.6528124436736, 'batch_acquisition_elapsed_time': 84.87273263698444})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5603448275862069, 'nll': 1.2856829413052262, 'f1': 0.3821840168737764, 'precision': 0.5802844118979594, 'recall': 0.4130477771247955, 'ROC_AUC': 0.75042724609375, 'PRC_AUC': 0.6270208155233651, 'specificity': 0.7181340454933921}, 'chosen_targets': [2, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 2, 0], 'chosen_samples': [1438, 277, 157, 347, 993, 983, 1273, 39, 1142, 268, 603, 1027, 842, 788, 875, 17, 837, 1197, 683, 1258], 'chosen_samples_score': [0.019632171368501594, 0.03538796385888243, 0.050003554280830365, 0.06444478565033052, 0.07857513721621112, 0.09224500517801282, 0.10569615796886467, 0.11831229690755674, 0.13082190638579494, 0.14297513174202692, 0.13766613428919428, 0.16575524186989732, 0.1805468461208637, 0.1994512335346812, 0.2127526614488584, 0.20584625156031322, 0.20739845140399993, 0.2432677387342892, 0.24044034340006526, 0.2456668036163201], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.9889257666655, 'batch_acquisition_elapsed_time': 83.71118910005316})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.625, 'nll': 1.0640309432457233, 'f1': 0.5801067849727234, 'precision': 0.5962059743989024, 'recall': 0.5908304081492188, 'ROC_AUC': 0.79644775390625, 'PRC_AUC': 0.6791837928555098, 'specificity': 0.7809881941007167}, 'chosen_targets': [2, 0, 0, 1, 1, 2, 0, 2, 0, 0, 2, 1, 0, 1, 2, 2, 1, 1, 2, 0], 'chosen_samples': [1341, 1272, 566, 817, 1444, 1250, 1086, 780, 437, 1335, 500, 1307, 710, 704, 369, 415, 638, 126, 305, 466], 'chosen_samples_score': [0.054978159209823074, 0.10231027156800598, 0.14680700913727263, 0.18929955643947505, 0.22837722227069834, 0.26643993402813715, 0.30311381283493244, 0.33713865607729954, 0.3706096658475877, 0.4023883575287286, 0.4193940098333009, 0.46378241333019954, 0.5126295165508257, 0.5100584232993768, 0.549013986155277, 0.5939128640058762, 0.6123524792193038, 0.6140522898298286, 0.6547301604459204, 0.6797660626673903], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.4386219359003, 'batch_acquisition_elapsed_time': 82.65190112404525})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5581896551724138, 'nll': 0.7555518314756197, 'f1': 0.42502183059458604, 'precision': 0.5289798570500325, 'recall': 0.443156128062178, 'ROC_AUC': 0.7391357421875, 'PRC_AUC': 0.5910830342814887, 'specificity': 0.7226557919237894}, 'chosen_targets': [1, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 0, 1], 'chosen_samples': [1218, 119, 667, 1322, 747, 1022, 509, 490, 1069, 1441, 537, 1135, 1407, 7, 1268, 1395, 232, 920, 877, 902], 'chosen_samples_score': [0.06690218620357724, 0.12032037136062423, 0.16766597802528538, 0.21015399854934413, 0.24940530718210407, 0.28294922137688605, 0.3139894902321938, 0.3434529051316346, 0.37083354500279064, 0.39645798883987293, 0.4249188382476401, 0.45050775555695743, 0.4621727587040869, 0.49597012784613526, 0.5195749800150224, 0.5156538501081815, 0.5594274688084866, 0.5678878696941592, 0.5575986581978558, 0.6067725461491413], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.73168603796512, 'batch_acquisition_elapsed_time': 81.72093433188275})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6767241379310345, 'nll': 0.7658371760927397, 'f1': 0.6563694828446979, 'precision': 0.6893421052631578, 'recall': 0.6369440164763311, 'ROC_AUC': 0.8394775390625, 'PRC_AUC': 0.7069241367807576, 'specificity': 0.810593439071963}, 'chosen_targets': [2, 0, 2, 0, 2, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 0, 0, 2, 1], 'chosen_samples': [795, 426, 346, 541, 519, 641, 1168, 332, 1135, 972, 744, 306, 640, 1378, 1397, 1042, 1112, 963, 723, 879], 'chosen_samples_score': [0.031190502517369323, 0.06166508254605929, 0.08959312118645268, 0.11647731647360837, 0.1417654148783134, 0.16619110888435262, 0.18918955030687457, 0.21130521197901242, 0.23255908722793261, 0.25347075033798205, 0.2747527179686333, 0.28600079809781676, 0.3121772767219735, 0.32557017864130167, 0.3552076650839826, 0.3791404932435398, 0.3802550953078292, 0.4063879296003279, 0.40496682438431364, 0.4396393924646258], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.5729133207351, 'batch_acquisition_elapsed_time': 80.65007809270173})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6616379310344828, 'nll': 0.8761504600787985, 'f1': 0.6522732518705543, 'precision': 0.6574379501829798, 'recall': 0.6475805683063581, 'ROC_AUC': 0.835693359375, 'PRC_AUC': 0.7259344212399756, 'specificity': 0.8097491057261172}, 'chosen_targets': [2, 1, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1], 'chosen_samples': [864, 826, 796, 758, 1356, 1133, 1314, 179, 895, 241, 1249, 159, 497, 789, 600, 1213, 365, 1236, 83, 1060], 'chosen_samples_score': [0.030429092739927555, 0.05699169318215391, 0.08142391742604582, 0.10438655875231628, 0.1264881648846341, 0.14814419439019932, 0.16904135686235833, 0.18928950142625123, 0.2092630069012147, 0.22862793189380248, 0.2527271745573163, 0.25470689926098267, 0.2737748571767291, 0.2915164947977047, 0.31434390495118514, 0.338994357040443, 0.3612105116638933, 0.37169791887357917, 0.37796682590519737, 0.4053484734723547], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.96073209820315, 'batch_acquisition_elapsed_time': 79.47375798691064})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6206896551724138, 'nll': 0.9726649317248114, 'f1': 0.592544067094059, 'precision': 0.5826198830409356, 'recall': 0.6330363146386272, 'ROC_AUC': 0.79339599609375, 'PRC_AUC': 0.6701827888719243, 'specificity': 0.7935456748221418}, 'chosen_targets': [0, 2, 1, 2, 1, 0, 2, 1, 1, 2, 0, 2, 2, 2, 1, 2, 1, 2, 1, 2], 'chosen_samples': [954, 276, 1191, 1122, 176, 1354, 466, 964, 186, 512, 599, 970, 636, 1312, 453, 395, 461, 597, 1163, 168], 'chosen_samples_score': [0.03154787364490186, 0.06159037801870404, 0.08473923779021497, 0.10667595954806108, 0.1276735258286794, 0.14793115616413566, 0.16788339882603065, 0.1868222237145849, 0.20530338445649576, 0.2228755023802922, 0.2350439596088041, 0.2324760343928869, 0.26012113336073917, 0.29968477836346885, 0.2935119412752911, 0.30336769309482925, 0.338252470366724, 0.3452285397260759, 0.3577919315778999, 0.36505268297562665], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.6342369168997, 'batch_acquisition_elapsed_time': 78.59490653080866})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.6551724137931034, 'nll': 1.0435431250210465, 'f1': 0.6409569333607749, 'precision': 0.6833464475815783, 'recall': 0.6250389142140124, 'ROC_AUC': 0.8250732421875, 'PRC_AUC': 0.715433666256324, 'specificity': 0.7912495376953936}, 'chosen_targets': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0], 'chosen_samples': [545, 1130, 970, 1322, 59, 757, 797, 145, 1109, 410, 933, 167, 1214, 1167, 77, 642, 700, 967, 1328, 343], 'chosen_samples_score': [0.04672712350726271, 0.0879877396672123, 0.1264515650739637, 0.16211838535932221, 0.1955386509865087, 0.22701490218273035, 0.25700734090201216, 0.28499534469865395, 0.3113547613084293, 0.3370889248352835, 0.35535827305818035, 0.38182226845875, 0.4007494051301226, 0.4242253029627925, 0.43440623584307403, 0.4867731791632295, 0.4878016454107996, 0.5159313955657581, 0.5127696900757588, 0.5653235402003034], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 559.195391387213, 'batch_acquisition_elapsed_time': 77.55568638304248})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5711206896551724, 'nll': 0.6965983818317282, 'f1': 0.48291583214829287, 'precision': 0.5047809256990194, 'recall': 0.542161565463238, 'ROC_AUC': 0.77728271484375, 'PRC_AUC': 0.6674538564144042, 'specificity': 0.7475131940497941}, 'chosen_targets': [1, 0, 1, 2, 1, 2, 2, 2, 1, 0, 1, 1, 2, 2, 1, 0, 2, 0, 1, 1], 'chosen_samples': [41, 502, 256, 400, 107, 1227, 562, 410, 901, 1221, 87, 223, 248, 676, 862, 917, 1260, 583, 731, 156], 'chosen_samples_score': [0.045824922658768086, 0.06249603948737181, 0.07511799421242737, 0.08729226560248549, 0.09916694672963011, 0.11071197033521862, 0.1220171827720904, 0.13292034054269486, 0.1436130755961429, 0.15408276919017716, 0.16919245274134553, 0.17451818864652324, 0.19069767383995107, 0.2024649707598769, 0.20079056986254074, 0.19727095827347974, 0.2337248083560155, 0.21925915972840748, 0.25099043085138106, 0.24013049050160973], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.46925866836682, 'batch_acquisition_elapsed_time': 76.65146931633353})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.646551724137931, 'nll': 0.6408466470652613, 'f1': 0.6025041249032669, 'precision': 0.6200299819696603, 'recall': 0.6310381888742409, 'ROC_AUC': 0.81353759765625, 'PRC_AUC': 0.7061253032110487, 'specificity': 0.7966804346870893}, 'chosen_targets': [1, 0, 1, 2, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 2, 1], 'chosen_samples': [959, 684, 432, 861, 629, 1157, 808, 559, 348, 964, 1077, 718, 830, 831, 421, 1059, 1154, 316, 1316, 1082], 'chosen_samples_score': [0.03717118835560829, 0.06912292905416095, 0.09337898720517201, 0.11167484764519342, 0.12893052736412125, 0.14558256586798723, 0.16211050645552483, 0.1782176984402133, 0.19396407091456958, 0.20963571097500555, 0.22674106583258258, 0.23655615458213575, 0.2510088514816804, 0.25585290484407075, 0.2942227806676687, 0.2937997672707624, 0.3090405826634264, 0.33002936553121387, 0.329801083201513, 0.3300227247069305], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.32213335996494, 'batch_acquisition_elapsed_time': 75.38585226796567})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6422413793103449, 'nll': 0.5932136404103246, 'f1': 0.6181513221689335, 'precision': 0.6725896894784443, 'recall': 0.6085890553111284, 'ROC_AUC': 0.82275390625, 'PRC_AUC': 0.6783010218874557, 'specificity': 0.7797988236584729}, 'chosen_targets': [0, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], 'chosen_samples': [1216, 209, 774, 612, 410, 364, 849, 471, 423, 1124, 1270, 269, 405, 935, 983, 810, 830, 362, 1211, 378], 'chosen_samples_score': [0.01892973609416515, 0.033677924622442346, 0.04424885865726069, 0.053999083748682786, 0.06347431237664969, 0.07267957188171481, 0.08172091663542957, 0.0903882272796519, 0.09887051010232906, 0.1072641916465118, 0.11037118142200875, 0.1349311088066294, 0.13225304450705444, 0.1471824207700898, 0.1343870490186525, 0.1531869495441729, 0.16886232585810212, 0.18680214762086678, 0.17027450516110676, 0.17399260786114645], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.5911635751836, 'batch_acquisition_elapsed_time': 74.2924165003933})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5366379310344828, 'nll': 0.8635734689646754, 'f1': 0.4497791909556616, 'precision': 0.5375660917049493, 'recall': 0.4389646436569831, 'ROC_AUC': 0.731201171875, 'PRC_AUC': 0.5498587128939754, 'specificity': 0.7273475471448854}, 'chosen_targets': [1, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2], 'chosen_samples': [402, 1239, 398, 730, 979, 511, 688, 961, 542, 190, 293, 318, 430, 1131, 858, 1024, 218, 448, 681, 384], 'chosen_samples_score': [0.008710702790314162, 0.01681898361381129, 0.024808520806166356, 0.03258493364883419, 0.040146138552095145, 0.0475195306202334, 0.05470273337955067, 0.06180107043298477, 0.06881099189073048, 0.07573832040245243, 0.08953818026905225, 0.0823485454376165, 0.09328542266193729, 0.11243402332770103, 0.11467751323319142, 0.11029016730358876, 0.1549317486843762, 0.10917811544941003, 0.12851660061791037, 0.14085122830419294], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.85229466017336, 'batch_acquisition_elapsed_time': 73.34256576001644})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6530172413793104, 'nll': 0.6457478753451643, 'f1': 0.622801223696191, 'precision': 0.6215646417673267, 'recall': 0.6317673671225189, 'ROC_AUC': 0.81829833984375, 'PRC_AUC': 0.6869633147011269, 'specificity': 0.8046424403048723}, 'chosen_targets': [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 0, 1, 2, 2, 2, 1, 1, 1], 'chosen_samples': [21, 567, 484, 274, 519, 1090, 698, 906, 84, 25, 646, 952, 1199, 688, 513, 1218, 1055, 58, 129, 331], 'chosen_samples_score': [0.010609141317221016, 0.020713858227656035, 0.030559426276538515, 0.04015689881243567, 0.049599698156657546, 0.058788177282366405, 0.06780731184229882, 0.07667908417966451, 0.08533928482844022, 0.09384850931069977, 0.10742355137648119, 0.11084839215085651, 0.1303824542150256, 0.1301008101512071, 0.14710429079023513, 0.14037622690087836, 0.13908950876940906, 0.1541292914865906, 0.16431260124338465, 0.16373877249178292], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.8133013783954, 'batch_acquisition_elapsed_time': 72.28584855888039})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6314655172413793, 'nll': 0.6084952518857759, 'f1': 0.5963778533946938, 'precision': 0.6004122591160398, 'recall': 0.6292588562927196, 'ROC_AUC': 0.8326416015625, 'PRC_AUC': 0.7001847566717759, 'specificity': 0.7879633557945719}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1], 'chosen_samples': [399, 1223, 636, 980, 560, 511, 1151, 408, 540, 216, 792, 1220, 1039, 5, 988, 1135, 293, 831, 183, 1155], 'chosen_samples_score': [0.02024065687581178, 0.03729078723112644, 0.05222053178774422, 0.06586405275465879, 0.0788938971433839, 0.09076985364651424, 0.10201746017728386, 0.11300227819149988, 0.12365022507919043, 0.13399865861412152, 0.14998056908696178, 0.1557173554485125, 0.15045921026826115, 0.16646252331152134, 0.17075583829734242, 0.19769439306242909, 0.18935660517014163, 0.19709122758907682, 0.21306902478191247, 0.22033948826084426], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.73026972822845, 'batch_acquisition_elapsed_time': 71.29493256099522})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6810344827586207, 'nll': 0.771093697383486, 'f1': 0.6724117071102048, 'precision': 0.7082619003490563, 'recall': 0.6571535973146545, 'ROC_AUC': 0.84771728515625, 'PRC_AUC': 0.7486296451842636, 'specificity': 0.8251015469708755}, 'chosen_targets': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1, 1, 0, 0, 0, 1], 'chosen_samples': [1221, 1132, 315, 1158, 215, 98, 1201, 1151, 823, 768, 795, 871, 746, 204, 1007, 894, 850, 78, 512, 948], 'chosen_samples_score': [0.0320064254861514, 0.05792880241112386, 0.08244386751861543, 0.10450144659032268, 0.12527829503587862, 0.1448683588390116, 0.16357147211732936, 0.18131400382147422, 0.19844181800515415, 0.21487003648042702, 0.23654150624181547, 0.2454145025382104, 0.259205616991558, 0.2705435882681275, 0.2904436918325395, 0.2923462281482365, 0.31344758293166564, 0.31164469840752673, 0.3278193351535297, 0.34679748094116825], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.7417635368183, 'batch_acquisition_elapsed_time': 69.95218865992501})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6616379310344828, 'nll': 0.6837973101385708, 'f1': 0.655406884623293, 'precision': 0.6592499414946938, 'recall': 0.6518312182002043, 'ROC_AUC': 0.8489990234375, 'PRC_AUC': 0.7472283021154338, 'specificity': 0.8097491057261172}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], 'chosen_samples': [244, 985, 864, 307, 1058, 714, 289, 755, 675, 763, 632, 756, 855, 939, 998, 782, 68, 1130, 524, 1165], 'chosen_samples_score': [0.03282043674615509, 0.06148541922277717, 0.08739031447028878, 0.11193379936165027, 0.1350374730636239, 0.1570501474450281, 0.17819426219840562, 0.1983965341640097, 0.21723634587836926, 0.23525174931171033, 0.2562965778068982, 0.2752527056903169, 0.27382182061948157, 0.29132467865390055, 0.3248206748716971, 0.31447103143286803, 0.3387449876262423, 0.3525481483526498, 0.35423311122800527, 0.3781873695154445], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.2057893751189, 'batch_acquisition_elapsed_time': 69.13278236519545})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.6939655172413793, 'nll': 0.8929513733962486, 'f1': 0.6844606824790916, 'precision': 0.6709427306911584, 'recall': 0.7040327832398864, 'ROC_AUC': 0.8475341796875, 'PRC_AUC': 0.7506393051281006, 'specificity': 0.8341424209542782}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 2, 1, 1, 2, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1], 'chosen_samples': [268, 1065, 549, 1089, 1148, 105, 744, 355, 511, 1060, 88, 371, 288, 559, 100, 484, 331, 894, 1125, 199], 'chosen_samples_score': [0.03499668466522743, 0.0629005920990362, 0.08941188511585607, 0.11460123686422197, 0.13779000756763127, 0.16055540912746435, 0.18264810556350675, 0.20393883293614046, 0.22493514602037745, 0.24495942900395296, 0.27399543203497956, 0.2786215971852588, 0.30337942613438784, 0.3145366333077444, 0.3344219820167922, 0.35531447246101067, 0.37084157080823843, 0.36899514370093023, 0.3976057972017646, 0.4073058415963313], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 478.94258398376405, 'batch_acquisition_elapsed_time': 67.83796766167507})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 0.7347626521669585, 'f1': 0.5971295796132984, 'precision': 0.5910657630522088, 'recall': 0.6168580610661966, 'ROC_AUC': 0.800537109375, 'PRC_AUC': 0.6705249534841453, 'specificity': 0.7945369926616146}, 'chosen_targets': [1, 2, 0, 2, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 'chosen_samples': [473, 895, 875, 121, 1012, 962, 932, 897, 399, 173, 986, 820, 440, 808, 497, 99, 343, 579, 972, 248], 'chosen_samples_score': [0.02485300940443269, 0.04846085196228911, 0.06495190896778702, 0.08086325160832963, 0.09607510359470917, 0.11036140767261315, 0.12436065081737624, 0.1381198943465609, 0.15140757197958443, 0.16415622190081702, 0.17922126781931524, 0.19820330876626713, 0.19831239370361686, 0.20658114827270424, 0.21595975090749775, 0.2446021425286169, 0.2414724474789285, 0.2633038903117271, 0.26835446029372534, 0.2662926773978622], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.06398423667997, 'batch_acquisition_elapsed_time': 66.8885724209249})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5926724137931034, 'nll': 0.6414250340955011, 'f1': 0.48365092181544966, 'precision': 0.6226026753702741, 'recall': 0.47553857801638105, 'ROC_AUC': 0.78204345703125, 'PRC_AUC': 0.6571951506603693, 'specificity': 0.7509783325724039}, 'chosen_targets': [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 1, 2], 'chosen_samples': [349, 302, 644, 300, 190, 171, 205, 495, 453, 237, 639, 478, 823, 778, 906, 933, 347, 618, 68, 140], 'chosen_samples_score': [0.00820281312019111, 0.016006760354462357, 0.023745782978384478, 0.03138204325449978, 0.03884930775618711, 0.04617464843645536, 0.05324125419404435, 0.06023370304468845, 0.06714421846506369, 0.07400422211972568, 0.0741542883319326, 0.09496707532602322, 0.09874759090265783, 0.09531653490332914, 0.09492677279766504, 0.11070499863842187, 0.11901576916284995, 0.1309298914257404, 0.1308227173793668, 0.13043862818876661], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.8319673272781, 'batch_acquisition_elapsed_time': 65.99824999179691})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.709051724137931, 'nll': 0.7492364686110924, 'f1': 0.7203841001873492, 'precision': 0.7051193592960256, 'recall': 0.7435087384911873, 'ROC_AUC': 0.86138916015625, 'PRC_AUC': 0.7585310867621387, 'specificity': 0.8418770570918181}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2, 0, 1, 1, 1], 'chosen_samples': [839, 168, 676, 461, 199, 1095, 219, 269, 860, 866, 468, 931, 905, 899, 1105, 624, 977, 726, 522, 744], 'chosen_samples_score': [0.04899704683746686, 0.08571626682571687, 0.11553640539714904, 0.14286061541262218, 0.16660948301653855, 0.1891712172164226, 0.21043785914447177, 0.2305466617604699, 0.24928028239740474, 0.2670567517788456, 0.281036188154685, 0.28671185728442694, 0.31501705530190627, 0.3292377731239693, 0.3445933717886529, 0.3381271420828753, 0.37910109992362173, 0.3932970520585073, 0.39090102047976494, 0.40380155856706423], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.44597406405956, 'batch_acquisition_elapsed_time': 64.49164600204676})
