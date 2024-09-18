store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT VARIATIONAL RATIOS (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 1970, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'local_rank': 0, 'experiment_task_id': 'covid_full_resnet_binary_scratch_vr_1970', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.variation_ratios', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_vr_1970', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=variation_ratios', '--acquisition_method=independent']
store['Distribution of training set classes:']={1: 1389, 0: 234}
store['Distribution of validation set classes:']={1: 205, 0: 27}
store['Distribution of test set classes:']={1: 399, 0: 65}
store['Distribution of pool classes:']={1: 1364, 0: 209}
store['Distribution of active set classes:']={0: 25, 1: 25}
store['active samples']=50
store['available samples']=1573
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[334, 101, 1408, 1336, 640, 386, 20, 888, 1505, 1400, 1587, 361, 744, 1311, 649, 1045, 1200, 1322, 602, 1307, 825, 557, 647, 502, 942, 1345, 564, 1146, 1608, 1365, 1607, 1394, 875, 248, 451, 165, 736, 1176, 672, 1512, 1252, 417, 727, 1369, 1129, 1065, 1579, 900, 9, 1254]
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.7133620689655172, 'nll': 1.055647948692585, 'f1': 0.6121839788345158, 'precision': 0.6152604038917505, 'recall': 0.7174281858492384, 'ROC_AUC': 0.8025959483601969, 'PRC_AUC': 0.9549723830555874, 'specificity': 0.7174281858492384}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], 'chosen_samples': [919, 134, 1332, 505, 1338, 57, 296, 290, 1472, 1133, 1507, 689, 562, 1461, 337, 1153, 870, 809, 377, 821], 'chosen_samples_score': [0.49906359574391745, 0.4974805031971514, 0.49503855299774513, 0.49421000328377196, 0.4933742947996167, 0.49293681937349865, 0.49137359307940953, 0.49128023832902235, 0.4907668706815673, 0.48966889039395467, 0.4862992608413005, 0.4859171704506101, 0.4859132683827151, 0.4857123647995062, 0.4842577362053969, 0.48361391063699477, 0.4804954853455935, 0.4794334749071414, 0.4787352764944879, 0.4779345547997266], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 165.558335585054, 'batch_acquisition_elapsed_time': 45.228208641987294})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8448275862068966, 'nll': 3.098975082923626, 'f1': 0.5965217391304347, 'precision': 0.6405594405594406, 'recall': 0.5813765182186235, 'ROC_AUC': 0.6511698805262165, 'PRC_AUC': 0.9305234978989816, 'specificity': 0.5813765182186235}, 'chosen_targets': [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1], 'chosen_samples': [35, 988, 1082, 178, 238, 40, 1352, 659, 1239, 1138, 1496, 722, 754, 310, 1172, 10, 557, 1282, 180, 584], 'chosen_samples_score': [0.49653361113614713, 0.49548489419830255, 0.4919473669141019, 0.49168128714859904, 0.4914745004933204, 0.4862321976631816, 0.478719598678377, 0.47780279201322273, 0.47732456794879696, 0.47292993694262775, 0.4723625277799143, 0.47040283002793504, 0.4693177331283751, 0.46258300099211014, 0.45927220974831684, 0.4554204463606516, 0.455095071681771, 0.4503861436445654, 0.4503427795454913, 0.4502774747165159], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 118.90770508814603, 'batch_acquisition_elapsed_time': 44.51404258189723})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8469827586206896, 'nll': 0.9129109218202788, 'f1': 0.716173721256817, 'precision': 0.6979323308270677, 'recall': 0.7436090225563909, 'ROC_AUC': 0.8249502380083107, 'PRC_AUC': 0.9546698916624752, 'specificity': 0.7436090225563909}, 'chosen_targets': [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1], 'chosen_samples': [68, 75, 1027, 51, 1151, 216, 1125, 1269, 1127, 651, 1019, 1157, 1022, 970, 1051, 138, 672, 141, 354, 393], 'chosen_samples_score': [0.49955447541393017, 0.4983544270988194, 0.49810035955496856, 0.4978755631730696, 0.4976428846693548, 0.49746132192932746, 0.49744515823823365, 0.49719603164359605, 0.4960000158058321, 0.49474254312946975, 0.49157370380498033, 0.487792809369868, 0.48723373211674303, 0.48705028223437397, 0.4865732750307574, 0.4857889361027491, 0.48530513393381536, 0.4852819640670384, 0.48282946502566315, 0.48233281659731553], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 142.10381297161803, 'batch_acquisition_elapsed_time': 44.062455328181386})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8922413793103449, 'nll': 2.363665087469693, 'f1': 0.7380307136404698, 'precision': 0.7944539579075292, 'recall': 0.7055330634278003, 'ROC_AUC': 0.8357542877723045, 'PRC_AUC': 0.9691418405745773, 'specificity': 0.7055330634278003}, 'chosen_targets': [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], 'chosen_samples': [808, 1436, 743, 570, 630, 1485, 323, 1367, 263, 532, 1358, 11, 56, 738, 622, 349, 190, 562, 610, 776], 'chosen_samples_score': [0.48979353265994807, 0.48561910676217324, 0.4838881046650224, 0.4817055494419561, 0.4683508501425321, 0.45044957637261407, 0.446147701122907, 0.43915376128763906, 0.4378590101179365, 0.41790710595218017, 0.4048953454099472, 0.3984173547818959, 0.39220415611647375, 0.3783652005609708, 0.37738925040383275, 0.3758081879307005, 0.3725570660811929, 0.37207945621231475, 0.3690227295801134, 0.3676968261748136], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 165.81921268627048, 'batch_acquisition_elapsed_time': 43.6088080429472})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8771551724137931, 'nll': 1.6343998086863551, 'f1': 0.6228986953732089, 'precision': 0.8028151469855506, 'recall': 0.593734335839599, 'ROC_AUC': 0.8169865819677258, 'PRC_AUC': 0.966190717466116, 'specificity': 0.593734335839599}, 'chosen_targets': [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], 'chosen_samples': [1207, 1341, 140, 603, 908, 1075, 1433, 781, 309, 765, 469, 979, 267, 379, 263, 337, 717, 678, 452, 982], 'chosen_samples_score': [0.4968592612010304, 0.49164960273875136, 0.48942661253051667, 0.47898231087242316, 0.47536375601241854, 0.4711936970174019, 0.46970367187280215, 0.4687281996320398, 0.467656492689974, 0.46205407805077714, 0.4577905158784332, 0.4531312910628311, 0.4357323153527629, 0.4350597848396567, 0.42772136923746373, 0.42716740881828785, 0.42709950028133237, 0.4217804261277649, 0.4164448668074021, 0.40389619683666855], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 96.31414354965091, 'batch_acquisition_elapsed_time': 42.92485047224909})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.896551724137931, 'nll': 0.43123722076416016, 'f1': 0.8131480922178597, 'precision': 0.781752808988764, 'recall': 0.8625795257374205, 'ROC_AUC': 0.8827746074975291, 'PRC_AUC': 0.9831103218906994, 'specificity': 0.8625795257374205}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], 'chosen_samples': [1469, 1174, 1051, 462, 1436, 376, 619, 969, 819, 184, 54, 729, 867, 887, 180, 267, 266, 1446, 1055, 1145], 'chosen_samples_score': [0.49593097020282284, 0.4940616598898555, 0.49073317525329174, 0.48802032694298725, 0.4839333590671826, 0.48310024561547027, 0.47506098546475206, 0.4746621818407062, 0.47413938532420485, 0.47135022024957585, 0.4704846068427576, 0.4701040720871933, 0.46689082700839624, 0.46553165326384893, 0.46460507402222584, 0.46336773465121983, 0.46331705348357066, 0.46280631463915256, 0.4606528773236819, 0.45766973568198943], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 142.82566600106657, 'batch_acquisition_elapsed_time': 42.54044064693153})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.8771551724137931, 'nll': 0.6069631905391298, 'f1': 0.7171185624899727, 'precision': 0.7465217391304347, 'recall': 0.6967611336032389, 'ROC_AUC': 0.7980586956279431, 'PRC_AUC': 0.970171263156135, 'specificity': 0.6967611336032389}, 'chosen_targets': [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], 'chosen_samples': [12, 679, 1237, 947, 324, 232, 436, 492, 797, 149, 97, 47, 323, 60, 1359, 176, 724, 580, 310, 84], 'chosen_samples_score': [0.4999550091930337, 0.4996736042363308, 0.4976164437248366, 0.496598167904307, 0.4964223408618046, 0.49626754163034015, 0.4959136342610755, 0.4954892769568199, 0.4950162665554443, 0.4931554756860963, 0.49015789213515315, 0.4897104476119205, 0.4870116621898305, 0.4848351816925808, 0.4819227059019098, 0.48142578606762454, 0.480441096366939, 0.48020176507202905, 0.4794322708544142, 0.4777001686199931], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 95.8655937477015, 'batch_acquisition_elapsed_time': 41.76749154087156})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8857758620689655, 'nll': 0.4722534048146215, 'f1': 0.7644602373404081, 'precision': 0.762791228871631, 'recall': 0.7661654135338345, 'ROC_AUC': 0.8800988308702169, 'PRC_AUC': 0.9847249568526268, 'specificity': 0.7661654135338345}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1], 'chosen_samples': [238, 394, 1396, 226, 1212, 41, 816, 783, 282, 256, 563, 414, 659, 49, 744, 1161, 184, 1134, 149, 666], 'chosen_samples_score': [0.4988550036359297, 0.49329024603751415, 0.4930870041367139, 0.4928644894880324, 0.49256365560120463, 0.4910827589535818, 0.4843982093276453, 0.4843915187554406, 0.483838707529725, 0.47785618122810514, 0.4777021742569665, 0.47732611802080116, 0.47730353063233255, 0.47634368577561215, 0.47572198199412685, 0.4733878582104134, 0.4720854234765697, 0.4719659566753758, 0.4705567818134805, 0.4701999327374964], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 119.52671724697575, 'batch_acquisition_elapsed_time': 41.23107509408146})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.8728448275862069, 'nll': 0.736026566604088, 'f1': 0.6832031476016894, 'precision': 0.7379716981132075, 'recall': 0.6556198187777135, 'ROC_AUC': 0.8220780590270056, 'PRC_AUC': 0.9633175871808651, 'specificity': 0.6556198187777135}, 'chosen_targets': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], 'chosen_samples': [1264, 825, 1283, 922, 648, 266, 237, 259, 1077, 924, 484, 44, 643, 513, 586, 731, 278, 165, 275, 1141], 'chosen_samples_score': [0.49388930472427517, 0.49045679738481285, 0.48984764706237294, 0.48796025307857804, 0.4876931958177657, 0.48767649439448557, 0.4866592411266685, 0.4861403857541232, 0.4859777582044965, 0.4846424756048422, 0.4842851016563443, 0.48046186397539803, 0.4789755155947396, 0.47450651897462015, 0.4674494793172911, 0.46587548758308783, 0.4635242460935953, 0.4593479138740677, 0.4576864633825891, 0.4559551552684885], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 119.44399729417637, 'batch_acquisition_elapsed_time': 40.68116643792018})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9137931034482759, 'nll': 0.5181190227640087, 'f1': 0.7904245709123758, 'precision': 0.8585317350715351, 'recall': 0.7502602660497397, 'ROC_AUC': 0.8436222705389874, 'PRC_AUC': 0.9820766467905174, 'specificity': 0.7502602660497397}, 'chosen_targets': [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'chosen_samples': [507, 1389, 761, 581, 470, 877, 428, 678, 1138, 187, 891, 711, 870, 257, 460, 1322, 1158, 841, 421, 247], 'chosen_samples_score': [0.4991767128136757, 0.498938194323315, 0.4962735856775835, 0.49531539199030583, 0.4948848883921483, 0.4938251519523582, 0.49360720675400593, 0.49279331093608314, 0.49003323374292684, 0.48282116287714405, 0.48139034075349196, 0.4813398892176971, 0.48117199137708644, 0.47969945457231, 0.4759744966193874, 0.47054360591853783, 0.47044241846215407, 0.46823786590340866, 0.465632360575035, 0.46362508273414993], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 189.81279155937955, 'batch_acquisition_elapsed_time': 40.139582100789994})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.875, 'nll': 1.1756255051185345, 'f1': 0.5940874811463046, 'precision': 0.8236397748592871, 'recall': 0.5731636784268364, 'ROC_AUC': 0.7446318590657918, 'PRC_AUC': 0.9274902192549956, 'specificity': 0.5731636784268364}, 'chosen_targets': [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], 'chosen_samples': [345, 399, 748, 322, 98, 278, 642, 311, 1278, 743, 937, 974, 490, 348, 1020, 1188, 420, 1209, 455, 299], 'chosen_samples_score': [0.49983837855244007, 0.49978855618964724, 0.49864102447545555, 0.496379626832689, 0.4959836034139836, 0.4934566426961987, 0.49247222028454263, 0.4922351355846578, 0.48777379061285264, 0.4857039303598384, 0.48373396136768154, 0.48262565576496963, 0.48254137667727937, 0.48114045476579015, 0.47802214713252233, 0.4707323733564265, 0.46969104497681335, 0.4680136848446109, 0.4666982679611342, 0.4647671172405613], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 119.57699475530535, 'batch_acquisition_elapsed_time': 39.45808161981404})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9331896551724138, 'nll': 0.8421555880842537, 'f1': 0.8277922109018641, 'precision': 0.9510643821391485, 'recall': 0.767977636398689, 'ROC_AUC': 0.9003364602780299, 'PRC_AUC': 0.9854250216312255, 'specificity': 0.767977636398689}, 'chosen_targets': [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 'chosen_samples': [436, 419, 560, 392, 321, 490, 93, 323, 724, 538, 1069, 730, 65, 1097, 822, 1110, 498, 1125, 1281, 1203], 'chosen_samples_score': [0.4980827207984758, 0.49696876627823905, 0.49634985329397585, 0.49510845835516615, 0.4942521053811636, 0.49188061375346537, 0.48816005804239393, 0.4714990953451921, 0.47021772048216937, 0.46958888000229604, 0.4676773860611203, 0.46158253725398035, 0.4515318407964479, 0.44063439281760974, 0.4321589666353839, 0.4278567765843694, 0.426831291490821, 0.4180214110354208, 0.4084030817462888, 0.4007827405743417], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 213.12874815007672, 'batch_acquisition_elapsed_time': 38.92132775019854})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.8642241379310345, 'nll': 0.44082020068990774, 'f1': 0.7560238701331219, 'precision': 0.730154486036839, 'recall': 0.798708309234625, 'ROC_AUC': 0.8661374917387887, 'PRC_AUC': 0.9707954963621447, 'specificity': 0.798708309234625}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1260, 688, 391, 563, 104, 900, 1014, 954, 388, 1119, 310, 875, 96, 208, 277, 1276, 79, 923, 658, 374], 'chosen_samples_score': [0.4984072783860606, 0.4979769395615319, 0.4949443039739747, 0.4942160046318118, 0.4923195537628645, 0.49169165592593056, 0.4898124594425103, 0.48956091147655667, 0.489264872766655, 0.48796376090664995, 0.4878922893086042, 0.48673210482253515, 0.4859342467801805, 0.48576031357296867, 0.4842921316861756, 0.4835407332443553, 0.48326003669401896, 0.4829604832779808, 0.48293612296863786, 0.4821191082368925], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 166.3877693079412, 'batch_acquisition_elapsed_time': 38.59641724778339})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 1.6754784419618804, 'f1': 0.7238095238095238, 'precision': 0.9295080350980972, 'recall': 0.667977636398689, 'ROC_AUC': 0.8890537377847192, 'PRC_AUC': 0.989856687886245, 'specificity': 0.667977636398689}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'chosen_samples': [757, 1148, 419, 591, 1300, 223, 991, 465, 736, 592, 164, 102, 649, 1268, 295, 1083, 911, 571, 866, 1044], 'chosen_samples_score': [0.49122395412438646, 0.48735196760095045, 0.4675660640268162, 0.442357453940035, 0.4262083247985503, 0.4252997109731541, 0.41311433658224406, 0.4070295504514061, 0.3993181257768993, 0.37997128822402004, 0.37324417419851696, 0.36586518755416375, 0.36142156932135017, 0.35963128119109666, 0.3534657945870925, 0.3447540365870462, 0.3433485343126057, 0.3394462825129957, 0.3326428999489508, 0.32699328874598155], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 213.39844607189298, 'batch_acquisition_elapsed_time': 37.85007409192622})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9116379310344828, 'nll': 0.5966938281881398, 'f1': 0.8154281999786555, 'precision': 0.8175, 'recall': 0.8133988818199345, 'ROC_AUC': 0.9183367770294993, 'PRC_AUC': 0.9888014156033929, 'specificity': 0.8133988818199345}, 'chosen_targets': [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], 'chosen_samples': [1263, 737, 132, 658, 1102, 77, 896, 377, 273, 2, 361, 772, 225, 358, 176, 1278, 205, 291, 1207, 1210], 'chosen_samples_score': [0.49698769671854126, 0.49678737930260375, 0.49376824963414934, 0.4916023060640592, 0.4912054235221721, 0.4817583002676732, 0.47801660577654725, 0.4762525042642859, 0.475695503314315, 0.46072401137451013, 0.45584202347653624, 0.45452299751297387, 0.45341126623453143, 0.45334674710163825, 0.44798164631606063, 0.4415651236821486, 0.4392201035571238, 0.4367720158623698, 0.4346651095506162, 0.43337819902715835], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 190.2473283293657, 'batch_acquisition_elapsed_time': 37.38187838811427})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 0.7605838117928341, 'f1': 0.7617625093353249, 'precision': 0.8434389140271493, 'recall': 0.719491035280509, 'ROC_AUC': 0.9018193003062311, 'PRC_AUC': 0.9885972387467383, 'specificity': 0.719491035280509}, 'chosen_targets': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 'chosen_samples': [1168, 275, 576, 1065, 545, 261, 930, 1231, 110, 1095, 708, 937, 904, 1069, 848, 1117, 541, 1055, 916, 742], 'chosen_samples_score': [0.4907987186572007, 0.46154334664120944, 0.45458763235987343, 0.4500545053953596, 0.4439128167056432, 0.44216363269650194, 0.4366792146509221, 0.43636555353161466, 0.4352025253849189, 0.42814746951455485, 0.41165342369314095, 0.4106222645455103, 0.4001855191491843, 0.38651536338843606, 0.3715947167740212, 0.3660179459386874, 0.3582171421472975, 0.34890591559871553, 0.3405977550607511, 0.3250700237386216], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 283.82665887800977, 'batch_acquisition_elapsed_time': 36.64774874271825})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 0.7229223580195986, 'f1': 0.8172268907563025, 'precision': 0.8718812184295117, 'recall': 0.7810294968189705, 'ROC_AUC': 0.9091220281579728, 'PRC_AUC': 0.9934113305944463, 'specificity': 0.7810294968189705}, 'chosen_targets': [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], 'chosen_samples': [479, 148, 377, 425, 286, 1080, 849, 945, 517, 790, 89, 999, 50, 1104, 528, 228, 606, 589, 557, 939], 'chosen_samples_score': [0.49869158505128897, 0.48436999893345967, 0.45042128356222655, 0.43244981713469, 0.3709101504507315, 0.36262559395237315, 0.35900132655490613, 0.3564133640259587, 0.35140142317681167, 0.3459721124655808, 0.3430734657695801, 0.34186284970459824, 0.33278840424855394, 0.3057167364410609, 0.3055546487204267, 0.2687514766804332, 0.26834493368238477, 0.25484229818621396, 0.2496995366287158, 0.2491952367262642], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 189.80263820383698, 'batch_acquisition_elapsed_time': 36.12136771483347})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.9245689655172413, 'nll': 0.31904539568670864, 'f1': 0.8569604086845466, 'precision': 0.8307291666666667, 'recall': 0.891748602274918, 'ROC_AUC': 0.910816537577204, 'PRC_AUC': 0.9945612946077326, 'specificity': 0.891748602274918}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1176, 315, 747, 467, 1107, 129, 765, 1206, 494, 826, 989, 894, 1032, 413, 230, 1096, 51, 625, 1158, 757], 'chosen_samples_score': [0.4973095583298043, 0.4951773649886465, 0.4921645067450937, 0.4909196683256485, 0.4837544896960212, 0.47370390080723834, 0.4709671976121449, 0.4670111143356872, 0.4643660798922702, 0.4613996471583498, 0.45541981055821945, 0.45208439714875626, 0.43012705802669016, 0.4295981387715412, 0.4288260624626514, 0.42123568194018035, 0.4205707076511299, 0.41190464297208307, 0.3959028834079532, 0.39533298705443587], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 236.8371632429771, 'batch_acquisition_elapsed_time': 35.517093556933105})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.9137931034482759, 'nll': 0.6952037153572872, 'f1': 0.7904245709123758, 'precision': 0.8585317350715351, 'recall': 0.7502602660497397, 'ROC_AUC': 0.8421342870220814, 'PRC_AUC': 0.9879924289657586, 'specificity': 0.7502602660497397}, 'chosen_targets': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1], 'chosen_samples': [529, 835, 567, 570, 1042, 939, 1067, 791, 1072, 1193, 88, 337, 983, 101, 1149, 1187, 648, 235, 193, 945], 'chosen_samples_score': [0.4892947693521559, 0.48181547816289305, 0.4764640499797431, 0.47358805327743925, 0.47186430797885015, 0.4648552445608667, 0.44633398125436485, 0.43927047542152076, 0.430222289853456, 0.42703392255740935, 0.42070674521657003, 0.40069252806807476, 0.395755794598998, 0.3927534869188296, 0.3703939487060156, 0.36966250958787894, 0.3671144423926943, 0.3597468545683107, 0.3595385703018694, 0.34808372246629204], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 166.52548626530915, 'batch_acquisition_elapsed_time': 34.85729099670425})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.41166351581441946, 'f1': 0.8212438084755092, 'precision': 0.8070279928528886, 'recall': 0.8379024484287643, 'ROC_AUC': 0.9223301739763146, 'PRC_AUC': 0.9884406719197919, 'specificity': 0.8379024484287643}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], 'chosen_samples': [103, 438, 779, 983, 416, 613, 1160, 1053, 714, 345, 1057, 423, 1026, 480, 232, 286, 421, 265, 1080, 936], 'chosen_samples_score': [0.49183097512619045, 0.48928601292626983, 0.48728385853537637, 0.48457107466584304, 0.4837843815018332, 0.48056997998583595, 0.4799086719925546, 0.47104619071299925, 0.46372577182394403, 0.4619769902270763, 0.4548705946579048, 0.4521021372681635, 0.451614946655732, 0.44852397063536975, 0.44590568316409307, 0.4412243530179072, 0.44060191416808825, 0.4348167657498502, 0.43319693492130973, 0.43130504212219745], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 213.35257986513898, 'batch_acquisition_elapsed_time': 34.550808366853744})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.1400862068965517, 'nll': 1.6626636899750808, 'f1': 0.2457466918714556, 'precision': 0.07004310344827586, 'recall': 0.5, 'ROC_AUC': 0.6574719356755521, 'PRC_AUC': 0.9099722214012846, 'specificity': 0.5}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], 'chosen_samples': [408, 1095, 254, 930, 703, 986, 122, 984, 452, 1100, 537, 1117, 163, 739, 330, 891, 815, 1045, 101, 926], 'chosen_samples_score': [0.39814041561407565, 0.320079207477456, 0.30848674397467657, 0.29586625440555125, 0.29055672069006366, 0.24438606333518054, 0.23771637549657787, 0.23562726831504588, 0.2297423078879708, 0.2243827054009948, 0.22349155172862512, 0.2210787551518315, 0.21688523798163784, 0.20402745951872459, 0.2037603338128936, 0.20241998504329783, 0.19981229866866634, 0.19951358232980887, 0.19779239946935145, 0.19030126657846247], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 96.13717367965728, 'batch_acquisition_elapsed_time': 34.10819232603535})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9418103448275862, 'nll': 0.28824368838606207, 'f1': 0.8800080454375665, 'precision': 0.8776077356479366, 'recall': 0.8824561403508773, 'ROC_AUC': 0.9751158353043372, 'PRC_AUC': 0.9955363883683611, 'specificity': 0.8824561403508773}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], 'chosen_samples': [770, 370, 656, 1120, 362, 198, 1116, 372, 760, 1021, 1031, 26, 220, 93, 824, 488, 515, 615, 937, 196], 'chosen_samples_score': [0.49701664615541297, 0.4967070087048999, 0.4914019785344027, 0.4825606749354535, 0.47871413283967323, 0.4656361941256033, 0.4615052994247788, 0.4588707231519148, 0.44884232814987657, 0.43530151702350706, 0.43415313066156525, 0.41792665023917674, 0.39360150022657436, 0.39303474829234997, 0.3831040066767424, 0.38167297545617274, 0.3764458924936559, 0.3758710376904584, 0.33525300274570435, 0.33416032770727266], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 283.0087963468395, 'batch_acquisition_elapsed_time': 33.40030737174675})
