store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT LEAST CONFIDENCE (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 1234, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_lc_1234', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.least_confidence', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_lc_1234', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=least_confidence', '--acquisition_method=independent']
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
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5129310344827587, 'nll': 1.5020642773858432, 'f1': 0.3426836962679339, 'precision': 0.45493012098456403, 'recall': 0.38099225951280463, 'ROC_AUC': 0.69439697265625, 'PRC_AUC': 0.5661591911070201, 'specificity': 0.6901330084179449}, 'chosen_targets': [1, 0, 0, 2, 2, 0, 2, 0, 1, 1, 0, 0, 0, 2, 0, 1, 0, 2, 1, 1], 'chosen_samples': [797, 347, 794, 1420, 782, 1191, 260, 195, 528, 1446, 282, 1203, 1157, 1072, 1325, 72, 285, 1283, 522, 754], 'chosen_samples_score': [-0.3439166449245007, -0.366000918928824, -0.3683324785909493, -0.3688577316441123, -0.37195795394375897, -0.37900224768593926, -0.39378001776022703, -0.4075962934628767, -0.4106516238061226, -0.41685810233660453, -0.4169184848144786, -0.4169792189756507, -0.41761755056984323, -0.43001058455801056, -0.43461116836822977, -0.4365588860906168, -0.4375642989655284, -0.4401208834040863, -0.44124116536021396, -0.44227080570832616], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.96585107594728, 'batch_acquisition_elapsed_time': 53.593446193728596})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.6228448275862069, 'nll': 1.346112086855132, 'f1': 0.5803026735449622, 'precision': 0.6102809518589268, 'recall': 0.5959940860983604, 'ROC_AUC': 0.77899169921875, 'PRC_AUC': 0.6669472771119368, 'specificity': 0.7746028617928195}, 'chosen_targets': [2, 0, 1, 0, 2, 2, 1, 1, 2, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 2], 'chosen_samples': [718, 1025, 805, 810, 251, 561, 75, 842, 1177, 674, 847, 748, 447, 1006, 1380, 1309, 683, 916, 293, 1214], 'chosen_samples_score': [-0.33753396455418716, -0.36025876519026434, -0.36061299858341966, -0.36094007525706906, -0.37162797429375033, -0.37254662912485065, -0.37959861055119476, -0.37983367735598983, -0.37995651902713035, -0.380091106397454, -0.40567986228218267, -0.42362338393756366, -0.4259658980802647, -0.42705546301360964, -0.4330880228163531, -0.44002602725460976, -0.44132747793804483, -0.45274975643099324, -0.45550170443610577, -0.45637316195151], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 280.5915696830489, 'batch_acquisition_elapsed_time': 53.20031126495451})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.5862068965517241, 'nll': 1.3120513126767914, 'f1': 0.565145147901028, 'precision': 0.590132597754549, 'recall': 0.5494176104025371, 'ROC_AUC': 0.77813720703125, 'PRC_AUC': 0.6313193978324279, 'specificity': 0.7610890543437426}, 'chosen_targets': [1, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1], 'chosen_samples': [1161, 519, 1247, 1444, 232, 1427, 1189, 1399, 818, 1380, 1354, 1504, 1445, 746, 155, 1128, 939, 421, 133, 228], 'chosen_samples_score': [-0.3737324221819901, -0.38400875428307296, -0.4137784853795218, -0.4218633337505107, -0.4248525540971264, -0.4396276539576245, -0.439986050774484, -0.45040179477163067, -0.4607681484073528, -0.46170650941925867, -0.4657350458880925, -0.46874986117321593, -0.47296268971858524, -0.4796281251770432, -0.4809596578903685, -0.48443409381855196, -0.48492202465004297, -0.4883608867326816, -0.4931642244688353, -0.4932945728641145], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.6146051287651, 'batch_acquisition_elapsed_time': 52.54449196578935})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6357758620689655, 'nll': 1.3530479957317483, 'f1': 0.6195871469938865, 'precision': 0.6050100134607177, 'recall': 0.6483499843813698, 'ROC_AUC': 0.790771484375, 'PRC_AUC': 0.6799209217196228, 'specificity': 0.8025656341597056}, 'chosen_targets': [2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 0, 1, 2, 2, 0, 0, 0], 'chosen_samples': [1329, 1469, 370, 79, 774, 829, 375, 40, 66, 589, 1392, 1332, 325, 971, 413, 646, 1386, 198, 454, 853], 'chosen_samples_score': [-0.4038994334850294, -0.40579639213453583, -0.40688083744328185, -0.4078488929983548, -0.41580249917449674, -0.42413840882681136, -0.44370322809768614, -0.44959527102824265, -0.4530412998604231, -0.455225804570848, -0.45792956736428975, -0.47251221116055786, -0.47325773304869134, -0.4760479113666599, -0.47611559282342797, -0.47706817451388767, -0.4795466214119247, -0.48250085344898286, -0.4849838985328962, -0.48840105294222474], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.5993888541125, 'batch_acquisition_elapsed_time': 51.80671620601788})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.5107758620689655, 'nll': 2.2599868774414062, 'f1': 0.2633024326513284, 'precision': 0.4885185185185185, 'recall': 0.35270996468601257, 'ROC_AUC': 0.72637939453125, 'PRC_AUC': 0.589104694566061, 'specificity': 0.6769675116620065}, 'chosen_targets': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 1], 'chosen_samples': [1198, 58, 1204, 1006, 152, 260, 25, 64, 990, 1155, 631, 1331, 1403, 937, 1097, 679, 462, 137, 806, 646], 'chosen_samples_score': [-0.3754343971692873, -0.38187359575723684, -0.38197718141415216, -0.39516549692585373, -0.3952956483581223, -0.41921206588016185, -0.42031365718210306, -0.42578129431965567, -0.42805103005295353, -0.43152331868041344, -0.43660354087365355, -0.4373107750098333, -0.43835287904441306, -0.4394460637443173, -0.448717895230476, -0.45611766081724137, -0.45709240705757287, -0.4589721949487147, -0.4612019465905139, -0.46805156737361997], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.68940301286057, 'batch_acquisition_elapsed_time': 50.97690894827247})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.6508620689655172, 'nll': 1.104509945573478, 'f1': 0.613496718335428, 'precision': 0.6240779595449628, 'recall': 0.6084052055041111, 'ROC_AUC': 0.802734375, 'PRC_AUC': 0.6861406737954331, 'specificity': 0.8005408418294081}, 'chosen_targets': [1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0, 1, 2, 0, 2, 1, 1, 1, 1, 1], 'chosen_samples': [523, 249, 286, 667, 681, 583, 358, 1368, 1244, 576, 1119, 1027, 1425, 791, 1291, 1073, 616, 153, 727, 1327], 'chosen_samples_score': [-0.371867788677428, -0.37287872032296415, -0.37709749833573664, -0.3876037205520483, -0.4028691503590629, -0.40816206873817484, -0.4141922787981401, -0.4256981995561723, -0.4302356080256831, -0.4311635254497696, -0.435195771444811, -0.4548627001011536, -0.45627078381042196, -0.4585895932879973, -0.4655530410135888, -0.4689107823219084, -0.4721045684891637, -0.4727677561359385, -0.4753687162037321, -0.47707473016435803], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 440.2856282941066, 'batch_acquisition_elapsed_time': 50.371857919730246})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.5969827586206896, 'nll': 0.8459312175882274, 'f1': 0.5207239087255878, 'precision': 0.5610931754999552, 'recall': 0.5500063533410633, 'ROC_AUC': 0.7939453125, 'PRC_AUC': 0.6582265569512256, 'specificity': 0.7525230701274258}, 'chosen_targets': [2, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 0, 0, 1], 'chosen_samples': [1148, 826, 1060, 100, 537, 1103, 1022, 1338, 107, 660, 967, 949, 591, 868, 332, 1077, 307, 929, 339, 115], 'chosen_samples_score': [-0.3636825402806048, -0.36479112052298573, -0.3710952244174957, -0.37172575575941713, -0.38885769130008263, -0.398530654397795, -0.39964448879976167, -0.40595466997741403, -0.40930007224918385, -0.41284405393644535, -0.4152234685452976, -0.4171847062677584, -0.41744293680460665, -0.42502733861991143, -0.43949216199386626, -0.4423320510414167, -0.44333538778083953, -0.4467715429906019, -0.45072780811737256, -0.45108253970917467], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 322.08282908890396, 'batch_acquisition_elapsed_time': 49.79067139001563})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 1.0304890336661503, 'f1': 0.6392436455347704, 'precision': 0.6392870133223272, 'recall': 0.6571250072798699, 'ROC_AUC': 0.80322265625, 'PRC_AUC': 0.6987232571294468, 'specificity': 0.803043033684292}, 'chosen_targets': [1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2], 'chosen_samples': [297, 846, 613, 1200, 37, 340, 143, 2, 145, 927, 804, 432, 647, 997, 713, 588, 302, 517, 1106, 779], 'chosen_samples_score': [-0.3416389374123788, -0.3633185447671425, -0.36610534212747836, -0.3723759573314674, -0.38243726796985605, -0.38419138738391473, -0.3844245738603186, -0.38539979123261264, -0.3857690685795534, -0.38797623566663164, -0.39167178570893524, -0.4005112059185214, -0.40187621635479726, -0.4049962059631829, -0.4182189273545543, -0.4211281073957122, -0.424682963897625, -0.4258051384223711, -0.4269894867699062, -0.4286407486954553], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.76207000389695, 'batch_acquisition_elapsed_time': 49.04006819007918})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5538793103448276, 'nll': 0.7856999759016365, 'f1': 0.43784682051830065, 'precision': 0.5152626579695303, 'recall': 0.5409591956670213, 'ROC_AUC': 0.73779296875, 'PRC_AUC': 0.6245227560266349, 'specificity': 0.7526092020949856}, 'chosen_targets': [1, 0, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 0, 2, 0], 'chosen_samples': [604, 792, 468, 550, 1339, 1351, 1061, 414, 709, 984, 386, 73, 249, 817, 197, 1284, 460, 289, 736, 660], 'chosen_samples_score': [-0.3367387025939424, -0.35040746650769583, -0.3527165660536123, -0.3533056849992691, -0.35562446019726657, -0.3608342047106142, -0.36443754161854675, -0.36618879862481, -0.3672632384962652, -0.3673069779137384, -0.3677639601126101, -0.3682993181968216, -0.36849365868091793, -0.36851701230614864, -0.3688142308643286, -0.3695592230398037, -0.37537042349479344, -0.3874711856626403, -0.38790436072809525, -0.3895647722712224], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.7812463371083, 'batch_acquisition_elapsed_time': 48.047714073210955})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6681034482758621, 'nll': 0.8322737463589372, 'f1': 0.6707329643258398, 'precision': 0.6889208036127469, 'recall': 0.6597200823816558, 'ROC_AUC': 0.83111572265625, 'PRC_AUC': 0.7160074000668439, 'specificity': 0.8154824466832936}, 'chosen_targets': [2, 0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2], 'chosen_samples': [447, 1131, 994, 382, 637, 632, 983, 286, 911, 581, 552, 109, 909, 947, 1238, 186, 1222, 1126, 722, 669], 'chosen_samples_score': [-0.4202222427266219, -0.4286276805112159, -0.43456340789318487, -0.43946922315321246, -0.4419030746737658, -0.44822027707669404, -0.45057899388096634, -0.4586134256412098, -0.46085104068216887, -0.46131581578767267, -0.4663403524732197, -0.4721552841293873, -0.47364774189075454, -0.4760496913435254, -0.4809059038232777, -0.4817378742062211, -0.4825707104092843, -0.4857633437275345, -0.48587254620575565, -0.4910806801505622], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.6311025307514, 'batch_acquisition_elapsed_time': 47.552720674779266})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6443965517241379, 'nll': 0.5848464308113888, 'f1': 0.5918008261982941, 'precision': 0.7006140155728587, 'recall': 0.5636021061325625, 'ROC_AUC': 0.82928466796875, 'PRC_AUC': 0.6962571626760021, 'specificity': 0.7826586013972584}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 2, 2], 'chosen_samples': [925, 917, 205, 999, 1141, 473, 654, 463, 1267, 774, 1079, 674, 684, 591, 772, 1323, 149, 840, 1275, 660], 'chosen_samples_score': [-0.3963747733877373, -0.40717940290825605, -0.410386485477213, -0.41296844132736377, -0.4146226754507097, -0.4257177957408532, -0.4278938810649769, -0.43028160358123757, -0.43457468820913386, -0.4346476942273617, -0.44131116786895774, -0.4434701626044951, -0.4468425338981978, -0.44767692411369736, -0.4485992155609061, -0.44864945519571553, -0.4494364665499457, -0.44961725910352646, -0.4503037042439251, -0.4510014008858379], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 243.0470914002508, 'batch_acquisition_elapsed_time': 46.89887363836169})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6767241379310345, 'nll': 0.7359015366126751, 'f1': 0.674010942208706, 'precision': 0.6739914507959717, 'recall': 0.6740814127712745, 'ROC_AUC': 0.84747314453125, 'PRC_AUC': 0.7378314043015286, 'specificity': 0.8194921298369575}, 'chosen_targets': [1, 1, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 1, 1], 'chosen_samples': [528, 1318, 408, 774, 261, 1069, 849, 1038, 926, 367, 126, 698, 464, 1028, 1142, 431, 341, 176, 1252, 811], 'chosen_samples_score': [-0.43136029365304646, -0.4345628305306417, -0.4362840176631653, -0.43793776719991745, -0.4517311661689362, -0.4544655620305968, -0.4575900417996938, -0.4613778715855983, -0.4687505346595826, -0.46899853592110513, -0.47176873024903854, -0.47487730609487333, -0.47911348929472247, -0.48023690409151576, -0.48060001886964787, -0.4839264303100787, -0.48477327182642693, -0.4857490056329409, -0.48643142289344227, -0.48809143877002575], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.9609240121208, 'batch_acquisition_elapsed_time': 45.95839371299371})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.5926724137931034, 'nll': 0.6002890488197064, 'f1': 0.4709340900645249, 'precision': 0.657028786280487, 'recall': 0.4670372782286885, 'ROC_AUC': 0.80828857421875, 'PRC_AUC': 0.6851352996452429, 'specificity': 0.7460022472877887}, 'chosen_targets': [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1], 'chosen_samples': [761, 556, 620, 347, 614, 457, 1123, 1173, 642, 1246, 93, 60, 617, 636, 933, 974, 695, 463, 655, 751], 'chosen_samples_score': [-0.39047581393518005, -0.3968771637558918, -0.39742559406381445, -0.4025539283947734, -0.4027425588848985, -0.4028818660108621, -0.40374232806619964, -0.40434041570134327, -0.40902494969370073, -0.41186779458344197, -0.41404986147796324, -0.41435561583067215, -0.41539179578727486, -0.4174540710329659, -0.41912069038269156, -0.42207484957242053, -0.42286888332881556, -0.4273072978184157, -0.43046251853949163, -0.4337856476389394], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 322.10116546601057, 'batch_acquisition_elapsed_time': 45.308083458803594})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6573275862068966, 'nll': 0.6838263807625606, 'f1': 0.6523675165876511, 'precision': 0.6421280473912053, 'recall': 0.6657370934523527, 'ROC_AUC': 0.8363037109375, 'PRC_AUC': 0.7132869567213506, 'specificity': 0.812916674305059}, 'chosen_targets': [1, 0, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2], 'chosen_samples': [24, 482, 593, 259, 541, 1094, 775, 455, 208, 113, 302, 783, 1260, 55, 1269, 681, 1132, 808, 308, 177], 'chosen_samples_score': [-0.40748165472581505, -0.40978335989709364, -0.41308458670148784, -0.41480377960683723, -0.4172530612615785, -0.4204949151819268, -0.42439632859342985, -0.42485825362467383, -0.42640027094571953, -0.4293411631655859, -0.43254716514446057, -0.44040906851834244, -0.44049911805031533, -0.442031591945246, -0.4424448568876849, -0.4438212623152697, -0.44588401870098227, -0.449625464500399, -0.45507464578707785, -0.456207253679842], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.7474409067072, 'batch_acquisition_elapsed_time': 44.72628103476018})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6120689655172413, 'nll': 0.6308898268074825, 'f1': 0.5481360697589088, 'precision': 0.5835783317796847, 'recall': 0.6088347178322401, 'ROC_AUC': 0.8192138671875, 'PRC_AUC': 0.7113066591440477, 'specificity': 0.7735542560103964}, 'chosen_targets': [1, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0, 2, 1, 1, 1, 2, 2, 2, 2, 1], 'chosen_samples': [927, 1001, 407, 595, 230, 855, 66, 465, 969, 771, 72, 242, 654, 208, 24, 158, 587, 372, 1153, 152], 'chosen_samples_score': [-0.33950448127781807, -0.3432553346809126, -0.34578903664682675, -0.3503059179556294, -0.3511955885422658, -0.35280501950652665, -0.3542915090456319, -0.3557828287066306, -0.35757472442503163, -0.3603083984131304, -0.3617641504322282, -0.3633678809451427, -0.3634706808953707, -0.36452345829927063, -0.3654837485689207, -0.36652247198328997, -0.36802235694992524, -0.368813888776412, -0.3710344834630076, -0.3712613635483576], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.44316484918818, 'batch_acquisition_elapsed_time': 44.08235718496144})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.6594827586206896, 'nll': 0.8526681045006061, 'f1': 0.6554636828609431, 'precision': 0.6422733931565374, 'recall': 0.6773528539737502, 'ROC_AUC': 0.83123779296875, 'PRC_AUC': 0.7040649353207538, 'specificity': 0.8179891488270193}, 'chosen_targets': [1, 1, 2, 0, 2, 2, 1, 2, 1, 2, 2, 2, 0, 0, 2, 1, 0, 1, 2, 1], 'chosen_samples': [674, 775, 858, 170, 72, 1172, 781, 833, 1242, 459, 1091, 1165, 590, 1156, 690, 1108, 717, 140, 12, 925], 'chosen_samples_score': [-0.38097768823160977, -0.3962144987754623, -0.4524541094811979, -0.46763815066275194, -0.4769392580794039, -0.47702659732120223, -0.49271384463940593, -0.4932421053556623, -0.4934910735109368, -0.49854905601765487, -0.49919778237414486, -0.500267546619365, -0.5008690107245612, -0.5017059582993074, -0.5018438254816066, -0.5069407583473781, -0.5076195647355989, -0.507912234241798, -0.5084255102760584, -0.5091233799781487], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 482.12919777678326, 'batch_acquisition_elapsed_time': 43.08264149632305})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.6724137931034483, 'nll': 0.5995802386053677, 'f1': 0.6234514130855594, 'precision': 0.7069462778098616, 'recall': 0.5951424736733429, 'ROC_AUC': 0.844970703125, 'PRC_AUC': 0.738418643925603, 'specificity': 0.8066272947549414}, 'chosen_targets': [2, 2, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 2, 1, 1, 1, 2, 2, 0], 'chosen_samples': [575, 296, 498, 966, 567, 979, 661, 246, 510, 685, 643, 47, 628, 542, 971, 492, 824, 691, 109, 549], 'chosen_samples_score': [-0.3655317710081368, -0.38593568031209646, -0.4057733431135278, -0.42439365567572707, -0.4313381399919609, -0.4365948603586146, -0.4393178943554954, -0.44445381752576985, -0.451349980894381, -0.45230461634276226, -0.45620738818306694, -0.45934022693963744, -0.4599840152940562, -0.46539207609277095, -0.47275397059985835, -0.4745562314276371, -0.47992384993131365, -0.480827605412111, -0.48086933108348584, -0.4818634092229276], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 282.93463395303115, 'batch_acquisition_elapsed_time': 42.5866716732271})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.5991379310344828, 'nll': 0.696558064427869, 'f1': 0.5167305236270753, 'precision': 0.6756449305972416, 'recall': 0.5009919153734971, 'ROC_AUC': 0.80987548828125, 'PRC_AUC': 0.6757443449370447, 'specificity': 0.7493967852407054}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0], 'chosen_samples': [1002, 903, 420, 77, 1196, 894, 267, 423, 320, 557, 130, 1106, 1008, 587, 480, 88, 1188, 495, 189, 657], 'chosen_samples_score': [-0.3940379250666072, -0.4075937351094768, -0.4087137909986249, -0.41095155931090777, -0.4119615507832564, -0.4134324056785522, -0.43303010725019203, -0.4391108513351767, -0.44708808493658886, -0.4535667353970302, -0.47962591442771446, -0.48194880994730654, -0.48344668941625385, -0.4865798107190789, -0.487327187595308, -0.49227805212092085, -0.49322502831372683, -0.4936399161862456, -0.49416349573412865, -0.4943927689543401], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.8670689142309, 'batch_acquisition_elapsed_time': 41.97438182588667})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5969827586206896, 'nll': 0.5945823274809738, 'f1': 0.5602916993498732, 'precision': 0.6262712555816005, 'recall': 0.5506758366556013, 'ROC_AUC': 0.81378173828125, 'PRC_AUC': 0.672933357276586, 'specificity': 0.7508841984794797}, 'chosen_targets': [1, 1, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0, 2, 1, 0, 1, 0, 2, 0, 2], 'chosen_samples': [557, 1023, 807, 432, 935, 176, 1088, 872, 766, 344, 696, 664, 44, 81, 1085, 111, 1065, 1045, 284, 99], 'chosen_samples_score': [-0.3536066816035039, -0.3573622528932844, -0.3577797687319097, -0.35860546242294206, -0.36373284484640356, -0.36417163571878375, -0.3682568168000968, -0.3755225665218679, -0.3771905214477177, -0.381044116929717, -0.3836354016491263, -0.38386298167368293, -0.38647095379054297, -0.38776988077946184, -0.38876836877719945, -0.38930558430548295, -0.389351484388036, -0.39471682124938484, -0.3962848705029814, -0.3979426193224215], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.4905085428618, 'batch_acquisition_elapsed_time': 41.21143319783732})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5193965517241379, 'nll': 0.6552925767569706, 'f1': 0.4571071840184402, 'precision': 0.4434410328450064, 'recall': 0.37948717948717947, 'ROC_AUC': 0.74267578125, 'PRC_AUC': 0.6019978079204543, 'specificity': 0.6808004205917092}, 'chosen_targets': [0, 2, 2, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 2, 0, 0, 0, 1, 0, 0], 'chosen_samples': [447, 379, 641, 1092, 783, 767, 618, 437, 259, 36, 580, 834, 893, 18, 492, 659, 937, 449, 873, 470], 'chosen_samples_score': [-0.3461550189185382, -0.3499329367115492, -0.3507789284437462, -0.3524842924177599, -0.3558060336912956, -0.35603784750941786, -0.36016200234576584, -0.36254481015500817, -0.36397564094789653, -0.3680120213670153, -0.369623399421685, -0.3722388619474033, -0.37435502063150655, -0.37612521656125736, -0.37789871633088173, -0.38128737155461023, -0.38387696612750294, -0.3886475111700764, -0.38946279809579176, -0.39028645583521715], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.78947943402454, 'batch_acquisition_elapsed_time': 40.51515530701727})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6918103448275862, 'nll': 0.5033351963964002, 'f1': 0.6819866871479775, 'precision': 0.7086254947979076, 'recall': 0.6769818453279118, 'ROC_AUC': 0.8607177734375, 'PRC_AUC': 0.7561946622743723, 'specificity': 0.814676414419306}, 'chosen_targets': [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1, 0], 'chosen_samples': [263, 602, 229, 916, 971, 828, 606, 227, 1039, 755, 1006, 553, 1050, 316, 498, 152, 441, 792, 959, 1144], 'chosen_samples_score': [-0.39159569045749004, -0.39602971348725874, -0.3979761546477717, -0.4070276988991703, -0.4085568724153565, -0.412960554474343, -0.41354543325748255, -0.41520685624013315, -0.42390413806489796, -0.4358774897369562, -0.4392586985181883, -0.43941427122729304, -0.4450047216825319, -0.4457420289179926, -0.45194956847355794, -0.45314920630067684, -0.45441369832244843, -0.45616130213863443, -0.4570789126637155, -0.45908428163572856], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.9556752196513, 'batch_acquisition_elapsed_time': 40.07435051910579})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.6336206896551724, 'nll': 0.5720343754209322, 'f1': 0.5690078712700329, 'precision': 0.6817038654039292, 'recall': 0.6008308317052897, 'ROC_AUC': 0.82269287109375, 'PRC_AUC': 0.6947478267350082, 'specificity': 0.7667368089328161}, 'chosen_targets': [1, 0, 1, 2, 0, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1], 'chosen_samples': [745, 392, 66, 643, 929, 93, 1077, 384, 698, 183, 130, 78, 968, 173, 511, 169, 589, 35, 1020, 1023], 'chosen_samples_score': [-0.3680517186409274, -0.369206707235555, -0.3718509644701097, -0.37351746959049037, -0.3749488938463043, -0.3775427758041617, -0.38034334963409283, -0.38057227939836624, -0.3832242431560182, -0.3840626857934969, -0.38990965767330427, -0.390612575900197, -0.3968538591110215, -0.3972179661415521, -0.397336657354213, -0.4047051326803047, -0.40625974047564517, -0.4063519849647481, -0.40695882579787895, -0.4091768577122313], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.3361425930634, 'batch_acquisition_elapsed_time': 39.21301209833473})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.665948275862069, 'nll': 0.5917139382197939, 'f1': 0.6214519421933367, 'precision': 0.6393944781503952, 'recall': 0.6823379765667603, 'ROC_AUC': 0.83929443359375, 'PRC_AUC': 0.7623727115467621, 'specificity': 0.8146266557488578}, 'chosen_targets': [1, 2, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2], 'chosen_samples': [533, 613, 607, 478, 760, 336, 843, 122, 442, 426, 653, 79, 814, 66, 472, 547, 229, 255, 821, 1070], 'chosen_samples_score': [-0.3473893934074869, -0.37584545682122744, -0.37633903480382164, -0.3822900254899903, -0.38611005830564266, -0.3886269680727938, -0.3904351901288497, -0.39534977916760455, -0.4012952274101171, -0.40443660742479337, -0.4049905254524794, -0.41210542176053405, -0.41456476793678476, -0.41564287481543144, -0.41587850140178506, -0.41679035462406205, -0.41691635765089763, -0.42074539897459706, -0.42084854994428506, -0.4232379591030265], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.1509496686049, 'batch_acquisition_elapsed_time': 38.567405748181045})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.6982758620689655, 'nll': 0.6108570098876953, 'f1': 0.699927820924739, 'precision': 0.699099215339524, 'recall': 0.704220471523796, 'ROC_AUC': 0.85699462890625, 'PRC_AUC': 0.7451636190763483, 'specificity': 0.8354321816965496}, 'chosen_targets': [1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0], 'chosen_samples': [94, 13, 779, 288, 768, 537, 1054, 25, 441, 565, 786, 866, 240, 916, 415, 1039, 898, 109, 189, 304], 'chosen_samples_score': [-0.3856198290699192, -0.4112250968758183, -0.4384131044676774, -0.44489284664416623, -0.444921038836139, -0.4526990814967604, -0.45311915386391677, -0.45447018421944274, -0.4546026761465756, -0.4599314839941827, -0.46388940815516877, -0.46484090843410425, -0.4655552806824188, -0.46570875287105024, -0.4676356743796709, -0.4681111469257945, -0.47058407916008593, -0.4707231203378012, -0.47092733475695375, -0.47138342715172976], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 282.8876991327852, 'batch_acquisition_elapsed_time': 37.98656691238284})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6788793103448276, 'nll': 0.5383519468636349, 'f1': 0.6696852425180598, 'precision': 0.7074067247021688, 'recall': 0.6492319075377097, 'ROC_AUC': 0.860595703125, 'PRC_AUC': 0.7432805902686292, 'specificity': 0.8083394585965668}, 'chosen_targets': [1, 1, 1, 1, 0, 0, 2, 2, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0], 'chosen_samples': [540, 944, 278, 1031, 704, 570, 695, 1029, 246, 912, 460, 510, 958, 993, 869, 356, 318, 58, 690, 663], 'chosen_samples_score': [-0.40099737310276784, -0.406543311978195, -0.40967406815961527, -0.41009926128278895, -0.41061531691601993, -0.41073056805626257, -0.4121966900647113, -0.41562859360539306, -0.41578494574162234, -0.4190331011708765, -0.4291724464810283, -0.4294832387948451, -0.4326166114721457, -0.4330939165408718, -0.43462921428276513, -0.4349109923010368, -0.435759579365622, -0.4358300622189563, -0.4360715122290088, -0.43624897802449314], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.1281623658724, 'batch_acquisition_elapsed_time': 37.20990824513137})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6271551724137931, 'nll': 0.5447574155084018, 'f1': 0.5489044540229885, 'precision': 0.6828162593692749, 'recall': 0.531526734329749, 'ROC_AUC': 0.81829833984375, 'PRC_AUC': 0.7083410370087824, 'specificity': 0.7657912123278124}, 'chosen_targets': [2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 0, 1, 0, 0, 1, 2, 1, 1], 'chosen_samples': [647, 808, 88, 628, 655, 540, 819, 108, 715, 199, 93, 203, 681, 995, 755, 861, 885, 300, 986, 77], 'chosen_samples_score': [-0.36615555925497223, -0.366913815705145, -0.36787866917848483, -0.37409412669199743, -0.37464742632452674, -0.37528733375091855, -0.37730143918055364, -0.3781764547574914, -0.37896904653388563, -0.38152086014214714, -0.38181779245970016, -0.38205966373996303, -0.39055739192802097, -0.3921233601666101, -0.3923737884531982, -0.3926205200545924, -0.39327706298989895, -0.40459967662781066, -0.4069890383556969, -0.4087116102790127], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 401.4913108837791, 'batch_acquisition_elapsed_time': 36.42671066801995})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6099137931034483, 'nll': 0.5692600381785425, 'f1': 0.5616796337461628, 'precision': 0.672478479872861, 'recall': 0.5393848906960614, 'ROC_AUC': 0.81903076171875, 'PRC_AUC': 0.6786260518391118, 'specificity': 0.7584398416219348}, 'chosen_targets': [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0], 'chosen_samples': [7, 896, 222, 781, 211, 528, 918, 657, 215, 961, 593, 514, 89, 972, 297, 476, 45, 121, 522, 905], 'chosen_samples_score': [-0.381985464015248, -0.39540522199200895, -0.4001682880867114, -0.4141090294885032, -0.4154695987823946, -0.4370308904632135, -0.44552254939276004, -0.44632433605061433, -0.4469727085769996, -0.4601836985230259, -0.46635352463308094, -0.4663549451900552, -0.46839413088353316, -0.469411319722675, -0.4720057505459528, -0.47739369796068903, -0.4794393293628453, -0.4812155094444634, -0.4827241055968532, -0.4833212246224802], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.49092054599896, 'batch_acquisition_elapsed_time': 35.93114203913137})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.7047413793103449, 'nll': 0.5429553656742491, 'f1': 0.6880078088029844, 'precision': 0.6770559306250727, 'recall': 0.706742880287171, 'ROC_AUC': 0.8546142578125, 'PRC_AUC': 0.767251415851068, 'specificity': 0.8378949448973647}, 'chosen_targets': [1, 2, 2, 2, 1, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2], 'chosen_samples': [788, 290, 871, 435, 347, 214, 274, 577, 931, 125, 328, 84, 635, 661, 906, 267, 261, 367, 133, 463], 'chosen_samples_score': [-0.3673716027868002, -0.3757302836852284, -0.38503240260725513, -0.406530820286203, -0.4084613891106835, -0.41132324729018477, -0.425638036259332, -0.4278397205135815, -0.42936705757203564, -0.43972573740750687, -0.4420991795575738, -0.4432001522324234, -0.4452628529681126, -0.44851607790477255, -0.4498739664616652, -0.4511268272672486, -0.4529097125873365, -0.45452757063730903, -0.45482921758637634, -0.45705613032811154], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 362.50860216794536, 'batch_acquisition_elapsed_time': 35.05543217808008})
