store = {}
store['args']={'experiment_description': 'COVID BINARY:RESNET BN DROPOUT MEAN STD (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.9375, 'quickquick': False, 'seed': 254, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_binary_scratch_meanstd_254', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_binary_config.py', 'type': 'AcquisitionFunction.mean_stddev', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_binary'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_binary_scratch_meanstd_254', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_binary_config.py', '--dataset=covid_binary', '--type=mean_stddev', '--acquisition_method=independent']
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
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.7823275862068966, 'nll': 0.7758560838370487, 'f1': 0.6670810630332393, 'precision': 0.6489856297548605, 'recall': 0.7446500867553498, 'ROC_AUC': 0.8262082305377324, 'PRC_AUC': 0.9646423394955237, 'specificity': 0.6923076923076923}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [1286, 738, 1082, 390, 1117, 1534, 655, 1136, 1521, 1269, 672, 1444, 380, 960, 1451, 246, 727, 579, 1083, 1272], 'chosen_samples_score': [0.17805661318843552, 0.15459984808650837, 0.14831918364650049, 0.14384914640443142, 0.1398392139363839, 0.13933988876583947, 0.13772549749254795, 0.13347342853087862, 0.12750480847992837, 0.125990269212403, 0.1257914111588482, 0.1253798381785401, 0.1246409502857743, 0.12029325712072561, 0.11971837627394251, 0.11895768824754172, 0.11870672090256569, 0.11701052572635698, 0.11691980261012602, 0.11652604033271152], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.76907589519396, 'batch_acquisition_elapsed_time': 55.65165412472561})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8857758620689655, 'nll': 2.0582794978700836, 'f1': 0.67066197051064, 'precision': 0.821590909090909, 'recall': 0.6309427414690573, 'ROC_AUC': 0.8287557638455627, 'PRC_AUC': 0.9706871432808583, 'specificity': 0.27692307692307694}, 'chosen_targets': [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1], 'chosen_samples': [665, 827, 847, 22, 1248, 644, 911, 908, 81, 699, 429, 15, 1456, 1131, 1052, 1233, 1306, 1122, 1099, 342], 'chosen_samples_score': [0.11244574658216697, 0.11199879707760774, 0.10995117352987543, 0.10963053222152472, 0.10826740024116412, 0.10806439701295512, 0.1072468043601442, 0.1070156709990552, 0.10672444621318275, 0.10619827599434706, 0.10221974915895729, 0.10185475248887915, 0.10125918411761278, 0.10115805245519624, 0.09996269076050765, 0.09891805459989507, 0.0986217728393323, 0.09810932676004067, 0.09800415481143014, 0.09789597871628429], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 239.0577360210009, 'batch_acquisition_elapsed_time': 54.89727856172249})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.9094827586206896, 'nll': 0.8141531450995083, 'f1': 0.8190730837789661, 'precision': 0.8080851521341792, 'recall': 0.8314632735685368, 'ROC_AUC': 0.9382405455513694, 'PRC_AUC': 0.9903358275502919, 'specificity': 0.7230769230769231}, 'chosen_targets': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], 'chosen_samples': [1242, 959, 886, 515, 1190, 175, 587, 1255, 1158, 276, 1423, 1195, 1451, 1085, 189, 285, 542, 1283, 1280, 600], 'chosen_samples_score': [0.2014398497491559, 0.20030563826199804, 0.1953490738919736, 0.19255233958228363, 0.1922426453947918, 0.18468257271185662, 0.18317484645891247, 0.17785171659641855, 0.1774467044093201, 0.17254410050733904, 0.16991022619877746, 0.16824826803959905, 0.16630313444821826, 0.163499762030034, 0.15928948088340322, 0.1592785754735262, 0.15614155784235573, 0.15570105165778414, 0.15400001630319743, 0.15352820235318038], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 437.6186484177597, 'batch_acquisition_elapsed_time': 54.657508709002286})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.9051724137931034, 'nll': 0.4470413799943595, 'f1': 0.8231462231462232, 'precision': 0.7961926445941245, 'recall': 0.8611528822055138, 'ROC_AUC': 0.9051777458386654, 'PRC_AUC': 0.9855439430314681, 'specificity': 0.8}, 'chosen_targets': [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [1091, 237, 980, 862, 687, 1325, 192, 1221, 664, 129, 1183, 942, 600, 46, 66, 1342, 1042, 1026, 133, 132], 'chosen_samples_score': [0.15373765841939205, 0.1357163936444212, 0.13116592765844592, 0.11963025786055016, 0.11951582751529832, 0.11836881878490363, 0.10789808285766542, 0.10681822248660225, 0.10549300852401905, 0.10508410991880132, 0.10450575912196236, 0.10418684356166347, 0.10397232901483981, 0.1016964026351947, 0.10127090795598637, 0.10037245549254883, 0.0998852428480644, 0.09868954778373745, 0.0980192947249193, 0.09762402531033085], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.6641093818471, 'batch_acquisition_elapsed_time': 54.7359825046733})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8857758620689655, 'nll': 0.3379632357893319, 'f1': 0.8027827900076185, 'precision': 0.7669510427121669, 'recall': 0.8691922112974744, 'ROC_AUC': 0.9492207821003319, 'PRC_AUC': 0.9914349972322001, 'specificity': 0.8461538461538461}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [35, 90, 1476, 42, 724, 224, 1452, 497, 1482, 619, 135, 1188, 442, 194, 474, 185, 1362, 1336, 482, 1455], 'chosen_samples_score': [0.11533063681411732, 0.11191617857560765, 0.11103136814819517, 0.10981670553585166, 0.10751880019718399, 0.10517659287873124, 0.1039211769202113, 0.10179998678574338, 0.09847741163263148, 0.09767744483114141, 0.09759892733275934, 0.09642045211874314, 0.0962497723607405, 0.09551645263782868, 0.09536338877887751, 0.09438637123168354, 0.09412548441586639, 0.09351876538434505, 0.09318661187649256, 0.0928031244608048], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.4057407383807, 'batch_acquisition_elapsed_time': 53.84894348401576})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.9310344827586207, 'nll': 0.3619036181219693, 'f1': 0.8568729516097937, 'precision': 0.8568729516097937, 'recall': 0.8568729516097937, 'ROC_AUC': 0.95675877124153, 'PRC_AUC': 0.9920879079044889, 'specificity': 0.7538461538461538}, 'chosen_targets': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [1471, 289, 902, 662, 540, 108, 482, 722, 920, 1373, 789, 549, 98, 7, 247, 1353, 153, 1412, 292, 51], 'chosen_samples_score': [0.07899681927575794, 0.07861543550658154, 0.07855015128655071, 0.07850452858880921, 0.07816514849202219, 0.07783961308237732, 0.07734876722384515, 0.07729310473749426, 0.07699654601072947, 0.07639260625343147, 0.07633353229720813, 0.07613299918388226, 0.0759908207643217, 0.07527208107507519, 0.07512151790715399, 0.07475752421804455, 0.07440704814203045, 0.07411396294575306, 0.07387140768231736, 0.07246950474654755], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.19394615432248, 'batch_acquisition_elapsed_time': 51.9653206160292})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.9267241379310345, 'nll': 0.31442221279801996, 'f1': 0.8552926068611264, 'precision': 0.8395403426409277, 'recall': 0.8736842105263158, 'ROC_AUC': 0.9059783143810921, 'PRC_AUC': 0.9947153084368496, 'specificity': 0.8}, 'chosen_targets': [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], 'chosen_samples': [391, 457, 885, 1107, 555, 333, 1373, 328, 1306, 1232, 250, 1410, 177, 267, 608, 918, 919, 236, 429, 252], 'chosen_samples_score': [0.17511732802420976, 0.1632042124838761, 0.14993157674295235, 0.14228081778077362, 0.1398740512060336, 0.137925011040616, 0.13722088403947202, 0.13651167951602283, 0.13456899655545168, 0.13274785722532884, 0.12968359632607734, 0.1267893750180197, 0.12656659969528866, 0.12502770263102653, 0.12413434406362711, 0.1215007123046788, 0.11897185732865997, 0.11790003224664276, 0.1151476438085652, 0.11462000658994852], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 478.18553569400683, 'batch_acquisition_elapsed_time': 51.42170932330191})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.8706896551724138, 'nll': 8.831397747171335, 'f1': 0.5594657889739857, 'precision': 0.8251526251526251, 'recall': 0.5513398881819934, 'ROC_AUC': 0.9261173715687605, 'PRC_AUC': 0.9680127936780444, 'specificity': 0.1076923076923077}, 'chosen_targets': [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], 'chosen_samples': [936, 172, 56, 1267, 635, 603, 41, 694, 658, 898, 1095, 929, 214, 1364, 474, 155, 466, 990, 1384, 586], 'chosen_samples_score': [0.09823578560731157, 0.08562361319957468, 0.07797602450654742, 0.07247277918049651, 0.07206603075410101, 0.0715392257110673, 0.07126833577255151, 0.06524277707098541, 0.06394040486469815, 0.06084862963918716, 0.06038052605014373, 0.05769867516888118, 0.057049228801420215, 0.056791139073173964, 0.0557143585922218, 0.04929006103736973, 0.04796517085502215, 0.045203206634202464, 0.03819395292608635, 0.036010360626218305], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.78641361696646, 'batch_acquisition_elapsed_time': 50.58942878060043})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.8771551724137931, 'nll': 3.816072135136045, 'f1': 0.6385255648038051, 'precision': 0.7843480049362401, 'recall': 0.606612685560054, 'ROC_AUC': 0.8187854528754912, 'PRC_AUC': 0.9676447129631018, 'specificity': 0.23076923076923078}, 'chosen_targets': [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'chosen_samples': [326, 593, 296, 918, 707, 504, 909, 741, 265, 1090, 346, 492, 240, 381, 1137, 201, 310, 816, 1077, 1138], 'chosen_samples_score': [0.11392296777977937, 0.09447069805709932, 0.08989021146029562, 0.08686097235363371, 0.08562318556021778, 0.08559224774709617, 0.08175480857266958, 0.08038269477370197, 0.07938164294037775, 0.07809680363293003, 0.07692093767094257, 0.07402185078630089, 0.07315467391792735, 0.07276876132664574, 0.07214471168673994, 0.0717237578056098, 0.07139394027988465, 0.07124282363986978, 0.07117905283777701, 0.07061274201538004], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.6033835997805, 'batch_acquisition_elapsed_time': 49.90549805667251})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 0.4954216069188611, 'f1': 0.8252354048964219, 'precision': 0.8575953725382179, 'recall': 0.800347021399653, 'ROC_AUC': 0.9087286921518244, 'PRC_AUC': 0.9868614681847862, 'specificity': 0.6307692307692307}, 'chosen_targets': [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], 'chosen_samples': [645, 1064, 1221, 428, 69, 481, 332, 1226, 44, 1172, 1386, 801, 431, 550, 6, 608, 901, 1234, 52, 767], 'chosen_samples_score': [0.0632547999792401, 0.06250166465680959, 0.061537311720223725, 0.06147288647758786, 0.05928670372661064, 0.05874169802391295, 0.05802030256472639, 0.05775300243509428, 0.05706838363322536, 0.05704019431961592, 0.05699096363788793, 0.05654429290803043, 0.05652164516594364, 0.05644283212364658, 0.05640651191993924, 0.05591889726488822, 0.05554589629877062, 0.0553352214292412, 0.0551249662902477, 0.05510311685855994], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 161.5950059951283, 'batch_acquisition_elapsed_time': 49.566741762682796})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9224137931034483, 'nll': 1.065177588627256, 'f1': 0.8082908690263049, 'precision': 0.8913682753848815, 'recall': 0.761711972238288, 'ROC_AUC': 0.9218786004597788, 'PRC_AUC': 0.9898103379223607, 'specificity': 0.5384615384615384}, 'chosen_targets': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1], 'chosen_samples': [868, 1253, 899, 1046, 1195, 525, 1025, 1100, 629, 391, 264, 598, 902, 1323, 800, 206, 1190, 361, 617, 969], 'chosen_samples_score': [0.1516737302610044, 0.14412167350034394, 0.1421614919300958, 0.13476456094455186, 0.12479927256577986, 0.1245573789308814, 0.1231164677007988, 0.12061666599004531, 0.11787487775531466, 0.11782494353778795, 0.1175112480989583, 0.1127877807134631, 0.10799614680215314, 0.10751888427064607, 0.10747296582893101, 0.1074259149071953, 0.10736636918185657, 0.10729751681995432, 0.10573778697523016, 0.10535944258743432], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.57050619088113, 'batch_acquisition_elapsed_time': 48.49622029112652})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.9202586206896551, 'nll': 0.3450363093409045, 'f1': 0.8504620798383373, 'precision': 0.8222768484229346, 'recall': 0.8892423366107576, 'ROC_AUC': 0.8928611115895599, 'PRC_AUC': 0.9881659758596955, 'specificity': 0.8461538461538461}, 'chosen_targets': [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'chosen_samples': [825, 86, 576, 329, 769, 688, 793, 757, 655, 6, 661, 1018, 787, 593, 721, 1235, 759, 1042, 211, 343], 'chosen_samples_score': [0.09711399603483273, 0.08515327727861956, 0.08267484885335116, 0.07968283388977768, 0.07927132812928354, 0.0787117326243035, 0.0786714452254712, 0.0776236572775702, 0.07657217589558232, 0.0757189152439309, 0.07501269189926611, 0.07399126166683617, 0.07393779935541817, 0.07376225235659098, 0.07282745253296141, 0.0727160452385193, 0.0725577430778592, 0.07226995497110669, 0.07224266780882632, 0.07223762595307806], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.0227310671471, 'batch_acquisition_elapsed_time': 47.8573804339394})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.9396551724137931, 'nll': 0.35247542940337084, 'f1': 0.873125, 'precision': 0.878122154930135, 'recall': 0.868324657798342, 'ROC_AUC': 0.9547806817490725, 'PRC_AUC': 0.9946732714359533, 'specificity': 0.7692307692307693}, 'chosen_targets': [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], 'chosen_samples': [1089, 423, 565, 1173, 97, 1032, 1276, 1070, 763, 1066, 732, 757, 628, 1230, 222, 372, 1031, 1293, 1323, 1075], 'chosen_samples_score': [0.06334869495627152, 0.06110317852143031, 0.059218898788451345, 0.05721541419769933, 0.0559778857401803, 0.05550813202354429, 0.054986408140467985, 0.05437611921512642, 0.05289049132511245, 0.05146964592003982, 0.05102316007687886, 0.05100917777126254, 0.050604938026398205, 0.049347984501119956, 0.0492309823203293, 0.04887014666083524, 0.04865544530646902, 0.048624826215553654, 0.04849693794103693, 0.04847951647443953], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.62492996267974, 'batch_acquisition_elapsed_time': 47.22362196492031})
