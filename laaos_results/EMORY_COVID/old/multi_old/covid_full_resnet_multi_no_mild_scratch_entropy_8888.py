store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT ENTROPY (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 8888, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_entropy_8888', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.entropy_sampling', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_entropy_8888', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=entropy_sampling', '--acquisition_method=independent']
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
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.540948275862069, 'nll': 2.399845780997441, 'f1': 0.42056351437794737, 'precision': 0.4857101615777768, 'recall': 0.44441382487015363, 'ROC_AUC': 0.70758056640625, 'PRC_AUC': 0.5373076093202801, 'specificity': 0.7267418226280901}, 'chosen_targets': [1, 1, 1, 0, 0, 2, 0, 2, 2, 1, 2, 2, 1, 2, 0, 1, 2, 1, 1, 2], 'chosen_samples': [405, 288, 1417, 1503, 430, 37, 210, 1226, 1491, 63, 199, 373, 1185, 1292, 156, 283, 725, 152, 1409, 572], 'chosen_samples_score': [1.08691664083858, 1.075824400309307, 1.0716171897438262, 1.0418806334890025, 1.037621049657802, 1.0361412211530416, 1.0287201850583834, 1.0140844646365001, 0.9991838923451837, 0.9982104929486029, 0.9917467371717867, 0.9913196710828847, 0.9885737114512942, 0.9794572705577893, 0.9749319474333462, 0.9642454974030179, 0.9568827696045938, 0.9559282453326745, 0.9431923884382835, 0.9416746324194469], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 438.0570697761141, 'batch_acquisition_elapsed_time': 54.09596401499584})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5560344827586207, 'nll': 1.3260857154583108, 'f1': 0.5437453183520599, 'precision': 0.5404819594850747, 'recall': 0.5803104665999564, 'ROC_AUC': 0.75372314453125, 'PRC_AUC': 0.6000194510336153, 'specificity': 0.7733258317045371}, 'chosen_targets': [2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 1, 1, 1, 2, 2, 1], 'chosen_samples': [1161, 41, 476, 456, 391, 859, 345, 235, 907, 968, 20, 959, 1082, 1343, 329, 1099, 1527, 263, 1413, 815], 'chosen_samples_score': [1.093045247526663, 1.0927833024290787, 1.0899969281504012, 1.089015862635029, 1.0888858694208996, 1.0862055821068823, 1.0860733347126776, 1.0804932466593076, 1.0796239542917807, 1.0782769935659644, 1.0764251492134542, 1.0758683721178508, 1.0752937301776964, 1.0711908621901736, 1.0682015550655337, 1.0671210404794629, 1.0670200333567457, 1.0604025412660791, 1.0578788914527841, 1.054273409481437], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.65669212397188, 'batch_acquisition_elapsed_time': 53.44425354292616})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.5862068965517241, 'nll': 1.5075056141820447, 'f1': 0.5709816079274207, 'precision': 0.598895572031283, 'recall': 0.5592631712701918, 'ROC_AUC': 0.78436279296875, 'PRC_AUC': 0.6385600292164846, 'specificity': 0.7545682678774027}, 'chosen_targets': [1, 2, 1, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 2, 1, 1], 'chosen_samples': [391, 804, 1044, 1406, 571, 1097, 1367, 1286, 648, 275, 93, 493, 361, 341, 323, 913, 1087, 416, 714, 472], 'chosen_samples_score': [1.0829052704944433, 1.0689442989039049, 1.0643839121687093, 1.064073916466424, 1.057573311978236, 1.051419901658953, 1.036500756956745, 1.026316300511821, 1.0211186597452895, 1.0043372054533908, 0.987885534538985, 0.9860270643895455, 0.9834134648696382, 0.9784298583823715, 0.9781099973178832, 0.9778528365414991, 0.9771931046670896, 0.9647397690598414, 0.961396894278036, 0.9477773860004315], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 437.3731694710441, 'batch_acquisition_elapsed_time': 52.75756189087406})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5775862068965517, 'nll': 0.9821393243197737, 'f1': 0.4617806425493492, 'precision': 0.5633622607921673, 'recall': 0.4604480693784844, 'ROC_AUC': 0.77178955078125, 'PRC_AUC': 0.6239967427045304, 'specificity': 0.7414396172714381}, 'chosen_targets': [0, 0, 1, 2, 1, 0, 1, 1, 2, 0, 2, 1, 0, 2, 1, 1, 0, 0, 2, 0], 'chosen_samples': [1047, 1279, 772, 675, 139, 6, 715, 369, 830, 1158, 1258, 131, 1450, 1154, 473, 114, 304, 109, 1323, 1283], 'chosen_samples_score': [1.0970670607742985, 1.0960070714420391, 1.0929286665305058, 1.0922933730117035, 1.086028256562042, 1.0816148882064966, 1.0791040666621383, 1.0676214961037318, 1.0672995999685297, 1.0602929783041395, 1.0594539631167155, 1.0570031372560322, 1.0556018155102267, 1.0520583702588553, 1.0475107894743583, 1.0431235295959893, 1.0410206368782697, 1.0343500638915915, 1.029005073151523, 1.0282206154202762], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 240.48000487685204, 'batch_acquisition_elapsed_time': 51.98979488806799})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5969827586206896, 'nll': 0.7109459844128839, 'f1': 0.5077908517377837, 'precision': 0.554841889623092, 'recall': 0.5852429094066509, 'ROC_AUC': 0.7886962890625, 'PRC_AUC': 0.6706098082321892, 'specificity': 0.7646169004305725}, 'chosen_targets': [0, 0, 2, 1, 2, 1, 1, 2, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 1, 0], 'chosen_samples': [827, 1156, 702, 353, 164, 1270, 169, 1165, 955, 418, 1093, 86, 1374, 1401, 974, 1417, 1180, 1160, 1343, 908], 'chosen_samples_score': [1.0983031086357073, 1.0982966215584316, 1.0978309101991792, 1.097340551009749, 1.0964563481863316, 1.096337537768675, 1.096039277588205, 1.0959596810481993, 1.0951516111649093, 1.095127416206781, 1.0947352343852415, 1.0933944529399062, 1.092908705686047, 1.0927061430155607, 1.0916680602201447, 1.0910434866763665, 1.0906926064201845, 1.0896327497366907, 1.089242252827987, 1.0891387496761646], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.31317254900932, 'batch_acquisition_elapsed_time': 51.2229635999538})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5883620689655172, 'nll': 0.9302702278926455, 'f1': 0.4730950053530698, 'precision': 0.6990717299578059, 'recall': 0.4769183119172795, 'ROC_AUC': 0.77520751953125, 'PRC_AUC': 0.6346184818070797, 'specificity': 0.7331214443283409}, 'chosen_targets': [0, 2, 1, 0, 0, 2, 0, 1, 1, 2, 1, 0, 1, 0, 2, 1, 1, 1, 2, 1], 'chosen_samples': [1158, 480, 1362, 665, 952, 868, 966, 138, 618, 1349, 1247, 843, 219, 522, 1142, 1334, 643, 1336, 1315, 113], 'chosen_samples_score': [1.095746573288456, 1.0937546567755114, 1.093285080584105, 1.091595557337912, 1.0882056495729902, 1.0877504361462846, 1.087498240309816, 1.086858565148185, 1.0864613226315056, 1.085031949549321, 1.0826051467276852, 1.0819113882819842, 1.0812259929693782, 1.077628978005151, 1.0773737515315776, 1.0766711659837447, 1.075946858377948, 1.0738747036788308, 1.0737181019230428, 1.0736498211605094], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.42997104488313, 'batch_acquisition_elapsed_time': 50.64725387794897})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6573275862068966, 'nll': 1.0841997080835803, 'f1': 0.6542772052755863, 'precision': 0.6376332772166106, 'recall': 0.6821804666529011, 'ROC_AUC': 0.8211669921875, 'PRC_AUC': 0.6970756736350281, 'specificity': 0.8142686333853909}, 'chosen_targets': [2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 1], 'chosen_samples': [1235, 230, 1005, 181, 1027, 1273, 300, 1409, 762, 868, 1097, 975, 1045, 946, 469, 597, 120, 228, 82, 114], 'chosen_samples_score': [1.0970661533234445, 1.0964271870470208, 1.0951359277232098, 1.0950173006502668, 1.0936779219499626, 1.0931765507362048, 1.0908073947299124, 1.0897510549499287, 1.0877960680150573, 1.0877012159285062, 1.0839738272912385, 1.083907305539657, 1.0744517660177797, 1.0730414908301018, 1.0717813785490433, 1.0704320660514124, 1.0694561717146667, 1.0667524313098935, 1.0655670978431304, 1.0640137240500156], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.22435155604035, 'batch_acquisition_elapsed_time': 50.093301850836724})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.6573275862068966, 'nll': 0.9609791328167093, 'f1': 0.6629170578555781, 'precision': 0.7003276787690403, 'recall': 0.6425803565283226, 'ROC_AUC': 0.83929443359375, 'PRC_AUC': 0.7169555080633156, 'specificity': 0.8065495286457175}, 'chosen_targets': [0, 1, 2, 1, 2, 0, 1, 1, 1, 2, 1, 0, 0, 0, 2, 1, 1, 2, 2, 1], 'chosen_samples': [631, 1055, 143, 1258, 289, 538, 809, 1311, 977, 1107, 548, 1310, 1087, 1146, 262, 48, 1138, 857, 332, 295], 'chosen_samples_score': [1.0912611698376025, 1.0732449132799293, 1.0522216625909522, 1.0467312895444791, 1.036212172729242, 1.006296889466078, 1.0056349986254436, 0.995937986796515, 0.9935663982766247, 0.9776194102174969, 0.9680321998682808, 0.9676366875765063, 0.9593329587591306, 0.9538934311604466, 0.9496270692961417, 0.9423936176253136, 0.940797940648392, 0.9396118274993731, 0.9365107001260222, 0.9207935615714724], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 479.6690295659937, 'batch_acquisition_elapsed_time': 49.249048138037324})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5862068965517241, 'nll': 0.8562752953891096, 'f1': 0.5769446026114591, 'precision': 0.5567470532448178, 'recall': 0.6245903418626937, 'ROC_AUC': 0.7978515625, 'PRC_AUC': 0.6752545231905378, 'specificity': 0.77226707777222}, 'chosen_targets': [2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2], 'chosen_samples': [321, 291, 1281, 753, 721, 954, 934, 1122, 1276, 757, 893, 632, 768, 184, 991, 520, 1170, 111, 375, 501], 'chosen_samples_score': [1.0919469938794413, 1.0913343633228592, 1.0907327571730134, 1.089573081181451, 1.0886021047359171, 1.0880631714397468, 1.0879773849146877, 1.0879678277801934, 1.0872100044625166, 1.0868616086904161, 1.0857127460516893, 1.0848904587355495, 1.0848744293720542, 1.0845231255013945, 1.0842097483182487, 1.0829224130289927, 1.0824324746699088, 1.0821703904681221, 1.0816849340804853, 1.0813247449315075], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.51717284182087, 'batch_acquisition_elapsed_time': 48.641565647907555})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5754310344827587, 'nll': 0.7319356984105604, 'f1': 0.4702895450007132, 'precision': 0.5665333129779758, 'recall': 0.5195312293185512, 'ROC_AUC': 0.78155517578125, 'PRC_AUC': 0.6627171130830347, 'specificity': 0.734244215263574}, 'chosen_targets': [1, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0], 'chosen_samples': [214, 992, 551, 763, 148, 1283, 862, 379, 160, 918, 260, 1271, 553, 87, 427, 938, 57, 223, 308, 207], 'chosen_samples_score': [1.0968621640860867, 1.0942650483250969, 1.0936736690285214, 1.0934544554654266, 1.0928410543811988, 1.0920564582762244, 1.091814877432248, 1.0911814883858155, 1.0908239215154887, 1.0906595024422554, 1.0906094532164534, 1.0905974938075098, 1.090351021656814, 1.0902057277831418, 1.0896871543687545, 1.0890295848428053, 1.0889917675841763, 1.0887595532324543, 1.0881083490284926, 1.0880693416431226], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 202.14527235878631, 'batch_acquisition_elapsed_time': 48.18659913120791})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6918103448275862, 'nll': 0.8735842869199556, 'f1': 0.6605921132236922, 'precision': 0.7169749953475781, 'recall': 0.6634475081666905, 'ROC_AUC': 0.83404541015625, 'PRC_AUC': 0.7237069426086027, 'specificity': 0.810478463079794}, 'chosen_targets': [1, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1], 'chosen_samples': [594, 36, 1140, 58, 65, 751, 718, 253, 953, 1059, 1112, 1322, 720, 1054, 415, 878, 477, 1330, 430, 1227], 'chosen_samples_score': [1.0402558181398351, 0.9933633332899743, 0.9562955451074747, 0.9467785299641364, 0.9440991067778839, 0.9272509637385098, 0.9193625709454166, 0.9158038601599581, 0.911566677637696, 0.9110592920486685, 0.9110117556917299, 0.9066360495317627, 0.9062621089061091, 0.9061761099919596, 0.905135897675027, 0.902729852247118, 0.902302226487852, 0.9011427940023403, 0.899131339153965, 0.894950935484903], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.32739133480936, 'batch_acquisition_elapsed_time': 47.34055743087083})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.646551724137931, 'nll': 0.7385977383317619, 'f1': 0.6071433524522769, 'precision': 0.6672348484848486, 'recall': 0.5941239536841437, 'ROC_AUC': 0.8226318359375, 'PRC_AUC': 0.706757688987264, 'specificity': 0.783930175490974}, 'chosen_targets': [2, 1, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 2, 1, 1], 'chosen_samples': [1254, 1029, 813, 1138, 512, 1237, 1123, 303, 1080, 1089, 1245, 332, 773, 712, 1275, 503, 975, 127, 577, 31], 'chosen_samples_score': [1.0671543639391694, 1.026868284458478, 1.0228881194857704, 1.0207861732541077, 0.9965801304243749, 0.9931158631317762, 0.9929594837062621, 0.9926509875028264, 0.9886792727667848, 0.9858469872772178, 0.9817557616224076, 0.9762363291060006, 0.9674158035130253, 0.9666880861484849, 0.9664500268526631, 0.9653273663533346, 0.9653231041551862, 0.9638958084236091, 0.9638522977271831, 0.9562782955754947], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 519.4384469427168, 'batch_acquisition_elapsed_time': 46.72695800801739})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6185344827586207, 'nll': 0.8038935168036099, 'f1': 0.6284121441909777, 'precision': 0.6311193180758399, 'recall': 0.6366299231775177, 'ROC_AUC': 0.819091796875, 'PRC_AUC': 0.6993470965189497, 'specificity': 0.7934536867537473}, 'chosen_targets': [2, 0, 1, 2, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1], 'chosen_samples': [23, 845, 122, 1233, 391, 1126, 1012, 307, 1238, 114, 159, 378, 437, 857, 868, 643, 97, 1064, 983, 584], 'chosen_samples_score': [1.084365910626015, 1.0803340937107533, 1.0773655105917999, 1.0742264016093106, 1.072888600317311, 1.0725996825824209, 1.0688959128141813, 1.0658250682327597, 1.065567327675179, 1.0638269074715259, 1.0625957077818586, 1.0615115800856996, 1.059694124278229, 1.059350262924097, 1.0584589490724836, 1.0558044903308486, 1.0539584741532042, 1.049873201497403, 1.0489775265940402, 1.0471968019021693], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.4488366479054, 'batch_acquisition_elapsed_time': 45.780214596074075})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6573275862068966, 'nll': 0.653811553428913, 'f1': 0.6588934503206418, 'precision': 0.6770019655682088, 'recall': 0.6499658507917851, 'ROC_AUC': 0.852783203125, 'PRC_AUC': 0.753216358147136, 'specificity': 0.7970059756962359}, 'chosen_targets': [0, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0, 0, 1, 2, 2, 1], 'chosen_samples': [800, 437, 426, 785, 1244, 751, 496, 1094, 8, 1208, 636, 1128, 334, 1114, 1259, 917, 397, 1141, 578, 1234], 'chosen_samples_score': [1.0953386648307246, 1.0945719144517394, 1.0938683081893616, 1.0910553195824122, 1.085853110521823, 1.0804176881219778, 1.0785267043974338, 1.0782587745013008, 1.0693936838789047, 1.0674207338478758, 1.0667620334375572, 1.0666069145375952, 1.0654018529175264, 1.0620023629134965, 1.0610010902573415, 1.060810264035838, 1.0557511930128853, 1.0556097018644746, 1.052235505549946, 1.050026340515947], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 399.6891791988164, 'batch_acquisition_elapsed_time': 45.471760435029864})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.6379310344827587, 'nll': 0.8218852076037176, 'f1': 0.6508807967631497, 'precision': 0.6701326757054833, 'recall': 0.6365858468738915, 'ROC_AUC': 0.8106689453125, 'PRC_AUC': 0.6881105760034766, 'specificity': 0.7932214432516913}, 'chosen_targets': [0, 1, 2, 0, 2, 2, 1, 1, 1, 1, 2, 0, 2, 2, 1, 2, 1, 2, 1, 1], 'chosen_samples': [533, 1036, 182, 654, 271, 218, 205, 811, 987, 997, 48, 1117, 771, 172, 31, 54, 598, 282, 1131, 97], 'chosen_samples_score': [1.0778776769154415, 1.0726686573572053, 1.0671652913907583, 1.064078237657653, 1.0605758750679446, 1.0569834165715806, 1.0548426516574754, 1.0480411790544526, 1.0431147993484375, 1.0429550072040739, 1.0413667282312709, 1.0391021138125187, 1.0335214517455427, 1.0317921362606377, 1.0240990138676647, 1.0198896237895942, 1.0193532857556953, 1.015762191554939, 1.0087480760472367, 1.008611684679515], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 518.84646368213, 'batch_acquisition_elapsed_time': 44.46993264509365})
store['iterations'].append({'num_epochs': 15, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 0.9383562022242052, 'f1': 0.5995380264019623, 'precision': 0.6520077406869859, 'recall': 0.575021045442272, 'ROC_AUC': 0.80096435546875, 'PRC_AUC': 0.6704795602956783, 'specificity': 0.7834415729908772}, 'chosen_targets': [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1], 'chosen_samples': [829, 123, 754, 1152, 968, 762, 63, 278, 837, 676, 244, 496, 1190, 256, 690, 1036, 2, 337, 528, 1168], 'chosen_samples_score': [1.0954456422154593, 1.0757304761715665, 1.0604062971663037, 1.0594224883292527, 1.052046894024741, 1.0505325921342652, 1.0420183631451185, 1.040919678836572, 1.0399680900446477, 1.0354606897574432, 1.0240490893845706, 1.02164518034038, 1.0176203134569017, 1.0139373280680057, 1.0137313310715832, 1.0110501825401765, 1.0075589684927027, 1.0051698147065617, 1.0046104145914785, 1.0034948198652236], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 598.9941104003228, 'batch_acquisition_elapsed_time': 43.75589695805684})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6508620689655172, 'nll': 0.5913763375117861, 'f1': 0.6005656872678149, 'precision': 0.655863453815261, 'recall': 0.6360384006522763, 'ROC_AUC': 0.8260498046875, 'PRC_AUC': 0.6912188508116055, 'specificity': 0.7888512279770598}, 'chosen_targets': [2, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1], 'chosen_samples': [487, 1120, 635, 437, 231, 131, 294, 232, 306, 494, 22, 643, 1139, 1156, 1047, 1138, 1033, 248, 365, 80], 'chosen_samples_score': [1.0847047007249249, 1.0846848201214805, 1.0840469016926964, 1.084034906335613, 1.0831361919153193, 1.0831316303421064, 1.083087768086552, 1.0825917669992515, 1.0822367629709224, 1.0819575065823779, 1.081921745073541, 1.081224650360952, 1.0811445219356002, 1.0804981880808682, 1.0802371784165583, 1.080020835660574, 1.0799524205797117, 1.0799394416204167, 1.0796647230163783, 1.0795749258521163], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.2774019530043, 'batch_acquisition_elapsed_time': 43.086291359271854})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6681034482758621, 'nll': 0.8602315968480604, 'f1': 0.6658438611919419, 'precision': 0.7297957052701326, 'recall': 0.6456238716201549, 'ROC_AUC': 0.82855224609375, 'PRC_AUC': 0.7123414893986483, 'specificity': 0.8175618717058524}, 'chosen_targets': [1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1], 'chosen_samples': [411, 657, 1185, 557, 318, 12, 426, 1057, 151, 116, 1044, 868, 419, 715, 266, 250, 596, 751, 636, 338], 'chosen_samples_score': [1.0601734946978452, 1.0513817936803906, 1.0466538954872373, 1.0456865213553082, 1.0371824107987972, 1.0371297970063473, 1.0364901230467884, 1.0345733705556566, 1.030439286764905, 1.0303131119216307, 1.0286571515043492, 1.0174152175762794, 1.0125226238254934, 1.010378617893492, 1.009590122080549, 1.0059808274888233, 1.0024870201299267, 0.9978951334687789, 0.9922794234926922, 0.9903722387791689], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 321.2159951943904, 'batch_acquisition_elapsed_time': 42.24746052129194})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.665948275862069, 'nll': 0.5982118146172886, 'f1': 0.6353974155022707, 'precision': 0.6352342904702456, 'recall': 0.6460932246912012, 'ROC_AUC': 0.84326171875, 'PRC_AUC': 0.7274204619795025, 'specificity': 0.811117069057178}, 'chosen_targets': [1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1], 'chosen_samples': [1022, 531, 193, 943, 1160, 102, 38, 601, 410, 492, 803, 1036, 1184, 269, 86, 753, 500, 873, 471, 239], 'chosen_samples_score': [1.0983914346829435, 1.096869779098529, 1.0944075301942158, 1.090258304911499, 1.0883021188542972, 1.0878035680933493, 1.0870258744211476, 1.0867446900440632, 1.0864893758331162, 1.0859917427214705, 1.085393715709429, 1.085069686814416, 1.0847740457731296, 1.0838442147814373, 1.0837068015535456, 1.0832314134543735, 1.083204423845945, 1.082900570067248, 1.082721915017466, 1.082482329539392], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.63600876228884, 'batch_acquisition_elapsed_time': 41.7593083512038})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5560344827586207, 'nll': 0.678683379600788, 'f1': 0.4648783530149636, 'precision': 0.47518497114635716, 'recall': 0.528748735949851, 'ROC_AUC': 0.77813720703125, 'PRC_AUC': 0.636695655611759, 'specificity': 0.7350502839008586}, 'chosen_targets': [2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 1, 2], 'chosen_samples': [855, 143, 205, 64, 132, 121, 166, 776, 810, 139, 320, 430, 633, 599, 1023, 1133, 747, 832, 283, 314], 'chosen_samples_score': [1.0977275399174435, 1.0972758370302516, 1.097023426380325, 1.0967241335004225, 1.09615624221827, 1.0960821724391094, 1.096075704211825, 1.0957095661140797, 1.09526053066955, 1.0952182939013508, 1.0952116873632438, 1.094956660345729, 1.0948407636375357, 1.094725042119794, 1.0946405397962726, 1.0946280864882603, 1.0945433160032607, 1.0944077292658412, 1.0944017208588277, 1.0941908450617814], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 242.2840722273104, 'batch_acquisition_elapsed_time': 41.20254583610222})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.6056034482758621, 'nll': 0.6183722923541891, 'f1': 0.5560668779091791, 'precision': 0.613311157796452, 'recall': 0.5450126272653631, 'ROC_AUC': 0.8099365234375, 'PRC_AUC': 0.6660193000287926, 'specificity': 0.7606640687402937}, 'chosen_targets': [1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2], 'chosen_samples': [1114, 804, 546, 77, 410, 553, 1010, 991, 881, 493, 453, 519, 51, 469, 123, 295, 569, 937, 873, 701], 'chosen_samples_score': [1.0969342355992513, 1.095963767672741, 1.094927083864031, 1.0937679934456501, 1.0884473360505837, 1.0849176441623019, 1.0847761318374363, 1.0846685244605103, 1.0808054005073606, 1.08002265972763, 1.0799642972108434, 1.07650692894312, 1.07650486298403, 1.0738799064493412, 1.0730206510205955, 1.0697153803510235, 1.0695844225245201, 1.06558022028906, 1.0653702537980099, 1.0645561128176162], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 321.5852158968337, 'batch_acquisition_elapsed_time': 40.32596398284659})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5064655172413793, 'nll': 0.7506429080305428, 'f1': 0.36419027083537525, 'precision': 0.3085054080629302, 'recall': 0.3404398100351022, 'ROC_AUC': 0.70001220703125, 'PRC_AUC': 0.5617851628676334, 'specificity': 0.6731781415689461}, 'chosen_targets': [1, 0, 1, 2, 0, 2, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2, 1, 1, 2], 'chosen_samples': [551, 57, 436, 969, 755, 346, 279, 744, 686, 946, 586, 27, 1025, 737, 774, 1046, 898, 90, 111, 262], 'chosen_samples_score': [1.0932517135779574, 1.0924640846987947, 1.092227344562334, 1.0913468889901687, 1.0908735675242704, 1.090308303315651, 1.0869324909812625, 1.0861787642956542, 1.0824531008269898, 1.0807455008449005, 1.0805288997170952, 1.0804964882346657, 1.0797648056320521, 1.079023497669298, 1.0782078167706008, 1.076362578050596, 1.0757960485296394, 1.0755778808512582, 1.0755743634426045, 1.0754449869848342], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.3687136261724, 'batch_acquisition_elapsed_time': 40.48141575511545})
store['iterations'].append({'num_epochs': 15, 'test_metrics': {'accuracy': 0.6228448275862069, 'nll': 0.8343549925705482, 'f1': 0.6187993507070932, 'precision': 0.6169144141875124, 'recall': 0.6313408991036494, 'ROC_AUC': 0.82037353515625, 'PRC_AUC': 0.6899698106185456, 'specificity': 0.7812613211887259}, 'chosen_targets': [2, 2, 0, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 1, 1], 'chosen_samples': [82, 65, 825, 685, 903, 439, 483, 660, 607, 912, 101, 649, 932, 585, 391, 622, 178, 640, 259, 279], 'chosen_samples_score': [1.0939135581461183, 1.055363681564593, 1.0405957960462957, 1.0324818854819613, 1.0107349676901813, 1.0016870133195384, 1.001386505476713, 0.9955336912020354, 0.993366827742431, 0.9931042919888116, 0.9898936565844246, 0.9893359891069753, 0.9794516164859831, 0.9755786879614499, 0.9702873433531432, 0.9689406152105446, 0.9545317863834721, 0.9544321681405902, 0.9498802853760646, 0.9498171257295343], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 599.5004884097725, 'batch_acquisition_elapsed_time': 38.99519067676738})
store['iterations'].append({'num_epochs': 7, 'test_metrics': {'accuracy': 0.540948275862069, 'nll': 0.7914889105435076, 'f1': 0.4441195292295587, 'precision': 0.5947568177339232, 'recall': 0.43691542644154663, 'ROC_AUC': 0.75311279296875, 'PRC_AUC': 0.6310003500353718, 'specificity': 0.7284721003716769}, 'chosen_targets': [1, 1, 1, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 0], 'chosen_samples': [89, 1025, 13, 216, 350, 289, 451, 621, 74, 314, 342, 712, 307, 884, 1064, 148, 430, 703, 616, 1054], 'chosen_samples_score': [1.098571179038717, 1.0985578492965864, 1.0985541294482, 1.0985044963243635, 1.0984082759469918, 1.0970268326208943, 1.0968566232973218, 1.0968319381631952, 1.0958310779444338, 1.0958109313106192, 1.0957447556270374, 1.0955813416333473, 1.0953024519970203, 1.0947448641831703, 1.0942723774387382, 1.0942296578840496, 1.093838209354352, 1.0937073169126763, 1.093612132343353, 1.093422010345288], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 281.2115244027227, 'batch_acquisition_elapsed_time': 38.559952777810395})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.5948275862068966, 'nll': 0.8139906916125067, 'f1': 0.5792601255558513, 'precision': 0.566473138328149, 'recall': 0.6105360896244646, 'ROC_AUC': 0.80841064453125, 'PRC_AUC': 0.6597582870650297, 'specificity': 0.7881982181740197}, 'chosen_targets': [2, 1, 1, 1, 0, 1, 0, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1], 'chosen_samples': [147, 263, 670, 95, 518, 131, 729, 640, 33, 112, 337, 999, 28, 156, 275, 818, 936, 163, 588, 583], 'chosen_samples_score': [1.096432272735504, 1.092204323325653, 1.0920786320363114, 1.087681897073558, 1.0870766486060972, 1.0864587773398577, 1.0862602647535093, 1.082451730350326, 1.0815263269689115, 1.0810858221633262, 1.0804650622979208, 1.0804130068882354, 1.0798116146799657, 1.079778730548552, 1.0791380490603437, 1.079053256644301, 1.078984785148497, 1.078036183136664, 1.0775599075614042, 1.076679255760435], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 361.02925995737314, 'batch_acquisition_elapsed_time': 37.603182347957045})
store['iterations'].append({'num_epochs': 12, 'test_metrics': {'accuracy': 0.6918103448275862, 'nll': 0.6467029966157058, 'f1': 0.6969199429895113, 'precision': 0.7017436791630342, 'recall': 0.6939817976778538, 'ROC_AUC': 0.8623046875, 'PRC_AUC': 0.7646451793742229, 'specificity': 0.8284019508308679}, 'chosen_targets': [0, 0, 2, 2, 0, 0, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2], 'chosen_samples': [303, 827, 828, 291, 4, 584, 25, 445, 649, 1018, 203, 40, 658, 659, 251, 279, 188, 263, 111, 541], 'chosen_samples_score': [1.0790425676554571, 1.0650834057185017, 1.0304177109910169, 1.0142946358461173, 1.004914779710283, 1.001103282390139, 1.000755201661039, 0.9988341290113203, 0.9964202996725309, 0.9915918360691878, 0.986709128376551, 0.9854433700883016, 0.9805425778693257, 0.9717373848576776, 0.9649361992685759, 0.9637077262249807, 0.9630141622575686, 0.961680333768562, 0.9611217936007381, 0.9525956054663274], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 481.1805082769133, 'batch_acquisition_elapsed_time': 37.11096000717953})
store['iterations'].append({'num_epochs': 17, 'test_metrics': {'accuracy': 0.6982758620689655, 'nll': 0.5940454417261584, 'f1': 0.6922559877129713, 'precision': 0.7126732835269581, 'recall': 0.6770388930362087, 'ROC_AUC': 0.8779296875, 'PRC_AUC': 0.7856234678458315, 'specificity': 0.8268479380850827}, 'chosen_targets': [1, 2, 1, 0, 2, 1, 2, 0, 1, 1, 0, 0, 2, 1, 2, 1, 2, 1, 1, 0], 'chosen_samples': [883, 992, 514, 629, 271, 633, 772, 100, 16, 12, 886, 683, 938, 686, 822, 505, 455, 196, 596, 217], 'chosen_samples_score': [1.060187999142347, 1.0196243518094361, 1.016724078362197, 1.0121861089757465, 0.9937438202426292, 0.9907501934893297, 0.9894682521213105, 0.9791198697322023, 0.9657726092781231, 0.9596295507398813, 0.9576624988366447, 0.9453572382030748, 0.9414041164127566, 0.930430187932878, 0.9279804744420184, 0.9276595441984072, 0.9258355542495882, 0.9247633563301615, 0.9227040835225784, 0.9193038373958827], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 677.7211130480282, 'batch_acquisition_elapsed_time': 36.2609818726778})
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5344827586206896, 'nll': 0.8936081590323612, 'f1': 0.4507986266606957, 'precision': 0.3797161647628937, 'recall': 0.36862481932686353, 'ROC_AUC': 0.73101806640625, 'PRC_AUC': 0.6041857744313894, 'specificity': 0.6940574325631798}, 'chosen_targets': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'chosen_samples': [858, 629, 266, 64, 691, 921, 535, 525, 537, 693, 811, 672, 798, 58, 124, 420, 602, 976, 637, 945], 'chosen_samples_score': [1.098092944817669, 1.0956902917435267, 1.0954743967714502, 1.0937229995353817, 1.0937170920706116, 1.0919762083147917, 1.0904729176459504, 1.0872553912692415, 1.0812470942004415, 1.0805770594543187, 1.0742115507974397, 1.0696348814653573, 1.065708816288109, 1.0653337867730848, 1.0616467884217173, 1.0575115073900536, 1.056980612617001, 1.0517489237479727, 1.0492798932013685, 1.038637625294593], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.68541326513514, 'batch_acquisition_elapsed_time': 35.755336694885045})
store['iterations'].append({'num_epochs': 17, 'test_metrics': {'accuracy': 0.6896551724137931, 'nll': 0.6331881819100216, 'f1': 0.6911044973544973, 'precision': 0.7083048388316092, 'recall': 0.6800162274919656, 'ROC_AUC': 0.88629150390625, 'PRC_AUC': 0.8003567361537227, 'specificity': 0.8264188786723573}, 'chosen_targets': [2, 1, 0, 2, 0, 2, 2, 0, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2], 'chosen_samples': [845, 109, 196, 978, 539, 11, 883, 473, 133, 567, 413, 543, 561, 836, 690, 123, 678, 25, 796, 180], 'chosen_samples_score': [1.0928900559773147, 1.0496660920871073, 1.0247519740830793, 1.0243843770580918, 1.00675875730483, 1.0011451044211654, 0.9668703269581587, 0.9634737439069468, 0.9415279424080711, 0.9366623000037948, 0.9338869231061377, 0.9241493718923266, 0.9127071364489783, 0.909635425549306, 0.90489373779211, 0.8680028701800881, 0.863577057265529, 0.8522116617184416, 0.8515224423846801, 0.8478790371897937], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 678.0565263410099, 'batch_acquisition_elapsed_time': 34.983139134943485})
store['iterations'].append({'num_epochs': 4, 'test_metrics': {'accuracy': 0.5, 'nll': 0.7534513802363955, 'f1': 0.6666666666666666, 'precision': 0.16666666666666666, 'recall': 0.3333333333333333, 'ROC_AUC': 0.69720458984375, 'PRC_AUC': 0.5457120813558971, 'specificity': 0.6666666666666666}, 'chosen_targets': [2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2, 2], 'chosen_samples': [824, 848, 80, 815, 352, 958, 881, 819, 11, 379, 601, 437, 855, 592, 346, 419, 6, 766, 375, 736], 'chosen_samples_score': [1.0410372979869824, 1.0389219227333963, 1.0317263671806876, 1.0268008530269863, 1.0241011701850917, 1.0238131853287755, 1.0163364964185566, 1.0159487842508597, 1.013981394545425, 1.0076422553428435, 1.006652525192129, 1.006081713852191, 1.0059396959826117, 1.0041341488263156, 1.0011375932865525, 1.0010577286774909, 1.0004484480240043, 0.993052233749594, 0.9928963983480755, 0.9918468918461623], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 162.08087180322036, 'batch_acquisition_elapsed_time': 34.37887397687882})
store['iterations'].append({'num_epochs': 11, 'test_metrics': {'accuracy': 0.7112068965517241, 'nll': 0.5092635976857153, 'f1': 0.709247461220551, 'precision': 0.742688996173615, 'recall': 0.6871141007110447, 'ROC_AUC': 0.8778076171875, 'PRC_AUC': 0.7645589916930672, 'specificity': 0.831204040520435}, 'chosen_targets': [1, 0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'chosen_samples': [731, 783, 259, 409, 880, 689, 819, 398, 641, 250, 902, 929, 600, 141, 50, 452, 456, 927, 333, 590], 'chosen_samples_score': [1.0110623392730924, 0.9948765439904195, 0.9893979888272452, 0.9591907727757438, 0.9531321713273853, 0.9444731893290552, 0.9351479561528315, 0.9254777359780076, 0.9106587162426873, 0.9105668053759659, 0.8901490113268546, 0.8859202605097077, 0.885567151146901, 0.8846138766978436, 0.8844411448117415, 0.8825058232819483, 0.8818856110776482, 0.8777029388367545, 0.8773238088091455, 0.8762004471655024], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 440.7506062728353, 'batch_acquisition_elapsed_time': 33.60802202206105})
