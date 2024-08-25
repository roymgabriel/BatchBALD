store = {}
store['args']={'experiment_description': 'COVID MULTI:RESNET BN DROPOUT VARIATIONAL RATIOS (SCRATCH)', 'batch_size': 16, 'scoring_batch_size': 32, 'test_batch_size': 64, 'validation_set_size': 500, 'early_stopping_patience': 3, 'epochs': 30, 'epoch_samples': 5056, 'num_inference_samples': 100, 'available_sample_k': 20, 'target_num_acquired_samples': 1000, 'target_accuracy': 0.7025, 'quickquick': False, 'seed': 9031, 'log_interval': 20, 'initial_samples_per_class': 25, 'initial_samples': None, 'min_remaining_percentage': 100, 'min_candidates_per_acquired_item': 3, 'initial_percentage': 100, 'reduce_percentage': 0, 'balanced_validation_set': False, 'balanced_test_set': False, 'no_cuda': False, 'experiment_task_id': 'covid_full_resnet_multi_no_mild_scratch_vr_9031', 'experiments_laaos': './experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', 'type': 'AcquisitionFunction.variation_ratios', 'acquisition_method': 'AcquisitionMethod.independent', 'dataset': 'DatasetEnum.covid_multi'}
store['cmdline']=['MIP/BatchBALD/src/run_experiment.py', '--experiment_task_id=covid_full_resnet_multi_no_mild_scratch_vr_9031', '--experiments_laaos=./experiment_configs/EMORY_COVID/scratch_multi_no_mild_config.py', '--dataset=covid_multi', '--type=variation_ratios', '--acquisition_method=independent']
store['Distribution of training set classes:']={2: 812, 1: 586, 0: 225}
store['Distribution of validation set classes:']={2: 113, 1: 83, 0: 36}
store['Distribution of test set classes:']={1: 167, 2: 232, 0: 65}
store['Distribution of pool classes:']={2: 787, 1: 561, 0: 200}
store['Distribution of active set classes:']={0: 25, 2: 25, 1: 25}
store['active samples']=75
store['available samples']=1548
store['validation samples']=232
store['test samples']=464
store['iterations']=[]
store['initial_samples']=[228, 1066, 633, 320, 426, 1411, 1324, 168, 1328, 330, 1391, 390, 105, 28, 1389, 254, 1138, 949, 537, 1223, 1435, 449, 712, 321, 1383, 1314, 1089, 1569, 745, 947, 356, 541, 611, 1275, 302, 806, 366, 851, 1297, 980, 1377, 630, 613, 556, 941, 229, 967, 76, 1596, 131, 1116, 213, 976, 1518, 1151, 448, 24, 1373, 1595, 1209, 65, 1285, 182, 1604, 244, 714, 419, 629, 509, 813, 465, 1510, 798, 484, 718]
store['iterations'].append({'num_epochs': 6, 'test_metrics': {'accuracy': 0.5991379310344828, 'nll': 1.6836763579269935, 'f1': 0.5414802488311504, 'precision': 0.5986356483275634, 'recall': 0.5258236841965935, 'ROC_AUC': 0.72894287109375, 'PRC_AUC': 0.6010343612179333, 'specificity': 0.7588142319691019}, 'chosen_targets': [0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 1], 'chosen_samples': [714, 1407, 1345, 1049, 515, 1151, 963, 794, 499, 780, 526, 1323, 1215, 1123, 790, 1540, 463, 1117, 1445, 473], 'chosen_samples_score': [0.6159021744950165, 0.5953770486121174, 0.589388784122513, 0.5858319396071217, 0.5835299321996541, 0.5566443476395533, 0.5557074655885499, 0.5528219905547167, 0.5471910835343781, 0.5449433540181658, 0.5404476202090764, 0.5403261266329058, 0.5235351181242205, 0.5225573398049752, 0.5221780219572143, 0.5181159706059599, 0.5179857015315956, 0.515424663410309, 0.5088323832255474, 0.5087549425045945], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 241.46800241991878, 'batch_acquisition_elapsed_time': 54.40228961780667})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.5754310344827587, 'nll': 1.4102463557802398, 'f1': 0.5390105654148459, 'precision': 0.530947976127102, 'recall': 0.557894688077426, 'ROC_AUC': 0.7564697265625, 'PRC_AUC': 0.6434118149898277, 'specificity': 0.7732703624264423}, 'chosen_targets': [1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2], 'chosen_samples': [475, 1206, 904, 525, 1345, 1214, 791, 1084, 1411, 982, 435, 1237, 545, 337, 100, 778, 299, 1468, 628, 296], 'chosen_samples_score': [0.640501227748504, 0.634332994300271, 0.6001811382755873, 0.5939053550724788, 0.5909965635470965, 0.5908709659411224, 0.5889649287810483, 0.5806530397111784, 0.5798806609199121, 0.5748515037049994, 0.5716407492838896, 0.5704772048643207, 0.5702123896188267, 0.5669972223278084, 0.5642102310966945, 0.5547881157201564, 0.5528119427875047, 0.5514289801935035, 0.5437897244005634, 0.5403323727550697], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 358.2561255050823, 'batch_acquisition_elapsed_time': 53.59820689773187})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6077586206896551, 'nll': 0.7797237922405374, 'f1': 0.4876875350770364, 'precision': 0.6585283296541575, 'recall': 0.48235094797143113, 'ROC_AUC': 0.76751708984375, 'PRC_AUC': 0.6408802461407496, 'specificity': 0.7642353445438745}, 'chosen_targets': [0, 0, 1, 0, 0, 1, 0, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], 'chosen_samples': [479, 352, 37, 948, 225, 593, 595, 1273, 796, 408, 631, 174, 140, 866, 1424, 1212, 663, 81, 1503, 1290], 'chosen_samples_score': [0.6589100246518522, 0.647707909022225, 0.645958215273158, 0.628029191730263, 0.5937175356337592, 0.5874623633322433, 0.5615390701119447, 0.547091572143318, 0.5154723251602393, 0.5127136830763067, 0.5115748550287098, 0.5103551717988989, 0.5097533937715178, 0.5095908830504849, 0.5079847558475592, 0.5063506001941742, 0.5058209596312329, 0.5053633955435932, 0.5034616782110098, 0.5033966938029197], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.17162944562733, 'batch_acquisition_elapsed_time': 52.95816320506856})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6293103448275862, 'nll': 1.3059033360974541, 'f1': 0.5824202491792009, 'precision': 0.6460824715862752, 'recall': 0.5590239944514155, 'ROC_AUC': 0.790771484375, 'PRC_AUC': 0.6540503163895379, 'specificity': 0.7853282559120429}, 'chosen_targets': [1, 1, 2, 2, 2, 0, 1, 0, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2], 'chosen_samples': [826, 264, 1116, 137, 314, 1113, 916, 1429, 476, 665, 669, 410, 732, 197, 1425, 193, 141, 969, 570, 492], 'chosen_samples_score': [0.6246634507505675, 0.6139308668817971, 0.6014451448083158, 0.5952966355850495, 0.5907304279193745, 0.579492635088771, 0.5676960481222768, 0.5608049938564454, 0.5437152258642513, 0.5421725768757357, 0.5289065025343804, 0.5260077997304323, 0.5245683582674336, 0.5219835088661875, 0.5197793314494858, 0.5189314218783743, 0.5188859894017723, 0.5090802112439063, 0.5075805279974621, 0.5060335888374966], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.20192630495876, 'batch_acquisition_elapsed_time': 52.58824523584917})
store['iterations'].append({'num_epochs': 13, 'test_metrics': {'accuracy': 0.6185344827586207, 'nll': 1.3639755249023438, 'f1': 0.5883967059126295, 'precision': 0.6320140223905915, 'recall': 0.5671650068563139, 'ROC_AUC': 0.78155517578125, 'PRC_AUC': 0.6555314632072138, 'specificity': 0.7821495571041851}, 'chosen_targets': [1, 1, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], 'chosen_samples': [1023, 330, 666, 792, 78, 1160, 1180, 707, 1224, 969, 181, 1317, 1148, 1424, 970, 1128, 924, 998, 276, 1452], 'chosen_samples_score': [0.5715003060186077, 0.5637225777478843, 0.5564992725013447, 0.5457645398039792, 0.5411508681005937, 0.5397557596316811, 0.5203087692049428, 0.5074613136343167, 0.5071340753168522, 0.5051109513227114, 0.5044633541365494, 0.49831521767616405, 0.4964707979846118, 0.4957363295782712, 0.4954714083318771, 0.4949225630579238, 0.4936390316465892, 0.49233733331875673, 0.4912307206936194, 0.48811752923467155], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 517.6997029818594, 'batch_acquisition_elapsed_time': 51.861976590938866})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6099137931034483, 'nll': 1.0433210175612877, 'f1': 0.5906204346439036, 'precision': 0.5867928367928368, 'recall': 0.5972136628599566, 'ROC_AUC': 0.787353515625, 'PRC_AUC': 0.655965117803555, 'specificity': 0.7893178973789984}, 'chosen_targets': [2, 0, 2, 1, 2, 0, 1, 0, 2, 0, 1, 2, 1, 1, 2, 0, 1, 2, 2, 2], 'chosen_samples': [480, 760, 1310, 936, 141, 1255, 145, 1112, 1044, 854, 238, 928, 993, 766, 1032, 1122, 781, 910, 614, 1380], 'chosen_samples_score': [0.6129335841831716, 0.6118009350046658, 0.6099747352052596, 0.6096958967119355, 0.6082497963828608, 0.6075244288752413, 0.6067711039210282, 0.605661683863062, 0.604378350542071, 0.594239630499939, 0.5874348255160948, 0.584660054395373, 0.5832826891090789, 0.5829942343073622, 0.5796156521649857, 0.57927370518622, 0.5742596810295792, 0.5736255548892288, 0.5720659092802649, 0.5694753140824452], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.6996170077473, 'batch_acquisition_elapsed_time': 51.13301342399791})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6271551724137931, 'nll': 1.1783685355350888, 'f1': 0.6140450327523087, 'precision': 0.6318066978444338, 'recall': 0.6032259089248558, 'ROC_AUC': 0.79376220703125, 'PRC_AUC': 0.6547221452692498, 'specificity': 0.7921337361748734}, 'chosen_targets': [2, 1, 1, 0, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 0, 2], 'chosen_samples': [14, 170, 1031, 943, 664, 868, 424, 345, 590, 1165, 865, 959, 727, 1017, 1329, 441, 258, 131, 18, 308], 'chosen_samples_score': [0.6567632945994819, 0.6180698246105341, 0.5994093954929118, 0.5806980730596043, 0.5781660629707999, 0.5755962875324924, 0.5736072361139113, 0.5680414742489693, 0.559658656311945, 0.5486433485269645, 0.544478100706642, 0.5418169858408024, 0.5407806573804425, 0.5357689511679338, 0.53277425603587, 0.53004614706862, 0.5184457780050133, 0.5130384973560171, 0.5074478966200284, 0.5061072962827374], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.8621227219701, 'batch_acquisition_elapsed_time': 50.600128972902894})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6422413793103449, 'nll': 1.183331456677667, 'f1': 0.59182658137882, 'precision': 0.633525420593472, 'recall': 0.5809585073884063, 'ROC_AUC': 0.79833984375, 'PRC_AUC': 0.6678643613869546, 'specificity': 0.7865514535206005}, 'chosen_targets': [1, 0, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2], 'chosen_samples': [227, 140, 1399, 617, 1130, 588, 143, 150, 1024, 531, 1072, 1337, 829, 56, 326, 1060, 1235, 619, 1200, 1044], 'chosen_samples_score': [0.642753394229473, 0.6413705967060574, 0.6227296524228767, 0.6221594175605635, 0.6108242105508692, 0.597224741118128, 0.5925515576928733, 0.5775792405278481, 0.5732919139462159, 0.5560641105850049, 0.5544352620666388, 0.5507438007215195, 0.5474558060177567, 0.545872236335809, 0.5446797726549035, 0.5395023445190151, 0.5379401741900827, 0.5379262988915265, 0.5374900452574864, 0.5354991993592089], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.978781118989, 'batch_acquisition_elapsed_time': 49.685966931283474})
store['iterations'].append({'num_epochs': 8, 'test_metrics': {'accuracy': 0.5797413793103449, 'nll': 1.2260943445666084, 'f1': 0.5695441475136233, 'precision': 0.565324120986038, 'recall': 0.5924236407820963, 'ROC_AUC': 0.77020263671875, 'PRC_AUC': 0.6486675904175175, 'specificity': 0.7811084442209669}, 'chosen_targets': [2, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 2, 1, 1, 2, 2, 2, 2], 'chosen_samples': [433, 756, 413, 534, 864, 334, 421, 1008, 32, 121, 1356, 316, 263, 858, 589, 682, 517, 76, 304, 93], 'chosen_samples_score': [0.6619378714523423, 0.6227823515023615, 0.6222461611071013, 0.6194281190856353, 0.6193518735627355, 0.6159755792499028, 0.6050189827808774, 0.6045041475802682, 0.6043269018678901, 0.6028475153158339, 0.5996403539436597, 0.5985515778618722, 0.5966416830902677, 0.5883858548201746, 0.5848207213915696, 0.5842711491898418, 0.5760753793688773, 0.5751037333590362, 0.5748387787109814, 0.5735707971116237], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 320.3596678399481, 'batch_acquisition_elapsed_time': 49.07830648589879})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5560344827586207, 'nll': 1.0506739780820649, 'f1': 0.46263309516207757, 'precision': 0.6522566446588792, 'recall': 0.46900588742938526, 'ROC_AUC': 0.74383544921875, 'PRC_AUC': 0.5604229266956039, 'specificity': 0.7530216752842281}, 'chosen_targets': [1, 1, 2, 0, 0, 2, 2, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0, 1, 0], 'chosen_samples': [1134, 175, 144, 104, 1020, 1208, 760, 1008, 768, 308, 74, 448, 986, 160, 1076, 627, 1297, 1291, 127, 171], 'chosen_samples_score': [0.6523457999816527, 0.6370162789892649, 0.622320187836281, 0.6214717971519222, 0.6163203481483207, 0.6146735804360048, 0.6134399772165251, 0.6061237260960926, 0.604882658913496, 0.5962669228736127, 0.5955906656147275, 0.5920705214998571, 0.5834406841193085, 0.5820431194835738, 0.5799184423296271, 0.5785743296378623, 0.5754287399451594, 0.574491297096072, 0.57400303254924, 0.5621517323801426], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.62673422601074, 'batch_acquisition_elapsed_time': 48.4178418898955})
store['iterations'].append({'num_epochs': 17, 'test_metrics': {'accuracy': 0.6206896551724138, 'nll': 1.6774886558795798, 'f1': 0.5594725732910316, 'precision': 0.6786243734555115, 'recall': 0.5442136946266618, 'ROC_AUC': 0.80322265625, 'PRC_AUC': 0.6744614203899725, 'specificity': 0.7903824710388534}, 'chosen_targets': [0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2], 'chosen_samples': [285, 671, 1234, 1324, 209, 176, 462, 317, 344, 326, 147, 1187, 698, 890, 1202, 1193, 680, 167, 466, 394], 'chosen_samples_score': [0.49994366807118173, 0.49968193638638736, 0.4995589798174149, 0.4986338698536289, 0.49789372311084523, 0.48769063912582067, 0.48731743290731444, 0.48643850455350945, 0.4812062254775078, 0.4808151099834751, 0.48055514594043913, 0.47392763018349604, 0.4730417243099032, 0.4725757546127084, 0.47198145246219814, 0.4717910967859933, 0.4711299914760919, 0.470126854199845, 0.4689686792388047, 0.46836437700279443], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 676.644428032916, 'batch_acquisition_elapsed_time': 48.18118441104889})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.6056034482758621, 'nll': 0.6406106619999327, 'f1': 0.5250254463303096, 'precision': 0.5462660372791092, 'recall': 0.5582135463820369, 'ROC_AUC': 0.81243896484375, 'PRC_AUC': 0.663396971758929, 'specificity': 0.7694366532842031}, 'chosen_targets': [1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 2], 'chosen_samples': [340, 1188, 103, 976, 1130, 450, 879, 559, 1196, 216, 728, 1099, 1239, 184, 645, 842, 172, 931, 585, 764], 'chosen_samples_score': [0.6201746960112406, 0.6195156195004243, 0.6193948156374284, 0.6151799988149902, 0.6147432844022791, 0.6147140269119309, 0.61466974876145, 0.6120822926272583, 0.6090349905382175, 0.608967162472273, 0.6088593124118532, 0.6085339067644816, 0.6084068272144921, 0.6079262217732184, 0.6078456848373116, 0.6064633328072025, 0.6062973083450758, 0.6058424312549988, 0.6056449327597178, 0.6050116607185662], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.80142814898863, 'batch_acquisition_elapsed_time': 47.87657253816724})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6400862068965517, 'nll': 0.9184777490023909, 'f1': 0.6192909772039005, 'precision': 0.6052625368731563, 'recall': 0.673148927608973, 'ROC_AUC': 0.810302734375, 'PRC_AUC': 0.6931777284069088, 'specificity': 0.8113151216599492}, 'chosen_targets': [1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 2, 2], 'chosen_samples': [1187, 999, 769, 646, 66, 1252, 816, 376, 992, 1138, 780, 424, 838, 517, 108, 690, 136, 269, 307, 179], 'chosen_samples_score': [0.6333795546446572, 0.5941648091496751, 0.5930727690170421, 0.5895446040564896, 0.588407666454142, 0.5862405319515329, 0.5803189378088838, 0.5802915517901002, 0.5769021303946993, 0.5717083757922214, 0.5709895975807429, 0.5708219733460669, 0.570520503924804, 0.56111828251613, 0.5544930709767637, 0.5540178343285735, 0.5511964264431901, 0.5505427948590337, 0.5494623097182986, 0.5463485072387955], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 360.5511725875549, 'batch_acquisition_elapsed_time': 46.307535232044756})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.6422413793103449, 'nll': 0.7635085664946457, 'f1': 0.6030303694974353, 'precision': 0.599482769355168, 'recall': 0.6346521810490424, 'ROC_AUC': 0.826904296875, 'PRC_AUC': 0.7205517226687366, 'specificity': 0.8015303773773224}, 'chosen_targets': [1, 1, 1, 1, 0, 1, 2, 2, 0, 2, 0, 1, 2, 2, 1, 2, 0, 0, 1, 0], 'chosen_samples': [1065, 991, 1017, 365, 615, 161, 2, 112, 224, 962, 57, 168, 1069, 391, 1234, 1000, 972, 731, 1154, 579], 'chosen_samples_score': [0.6516252230445738, 0.6356159699428792, 0.62271841923819, 0.6116975450777895, 0.6047624308071944, 0.5990256294958496, 0.5963559811278845, 0.5957953941286286, 0.588477907204707, 0.5800332804242291, 0.566620402104546, 0.5664139663392729, 0.5655326544889118, 0.5648503928734567, 0.5616953629607467, 0.5577890367094681, 0.5563161277942112, 0.5532136957330951, 0.5516535683801879, 0.5516334610252009], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.0834759892896, 'batch_acquisition_elapsed_time': 45.433355771936476})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.6077586206896551, 'nll': 1.2319859471814385, 'f1': 0.6072288141776759, 'precision': 0.7009640618911371, 'recall': 0.593983650735664, 'ROC_AUC': 0.78814697265625, 'PRC_AUC': 0.6227731247323156, 'specificity': 0.789253443896517}, 'chosen_targets': [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 0, 1, 1, 1, 2, 2, 1], 'chosen_samples': [1215, 981, 694, 79, 877, 221, 923, 375, 186, 1143, 602, 195, 359, 778, 1055, 291, 89, 150, 214, 96], 'chosen_samples_score': [0.581079727227596, 0.5489735361813317, 0.5423646061194232, 0.502783838901037, 0.5025384239315889, 0.5019014234261939, 0.5009008380233716, 0.49736064738670105, 0.49691009401849406, 0.49661461243040217, 0.4963524311099492, 0.4954583330164001, 0.4947239597857047, 0.4936302136648968, 0.49045759980719505, 0.49024972238333686, 0.49023700592729846, 0.4899966958738816, 0.489741896185907, 0.48898553495988073], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 558.4645107169636, 'batch_acquisition_elapsed_time': 44.72727834386751})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6551724137931034, 'nll': 0.7200686356116985, 'f1': 0.6155486404919511, 'precision': 0.6828936656982049, 'recall': 0.589017323443299, 'ROC_AUC': 0.832275390625, 'PRC_AUC': 0.7274161637881643, 'specificity': 0.8001091271659936}, 'chosen_targets': [1, 1, 0, 1, 0, 2, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 2, 0, 0, 1], 'chosen_samples': [993, 841, 470, 766, 800, 945, 676, 491, 887, 559, 261, 340, 257, 1174, 2, 864, 911, 682, 86, 643], 'chosen_samples_score': [0.6477696332936485, 0.6193302150781054, 0.6168237906296141, 0.6134888165781757, 0.597464718067978, 0.5966762293819431, 0.5930465100218134, 0.5786560643905244, 0.5725887125612299, 0.5680045312608416, 0.567343631643707, 0.5640694810120983, 0.5597823676516281, 0.5525860410746091, 0.548140441399326, 0.5417882023852791, 0.540814248114557, 0.5355140098055411, 0.5238926209314727, 0.5227227478513456], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.78402430191636, 'batch_acquisition_elapsed_time': 44.09153552586213})
store['iterations'].append({'num_epochs': 14, 'test_metrics': {'accuracy': 0.6788793103448276, 'nll': 1.0300594198292699, 'f1': 0.6526730138077997, 'precision': 0.7147655643533612, 'recall': 0.6315545301969006, 'ROC_AUC': 0.82525634765625, 'PRC_AUC': 0.7040475145759246, 'specificity': 0.8234329219658924}, 'chosen_targets': [1, 1, 2, 2, 0, 2, 2, 1, 2, 2, 2, 1, 0, 2, 0, 2, 2, 2, 0, 2], 'chosen_samples': [69, 747, 1022, 2, 637, 265, 574, 1027, 430, 358, 1227, 976, 974, 160, 1217, 228, 149, 893, 878, 240], 'chosen_samples_score': [0.5864122369249716, 0.5476254731576577, 0.5298795842143766, 0.5121620675471743, 0.5093666216039936, 0.5027275020040693, 0.5011743510846483, 0.5011452792968993, 0.4990650023789087, 0.4985808784614868, 0.49830297910730914, 0.49758039773084983, 0.4975053156023278, 0.4972610192590924, 0.49696208434293576, 0.49555756835605125, 0.49423233564029334, 0.49410529230739897, 0.49380466472343776, 0.4930311332232422], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 559.1393015477806, 'batch_acquisition_elapsed_time': 43.56194549286738})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6896551724137931, 'nll': 0.5439918123442551, 'f1': 0.660475724991854, 'precision': 0.6881812616535053, 'recall': 0.6570879461236677, 'ROC_AUC': 0.84783935546875, 'PRC_AUC': 0.7416006937285329, 'specificity': 0.8153030899552376}, 'chosen_targets': [1, 2, 1, 1, 1, 0, 0, 2, 0, 2, 0, 1, 2, 1, 1, 2, 1, 1, 1, 2], 'chosen_samples': [642, 285, 513, 452, 317, 67, 1156, 943, 609, 247, 504, 650, 519, 450, 877, 430, 393, 78, 470, 536], 'chosen_samples_score': [0.6473575199166778, 0.6391435392438956, 0.610042748501715, 0.5922424609840073, 0.5893268626118753, 0.5886532277576713, 0.5885821568195677, 0.5836256270751422, 0.5820601557046157, 0.5804918714636768, 0.5783434676702726, 0.5777073350079727, 0.5766613158701004, 0.5763674605339193, 0.5747413679942823, 0.5742133231388111, 0.5737461080680335, 0.5733402099946404, 0.5733388039093745, 0.5692006697376669], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.82358382595703, 'batch_acquisition_elapsed_time': 42.85827064793557})
store['iterations'].append({'num_epochs': 9, 'test_metrics': {'accuracy': 0.6982758620689655, 'nll': 0.6137446041764885, 'f1': 0.6897962512700198, 'precision': 0.7272989493486387, 'recall': 0.6689839154582082, 'ROC_AUC': 0.845703125, 'PRC_AUC': 0.7406048129650945, 'specificity': 0.8297605662245711}, 'chosen_targets': [0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 0, 0, 1], 'chosen_samples': [432, 310, 1180, 84, 506, 148, 527, 704, 1037, 419, 632, 189, 1151, 449, 902, 1137, 178, 314, 24, 985], 'chosen_samples_score': [0.5980853262229977, 0.5905592258268051, 0.5881774725327149, 0.5879939552151883, 0.526513184933642, 0.5251209353410689, 0.5181270317352528, 0.517525716239496, 0.5172264272952473, 0.5140326513428113, 0.5129229706747934, 0.5109994582669594, 0.5102687036023836, 0.5078792374311634, 0.5077611423148837, 0.5061716592797083, 0.5056587082882671, 0.5050753757907015, 0.5033536799876595, 0.5024385354039637], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 359.7101250109263, 'batch_acquisition_elapsed_time': 42.245034790597856})
store['iterations'].append({'num_epochs': 5, 'test_metrics': {'accuracy': 0.5668103448275862, 'nll': 0.5873284833184604, 'f1': 0.6531825662260444, 'precision': 0.4088249937764501, 'recall': 0.5070733863837312, 'ROC_AUC': 0.765869140625, 'PRC_AUC': 0.660958426733913, 'specificity': 0.7214300118111371}, 'chosen_targets': [1, 1, 0, 1, 0, 1, 0, 2, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0, 2, 1], 'chosen_samples': [901, 825, 504, 750, 5, 835, 827, 755, 724, 240, 907, 813, 412, 523, 606, 547, 562, 396, 639, 198], 'chosen_samples_score': [0.6632945698074721, 0.6534305494960317, 0.6529281611185267, 0.6491525286377532, 0.6473240524147463, 0.6466897433596515, 0.6461807095903647, 0.6459482972882601, 0.6458824783990929, 0.6458043446751973, 0.6456786138917703, 0.6448595922596903, 0.6429856983276663, 0.6422291462027746, 0.6418478840389236, 0.6405450927353049, 0.6397857568707566, 0.6393031102469771, 0.638733677350062, 0.6378001583977123], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 201.74899928085506, 'batch_acquisition_elapsed_time': 41.57309846393764})
store['iterations'].append({'num_epochs': 10, 'test_metrics': {'accuracy': 0.709051724137931, 'nll': 0.5197064958769699, 'f1': 0.6943955548381479, 'precision': 0.7157474120324521, 'recall': 0.6872447412866575, 'ROC_AUC': 0.85467529296875, 'PRC_AUC': 0.7686625122043992, 'specificity': 0.8276878338886808}, 'chosen_targets': [2, 0, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1], 'chosen_samples': [1049, 686, 913, 707, 997, 477, 255, 471, 594, 820, 744, 768, 465, 648, 240, 761, 685, 735, 1087, 871], 'chosen_samples_score': [0.6225850931855843, 0.5998657356541779, 0.5931313783895615, 0.591443972534272, 0.5871998397653444, 0.584976037322367, 0.5833841771143323, 0.581856510998306, 0.5806419604747804, 0.5735081851025479, 0.5712692872097316, 0.5710208966801732, 0.5710150154038931, 0.5708065228395497, 0.5681147954856517, 0.5676749970157113, 0.5672864704967374, 0.5653175049180235, 0.5631444877999244, 0.5630532675125719], 'chosen_samples_orignal_score': None, 'train_model_elapsed_time': 400.7515333830379, 'batch_acquisition_elapsed_time': 40.784636843018234})
