periods={'tick': 1, 'min': 2, '5min': 3, '10min': 4, '15min': 5, '30min': 6, 'hour': 7, 'daily': 8, 'week': 9, 'month': 10}
#print ("ticker="+ticker+"; period="+str(period)+"; start="+start+"; end="+end)
#каждой акции Финам присвоил цифровой код:
ticker2finam={'ABRD':82460,'AESL':181867,'AFKS':19715,'AFLT':29,'AGRO':399716,'AKRN':17564,'ALBK':82616,'ALNU':81882,'ALRS':81820,'AMEZ':20702,'APTK':13855,'AQUA':35238,'ARMD':19676,'ARSA':19915,'ASSB':16452,'AVAN':82843,'AVAZ':39,'AVAZP':40,'BANE':81757,'BANEP':81758,'BGDE':175840,'BISV':35242,'BISVP':35243,'BLNG':21078,'BRZL':81901,'BSPB':20066,'CBOM':420694,'CHEP':20999,'CHGZ':81933,'CHKZ':21000,'CHMF':16136,'CHMK':21001,'CHZN':19960,'CLSB':16712,'CLSBP':16713,'CNTL':21002,'CNTLP':81575,'DASB':16825,'DGBZ':17919,'DIOD':35363,'DIXY':18564,'DVEC':19724,'DZRD':74744,'DZRDP':74745,'ELTZ':81934,'ENRU':16440,'EPLN':451471,'ERCO':81935,'FEES':20509,'FESH':20708,'FORTP':82164,'GAZA':81997,'GAZAP':81998,'GAZC':81398,'GAZP':16842,'GAZS':81399,'GAZT':82115,'GCHE':20125,'GMKN':795,'GRAZ':16610,'GRNT':449114,'GTLC':152876,'GTPR':175842,'GTSS':436120,'HALS':17698,'HIMC':81939,'HIMCP':81940,'HYDR':20266,'IDJT':388276,'IDVP':409486,'IGST':81885,'IGST03':81886,'IGSTP':81887,'IRAO':20516,'IRGZ':9,'IRKT':15547,'ISKJ':17137,'JNOS':15722,'JNOSP':15723,'KAZT':81941,'KAZTP':81942,'KBSB':19916,'KBTK':35285,'KCHE':20030,'KCHEP':20498,'KGKC':83261,'KGKCP':152350,'KLSB':16329,'KMAZ':15544,'KMEZ':22525,'KMTZ':81903,'KOGK':20710,'KRKN':81891,'KRKNP':81892,'KRKO':81905,'KRKOP':81906,'KROT':510,'KROTP':511,'KRSB':20912,'KRSBP':20913,'KRSG':15518,'KSGR':75094,'KTSB':16284,'KTSBP':16285,'KUBE':522,'KUNF':81943,'KUZB':83165,'KZMS':17359,'KZOS':81856,'KZOSP':81857,'LIFE':74584,'LKOH':8,'LNTA':385792,'LNZL':21004,'LNZLP':22094,'LPSB':16276,'LSNG':31,'LSNGP':542,'LSRG':19736,'LVHK':152517,'MAGE':74562,'MAGEP':74563,'MAGN':16782,'MERF':20947,'MFGS':30,'MFGSP':51,'MFON':152516,'MGNT':17086,'MGNZ':20892,'MGTS':12984,'MGTSP':12983,'MGVM':81829,'MISB':16330,'MISBP':16331,'MNFD':80390,'MOBB':82890,'MOEX':152798,'MORI':81944,'MOTZ':21116,'MRKC':20235,'MRKK':20412,'MRKP':20107,'MRKS':20346,'MRKU':20402,'MRKV':20286,'MRKY':20681,'MRKZ':20309,'MRSB':16359,'MSNG':6,'MSRS':16917,'MSST':152676,'MSTT':74549,'MTLR':21018,'MTLRP':80745,'MTSS':15523,'MUGS':81945,'MUGSP':81946,'MVID':19737,'NAUK':81992,'NFAZ':81287,'NKHP':450432,'NKNC':20100,'NKNCP':20101,'NKSH':81947,'NLMK':17046,'NMTP':19629,'NNSB':16615,'NNSBP':16616,'NPOF':81858,'NSVZ':81929,'NVTK':17370,'ODVA':20737,'OFCB':80728,'OGKB':18684,'OMSH':22891,'OMZZP':15844,'OPIN':20711,'OSMP':21006,'OTCP':407627,'PAZA':81896,'PHOR':81114,'PHST':19717,'PIKK':18654,'PLSM':81241,'PLZL':17123,'PMSB':16908,'PMSBP':16909,'POLY':175924,'PRFN':83121,'PRIM':17850,'PRIN':22806,'PRMB':80818,'PRTK':35247,'PSBR':152320,'QIWI':181610,'RASP':17713,'RBCM':74779,'RDRB':181755,'RGSS':181934,'RKKE':20321,'RLMN':152677,'RLMNP':388313,'RNAV':66644,'RODNP':66693,'ROLO':181316,'ROSB':16866,'ROSN':17273,'ROST':20637,'RSTI':20971,'RSTIP':20972,'RTGZ':152397,'RTKM':7,'RTKMP':15,'RTSB':16783,'RTSBP':16784,'RUAL':414279,'RUALR':74718,'RUGR':66893,'RUSI':81786,'RUSP':20712,'RZSB':16455,'SAGO':445,'SAGOP':70,'SARE':11,'SAREP':24,'SBER':3,'SBERP':23,'SELG':81360,'SELGP':82610,'SELL':21166,'SIBG':436091,'SIBN':2,'SKYC':83122,'SNGS':4,'SNGSP':13,'STSB':20087,'STSBP':20088,'SVAV':16080,'SYNG':19651,'SZPR':22401,'TAER':80593,'TANL':81914,'TANLP':81915,'TASB':16265,'TASBP':16266,'TATN':825,'TATNP':826,'TGKA':18382,'TGKB':17597,'TGKBP':18189,'TGKD':18310,'TGKDP':18391,'TGKN':18176,'TGKO':81899,'TNSE':420644,'TORS':16797,'TORSP':16798,'TRCN':74561,'TRMK':18441,'TRNFP':1012,'TTLK':18371,'TUCH':74746,'TUZA':20716,'UCSS':175781,'UKUZ':20717,'UNAC':22843,'UNKL':82493,'UPRO':18584,'URFD':75124,'URKA':19623,'URKZ':82611,'USBN':81953,'UTAR':15522,'UTII':81040,'UTSY':419504,'UWGN':414560,'VDSB':16352,'VGSB':16456,'VGSBP':16457,'VJGZ':81954,'VJGZP':81955,'VLHZ':17257,'VRAO':20958,'VRAOP':20959,'VRSB':16546,'VRSBP':16547,'VSMO':15965,'VSYD':83251,'VSYDP':83252,'VTBR':19043,'VTGK':19632,'VTRS':82886,'VZRZ':17068,'VZRZP':17067,'WTCM':19095,'WTCMP':19096,'YAKG':81917,'YKEN':81766,'YKENP':81769,'YNDX':388383,'YRSB':16342,'YRSBP':16343,'ZHIV':181674,'ZILL':81918,'ZMZN':556,'ZMZNP':603,'ZVEZ':82001}


ibkr_tickers = ['AFKS','AFLT','ALRS','CBOM','CHMF','DSKY','FEES','FIVE','GAZP','GMKN','HYDR','IRAO',
'LKOH','MAGN','MGNT','MOEX','MTLR','MTSS','MVID','NLMK','NVTK','PHOR','PIKK','PLZL','POLY','RNFT',
'ROSN','RTKM','RUAL','SBER','SBERP','SFIN','SNGS','SNGSP','TATN','TATNP','TGKD','TRMK','TRNFP','VTBR','YNDX']



ibkr_info = {'AFKS':{'precise':0.001, 'min_quantity':100},
             'AFLT':{'precise':0.02, 'min_quantity':10},
             'ALRS':{'precise':0.01, 'min_quantity':10},
             'CBOM':{'precise':0.001, 'min_quantity':100},
             'CHMF':{'precise':0.2, 'min_quantity':10},
             'DSKY':{'precise':0.02, 'min_quantity':10},
             'FEES':{'precise':0.00002, 'min_quantity':10000},
             'FIVE':{'precise':0.5, 'min_quantity':10},
             'GAZP':{'precise':0.01, 'min_quantity':10},
             'GMKN':{'precise':2.0, 'min_quantity':1},
             'HYDR':{'precise':0.0001, 'min_quantity':1000},
             'IRAO':{'precise':0.0005, 'min_quantity':1000},
             'LKOH':{'precise':0.5, 'min_quantity':1},
             'MAGN':{'precise':0.005, 'min_quantity':100},
             'MGNT':{'precise':0.5, 'min_quantity':1},
             'MOEX':{'precise':0.01, 'min_quantity':10},
             'MTLR':{'precise':0.01, 'min_quantity':10},
             'MTSS':{'precise':0.05, 'min_quantity':10},
             'MVID':{'precise':0.1, 'min_quantity':10},
             'NLMK':{'precise':0.02, 'min_quantity':10},
             #'NVTK':{'precise':0.001, 'granularity':100},
             'PHOR':{'precise':1.0, 'min_quantity':1},
             'PIKK':{'precise':0.1, 'min_quantity':10},
             'PLZL':{'precise':0.5, 'min_quantity':1},
             'POLY':{'precise':0.1, 'min_quantity':1},
             'RNFT':{'precise':0.2, 'min_quantity':1},
             #'ROSN':{'precise':0.001, 'granularity':100},
             'RTKM':{'precise':0.01, 'min_quantity':10},
             'RUAL':{'precise':0.005, 'min_quantity':10},
             #'SBER':{'precise':0.001, 'granularity':100},
             #'SBERP':{'precise':0.001, 'granularity':100},
             'SFIN':{'precise':0.2, 'min_quantity':10},
             'SNGS':{'precise':0.005, 'min_quantity':100},
             'SNGSP':{'precise':0.005, 'min_quantity':100},
             'TATN':{'precise':0.1, 'min_quantity':10},
             'TATNP':{'precise':0.1, 'min_quantity':10},
             #'TGKD':{'precise':0.000005, 'granularity':100000},
             'TRMK':{'precise':0.02, 'min_quantity':10},
             #'TRNFP':{'precise':0.001, 'granularity':100},
             #'VTBR':{'precise':0.001, 'granularity':100},
             'YNDX':{'precise':0.2, 'min_quantity':1},
            }



top_tickers = ['ROSN', 'MGNT', 'ALRS', 'SNGSP', 'VTBR', 'SBERP', 'SBER', 'MOEX',
       'RUAL', 'NLMK', 'HYDR', 'MTSS', 'AFKS', 'IRAO', 'TATN', 'NVTK',
       'PLZL', 'SNGS', 'FEES', 'SIBN', 'RSTI', 'BANEP', 'LKOH', 'CHMF',
       'POLY', 'GMKN', 'YNDX', 'LSRG', 'MAGN', 'TATNP', 'ENRU', 'MRKP',
       'OGKB', 'LNTA', 'UPRO', 'RASP', 'MTLRP', 'PHOR', 'TRMK', 'CBOM',
       'APTK', 'MRKV', 'MSST', 'LSNGP', 'TRNFP', 'PIKK', 'NKNCP', 'RTKMP',
       'SIBG', 'KBTK', 'MRKC', 'URKA', 'TANL', 'MSTT', 'RUGR', 'MRKY',
       'MVID', 'GCHE', 'NMTP', 'AKRN', 'RLMN', 'GRNT', 'MRKZ', 'MSRS',
       'MRKK', 'PMSB', 'BLNG', 'TTLK', 'RBCM', 'UWGN', 'NKNC', 'TRCN',
       'KLSB', 'IRKT', 'DASB', 'RGSS', 'PMSBP', 'TGKD', 'SVAV', 'TUCH',
       'MRKS', 'PRFN', 'LSNG', 'MFON', 'ROLO', 'CNTL', 'VSMO', 'MRKU',
       'UCSS', 'PLSM', 'ALNU', 'ASSB', 'FESH', 'AMEZ', 'DVEC', 'KRKNP',
       'MGVM', 'KMAZ', 'VLHZ', 'VZRZP', 'ELTZ', 'MOBB', 'KUZB', 'IRGZ',
       'ROST', 'VDSB', 'ISKJ', 'STSB', 'KMEZ', 'NKHP', 'USBN', 'LNZL',
       'RLMNP', 'NAUK', 'PRTK', 'NNSBP', 'SELG', 'ABRD', 'LNZLP', 'CHEP',
       'DIOD', 'STSBP', 'SZPR', 'BRZL', 'CLSBP', 'UTAR', 'KZOSP', 'AQUA',
       'KAZT', 'VGSBP', 'SYNG', 'NNSB', 'MGTS', 'NFAZ', 'HIMCP', 'TGKDP',
       'VGSB', 'VJGZ', 'SELGP', 'ZILL', 'VRSBP', 'RKKE', 'NSVZ', 'ODVA',
       'WTCM', 'KROT', 'RUSP', 'CHMK', 'MGNZ', 'OMZZP', 'GAZA', 'KUBE',
       'UKUZ', 'KAZTP', 'ZVEZ', 'TGKBP', 'TASBP', 'YAKG', 'NKSH', 'GAZAP',
       'BISVP', 'MRSB', 'HALS', 'SAREP', 'AVAN', 'VSYD', 'YKENP', 'YRSBP',
       'SAGOP', 'MAGE', 'TORSP', 'PRMB', 'KROTP', 'URKZ', 'MAGEP', 'UNKL',
       'SAGO', 'ALBK', 'YRSB', 'MFGS', 'KRSB', 'MFGSP', 'RZSB', 'VJGZP',
       'RTSB', 'ARSA', 'CHGZ', 'KTSBP', 'JNOS', 'LPSB', 'TORS', 'WTCMP',
       'RTGZ', 'TGKN', 'KRSBP', 'KCHEP', 'DZRD', 'KRKN', 'MISBP', 'MISB',
       'PAZA']


